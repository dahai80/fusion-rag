from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool

from .._validators import ValidationError, validate_identifier, validate_path_under_root
from ..engine.chunker import Chunker
from ..engine.contextualizer import Contextualizer
from ..engine.document import CONTENT_TYPE_MAP, DocumentParser, DocumentType, ParseResult
from ..engine.llm_errors import LLMUnavailable
from ..store.metadata_store import MetadataStore
from ..store.vector_store import VectorStore
from .access import require_kb_action
from .app_state import (
    _persist_watches,
    _watch_cap,
    get_embed_client,
    get_kb_locks,
    get_kb_manager,
    get_tasks,
    get_watches,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["documents"])

# _doc_parser is stateless and safe to share across requests.
_doc_parser = DocumentParser()

# 硬伤1: per-app state (_kb_manager / _embed_client / _tasks / _watches /
# _kb_locks) now lives on app.state, read per-request via a contextvar. The
# mutable dicts (tasks/watches/locks) are per-app so a reload rebuilds them
# fresh instead of carrying stale module globals across app instances.
_get_kb_manager = get_kb_manager
_get_embed_client = get_embed_client


def _kb_lock(kb_id: str) -> asyncio.Lock:
    # L12: per-KB lock guarding the read-modify-write of file_count/chunk_count.
    # Concurrent ingest on the same KB otherwise loses updates (both read N,
    # both write N+1). Keyed by kb_id; lazily created in app.state.kb_locks.
    locks = get_kb_locks()
    lock = locks.get(kb_id)
    if lock is None:
        lock = asyncio.Lock()
        locks[kb_id] = lock
    return lock


def _get_base(kb_id: str):
    # F12: confine kb_id before any path construction.
    try:
        validate_identifier(kb_id, field="kb_id")
    except ValueError:
        raise HTTPException(400, f"Invalid kb_id: {kb_id}")
    try:
        return _get_kb_manager().get(kb_id)
    except KeyError:
        raise HTTPException(404, f"Knowledge base '{kb_id}' not found")


def _check_ingest_root(file_path: str) -> None:
    """F15: LFI guard — when FUSION_RAG_INGEST_ROOTS is set, confine reads to it.

    Local-first tool: an unset env means no confinement (operator opted into
    unrestricted local indexing). When set (colon-separated), every ingest path
    must resolve under one of the configured roots, blocking /etc/passwd etc.
    """
    roots_env = os.environ.get("FUSION_RAG_INGEST_ROOTS", "").strip()
    if not roots_env:
        return
    roots = [r.strip() for r in roots_env.split(":") if r.strip()]
    if not roots:
        return
    for root in roots:
        try:
            validate_path_under_root(file_path, root=root, field="file_path")
            return
        except ValidationError:
            continue
    logger.warning("ingest path rejected by root confinement: %s", file_path)
    raise HTTPException(403, "file_path not under a configured ingest root")


def _get_vector_store(kb_id: str) -> VectorStore:
    kb = _get_base(kb_id)
    backend_type = os.environ.get("FUSION_RAG_STORE_BACKEND", "local")
    # 硬伤A: reuse one pooled backend handle per KB (see app_state
    # get_or_create_vec_store). Prior per-request construction leaked handles.
    from .app_state import get_or_create_vec_store

    return get_or_create_vec_store(kb.vector_path, backend_type)


def _get_meta_store(kb_id: str) -> MetadataStore:
    kb = _get_base(kb_id)
    return MetadataStore(kb.metadata_path)


def _check_embed_model(kb_id: str) -> None:
    # D7: per-KB embedding_model was decorative — server builds ONE
    # EmbeddingClient from FUSION_RAG_EMBED at startup, and the embed path
    # never read kb.config.embedding_model. A KB configured with a different
    # model silently got the service-wide model's vectors (wrong dimension on
    # a real model switch → mixed-dimension recall corruption). Fail visibly
    # instead: reject ingest when the KB's configured model differs from the
    # service-wide client. All KBs share one service-wide embedding model.
    embed = _get_embed_client()
    kb = _get_base(kb_id)
    if kb.config.embedding_model and kb.config.embedding_model != embed.model:
        logger.error(
            "ingest rejected: KB %s configured embedding_model=%r but service runs %r",
            kb_id,
            kb.config.embedding_model,
            embed.model,
        )
        raise HTTPException(
            400,
            (
                f"embedding_model mismatch: KB '{kb_id}' configured '{kb.config.embedding_model}' "
                f"but the service runs a single embedding model '{embed.model}'. "
                f"All KBs share one service-wide model (set FUSION_RAG_EMBED at startup); "
                f"re-create the KB with embedding_model='{embed.model}' to ingest."
            ),
        )


async def _write_doc_to_stores(
    *,
    vec_store: VectorStore,
    meta_store: MetadataStore,
    doc_id: str,
    doc_path: str,
    doc_name: str,
    doc_type: str,
    file_size: int,
    metadata: dict,
    chunks: list,
    vectors: list,
    chunk_dicts: list,
    chars: int,
    chunk_metadata: dict | None = None,
) -> tuple[bool, str | None, bool]:
    """P4-1 + 硬伤B: the ONE place that writes a doc's chunks to all stores.

    R1: every sync store call (add_batch / delete_by_doc / add_document /
    add_chunk / delete_chunks_by_doc / delete_document) runs in a threadpool so
    a slow LanceDB/jieba/SQLite write never freezes the event loop — aligns the
    write path with the search path (routes_search already wraps vec_store.search).

    H2: a rollback that ALSO fails is no longer swallowed. The prior code logged
    rollback exceptions and returned (False, err) with no signal that orphan
    vectors were left behind — an operator saw a 500, assumed "write failed so
    nothing landed", while half the vectors were already durable and unowned.
    Now returns a 3-tuple (ok, err, rollback_failed): callers map
    rollback_failed=True to 503 + an orphan-data audit warning so it is
    discoverable and cleanable. A clean rollback keeps rollback_failed=False.

    Returns (True, None, False) on success; (False, err, rb) on failure.
    """
    records = []
    for i, (chunk, vector, cd) in enumerate(zip(chunks, vectors, chunk_dicts)):
        chunk_id = f"{doc_id}_{i}"
        records.append(
            {
                "id": chunk_id,
                "vector": vector,
                "text": chunk.text,
                "doc_path": doc_path,
                "doc_name": doc_name,
                "doc_type": doc_type,
                "chunk_index": i,
                "metadata": metadata,
                "context": cd.get("context", ""),
            }
        )

    # 硬伤B: write the hard-undo side (vectors + bm25) FIRST.
    rollback_failed = False
    try:
        # P0-1: add_batch writes vectors AND bm25 in one call (both backends
        # do the bm25 add inside add_batch). One write, one rollback path.
        # R1: run in threadpool — add_batch does LanceDB batch insert + jieba
        # tokenization, both sync and slow on large docs.
        await run_in_threadpool(vec_store.add_batch, records)
    except Exception as e:
        logger.error("_write_doc_to_stores: add_batch (vectors+bm25) failed for doc %s: %s", doc_id, e)
        try:
            await run_in_threadpool(vec_store.delete_by_doc, doc_path)
        except Exception as de:
            # H2: rollback failed — orphan vectors may be durable. Signal it.
            rollback_failed = True
            logger.error(
                "_write_doc_to_stores: rollback delete_by_doc failed for %s: %s "
                "(ORPHAN VECTORS LIKELY DURABLE — operator must clean up)",
                doc_id,
                de,
            )
        return (False, f"index write failed: {e}", rollback_failed)

    # Metadata SECOND — cheap, atomic in SQLite. If this throws, the vectors
    # are already durable, so roll them back too and return a clean failure
    # (the caller decides whether to retry). A crash BETWEEN add_batch and
    # here leaves orphan vectors — recoverable by re-ingest over doc_path, the
    # strictly better of the two orphan kinds per 硬伤B.
    try:
        await run_in_threadpool(
            meta_store.add_document, doc_id, doc_path, doc_name, doc_type, file_size, metadata=metadata
        )
        for i, (chunk, cd) in enumerate(zip(chunks, chunk_dicts)):
            chunk_id = f"{doc_id}_{i}"
            per_chunk_meta = chunk_metadata if chunk_metadata is not None else metadata
            await run_in_threadpool(
                meta_store.add_chunk, chunk_id, doc_id, doc_path, i, chunk.text, chunk.tokens, metadata=per_chunk_meta
            )
        await run_in_threadpool(meta_store.update_chunk_count, doc_id, len(chunks), chars)
    except Exception as e:
        logger.error("_write_doc_to_stores: metadata write failed for doc %s (rolling back vectors): %s", doc_id, e)
        try:
            await run_in_threadpool(vec_store.delete_by_doc, doc_path)
        except Exception as de:
            rollback_failed = True
            logger.error(
                "_write_doc_to_stores: metadata-rollback delete_by_doc failed for %s: %s "
                "(ORPHAN VECTORS LIKELY DURABLE)",
                doc_id,
                de,
            )
        try:
            await run_in_threadpool(meta_store.delete_chunks_by_doc, doc_id)
            await run_in_threadpool(meta_store.delete_document, doc_id)
        except Exception as de:
            # Metadata rollback failure is less severe (cheap, row-atomic) but
            # still an orphan signal — a doc row may point at deleted vectors.
            rollback_failed = True
            logger.error(
                "_write_doc_to_stores: metadata-rollback meta cleanup failed for %s: %s (ORPHAN META ROW)",
                doc_id,
                de,
            )
        return (False, f"metadata write failed: {e}", rollback_failed)

    logger.info("_write_doc_to_stores: doc %s indexed, %d chunks", doc_id, len(chunks))
    return (True, None, False)


async def _index_document(kb_id: str, file_path: str, contextualize: bool = True) -> dict:
    kb = _get_base(kb_id)
    # D7: reject before any parse/embed work if the KB's configured embedding
    # model differs from the service-wide client (would persist cross-model
    # vectors). Returns the dict-error contract this internal helper uses.
    embed = _get_embed_client()
    if kb.config.embedding_model and kb.config.embedding_model != embed.model:
        logger.error(
            "ingest rejected: KB %s configured embedding_model=%r but service runs %r",
            kb_id,
            kb.config.embedding_model,
            embed.model,
        )
        return {
            "error": (
                f"embedding_model mismatch: KB '{kb_id}' configured '{kb.config.embedding_model}' "
                f"but the service runs a single embedding model '{embed.model}'. "
                f"All KBs share one service-wide model (set FUSION_RAG_EMBED at startup)."
            ),
            "status_code": 400,
        }
    # F15: LFI guard — reject paths escaping the configured ingest root.
    _check_ingest_root(file_path)
    result = await _doc_parser.parse(file_path)
    if result.error:
        return {"error": result.error}

    chunker = Chunker(strategy=kb.config.chunk_strategy, chunk_size=kb.config.chunk_size)
    chunks = await chunker.chunk(result)
    if not chunks:
        return {"error": "no chunks produced"}

    contextualizer = Contextualizer(enabled=contextualize, api_key=embed.api_key)
    chunk_dicts = [{"id": f"tmp_{i}", "text": c.text} for i, c in enumerate(chunks)]
    # L1: contextualizer raises LLMUnavailable only when EVERY chunk fails. On
    # total LLM failure, contextualize=True silently degraded retrieval before.
    # Decision: log + proceed with empty context (ingest must not fail-closed
    # on an enhancement); the route surfaces this via the contextualize flag.
    try:
        chunk_dicts = await contextualizer.contextualize(chunk_dicts, result.content)
    except LLMUnavailable as e:
        logger.warning("index_document: contextualization fully unavailable, ingesting without context: %s", e)
        for cd in chunk_dicts:
            cd.setdefault("context", "")

    embed_texts = []
    for cd, c in zip(chunk_dicts, chunks):
        ctx = cd.get("context", "")
        embed_texts.append((ctx + " " + c.text).strip() if ctx else c.text)
    vectors = await embed.embed_batch(embed_texts)

    vec_store = _get_vector_store(kb_id)
    meta_store = _get_meta_store(kb_id)

    doc_id = uuid.uuid4().hex[:16]
    # P4-1 + 硬伤B: single write path. vectors-first ordering (hard-undo side
    # before cheap metadata) with unified rollback lives in _write_doc_to_stores.
    ok, err, rollback_failed = await _write_doc_to_stores(
        vec_store=vec_store,
        meta_store=meta_store,
        doc_id=doc_id,
        doc_path=result.file_path,
        doc_name=result.file_name,
        doc_type=result.doc_type.value,
        file_size=result.metadata.get("size", 0),
        metadata=result.metadata,
        chunks=chunks,
        vectors=vectors,
        chunk_dicts=chunk_dicts,
        chars=result.chars,
    )
    if not ok:
        # H2: rollback_failed means orphan vectors/meta may be durable. Signal
        # 503 (not 500) so ops distinguish "clean retry" from "needs cleanup".
        code = 503 if rollback_failed else 500
        msg = err
        if rollback_failed:
            msg = f"{err} [ROLLBACK FAILED — orphan data likely durable, grep log for ORPHAN doc_id={doc_id}]"
        return {"error": msg, "status_code": code, "rollback_failed": rollback_failed}

    # L12: per-KB lock so concurrent ingest can't both read N and write N+1.
    async with _kb_lock(kb_id):
        kb.file_count += 1
        kb.chunk_count += len(chunks)
        _get_kb_manager().update(kb_id, file_count=kb.file_count, chunk_count=kb.chunk_count)

    return {"doc_id": doc_id, "chunks": len(chunks), "chars": result.chars}


@router.post("/bases/{kb_id}/documents", dependencies=[Depends(require_kb_action("write"))])
async def upload_document(kb_id: str, data: dict[str, Any]) -> dict[str, Any]:
    file_path = data.get("file_path", "")
    if not file_path:
        raise HTTPException(400, "file_path is required")
    result = await _index_document(kb_id, file_path, contextualize=data.get("contextualize", True))
    if "error" in result:
        raise HTTPException(result.get("status_code", 400), result["error"])
    return result


@router.post("/bases/{kb_id}/documents/batch", dependencies=[Depends(require_kb_action("write"))])
async def batch_upload_documents(kb_id: str, data: dict[str, Any]) -> dict[str, Any]:
    file_paths = data.get("file_paths", [])
    if not file_paths:
        raise HTTPException(400, "file_paths is required")

    contextualize = data.get("contextualize", True)
    total_chunks = 0
    total_chars = 0
    indexed = []
    errors = []

    for fp in file_paths:
        if not os.path.isfile(fp):
            errors.append({"file": fp, "error": "file not found"})
            continue
        try:
            result = await _index_document(kb_id, fp, contextualize=contextualize)
            if "error" in result:
                errors.append({
                    "file": fp,
                    "error": result["error"],
                    "rollback_failed": result.get("rollback_failed", False),
                })
                continue
            indexed.append({"doc_id": result["doc_id"], "file": fp, "chunks": result["chunks"]})
            total_chunks += result["chunks"]
            total_chars += result["chars"]
        except Exception as e:
            logger.error("batch upload: failed %s: %s", fp, e)
            errors.append({"file": fp, "error": str(e)})

    return {
        "indexed": len(indexed),
        "total_chunks": total_chunks,
        "total_chars": total_chars,
        "documents": indexed,
        "errors": errors,
    }


@router.post("/bases/{kb_id}/documents/ingest", dependencies=[Depends(require_kb_action("write"))])
async def ingest_content(kb_id: str, data: dict[str, Any]) -> dict[str, Any]:
    kb = _get_base(kb_id)
    content = data.get("content", "")
    if not content:
        raise HTTPException(400, "content is required")
    content_type = data.get("content_type", "text")
    metadata = data.get("metadata", {})
    doc_name = data.get("doc_name", f"inline_{uuid.uuid4().hex[:8]}.{content_type}")
    doc_path = f"inline://{doc_name}"

    # D5: reuse the single content-type map on document.py instead of a
    # second divergent copy here (a new type added to EXTENSION_MAP/CONTENT_TYPE_MAP
    # is now visible at the inline-ingest entry too).
    doc_type_enum = CONTENT_TYPE_MAP.get(content_type, DocumentType.TXT)
    parse_result = ParseResult(
        file_path=doc_path,
        file_name=doc_name,
        content=content,
        chars=len(content),
        doc_type=doc_type_enum,
        metadata={},
    )

    chunker = Chunker(strategy=kb.config.chunk_strategy, chunk_size=kb.config.chunk_size)
    chunks = await chunker.chunk(parse_result)

    contextualize = data.get("contextualize", True)
    _check_embed_model(kb_id)
    embed = _get_embed_client()
    contextualizer = Contextualizer(enabled=contextualize, api_key=embed.api_key)
    chunk_dicts = [{"id": f"tmp_{i}", "text": c.text} for i, c in enumerate(chunks)]
    chunk_dicts = await contextualizer.contextualize(chunk_dicts, content)

    embed_texts = []
    for cd, c in zip(chunk_dicts, chunks):
        ctx = cd.get("context", "")
        embed_texts.append((ctx + " " + c.text).strip() if ctx else c.text)
    vectors = await embed.embed_batch(embed_texts)

    vec_store = _get_vector_store(kb_id)
    meta_store = _get_meta_store(kb_id)

    doc_id = uuid.uuid4().hex[:16]
    # P4-1 + 硬伤B: single write path, vectors-first. chunk_metadata=metadata
    # so the inline content's user metadata lands on each chunk row too.
    ok, err, rollback_failed = await _write_doc_to_stores(
        vec_store=vec_store,
        meta_store=meta_store,
        doc_id=doc_id,
        doc_path=doc_path,
        doc_name=doc_name,
        doc_type=content_type,
        file_size=len(content.encode("utf-8")),
        metadata=metadata,
        chunks=chunks,
        vectors=vectors,
        chunk_dicts=chunk_dicts,
        chars=len(content),
        chunk_metadata=metadata,
    )
    if not ok:
        # H2: 503 on rollback_failed (orphan data durable), else 500.
        code = 503 if rollback_failed else 500
        msg = err
        if rollback_failed:
            msg = f"{err} [ROLLBACK FAILED — orphan data likely durable, grep log for ORPHAN doc_id={doc_id}]"
        raise HTTPException(code, msg)

    # L12: per-KB lock around the count read-modify-write.
    async with _kb_lock(kb_id):
        kb.file_count += 1
        kb.chunk_count += len(chunks)
        _get_kb_manager().update(kb_id, file_count=kb.file_count, chunk_count=kb.chunk_count)

    logger.info("ingested inline content %s into kb %s: %d chunks", doc_id, kb_id, len(chunks))
    return {"doc_id": doc_id, "chunks": len(chunks), "chars": len(content)}


_doc_delete_dep = Depends(require_kb_action("delete", resolve_doc_path=True))


@router.delete("/bases/{kb_id}/documents/{doc_id}", dependencies=[_doc_delete_dep])
async def delete_document(kb_id: str, doc_id: str) -> dict[str, Any]:
    kb = _get_base(kb_id)
    meta_store = _get_meta_store(kb_id)
    vec_store = _get_vector_store(kb_id)

    doc = await run_in_threadpool(meta_store.get_document, doc_id)
    if not doc:
        raise HTTPException(404, f"Document '{doc_id}' not found")

    doc_path = doc.get("file_path", "")
    chunks = await run_in_threadpool(meta_store.get_chunks_by_doc, doc_id)
    chunk_count = len(chunks)

    # P0-8/P1-1: the three-store delete (vectors → chunks → doc) had no
    # try/except and no lock. A partial failure left orphan vectors with no
    # metadata (or vice-versa), and concurrent delete/ingest raced the count
    # read-modify-write. Now: lock the KB, delete under try/except, and on
    # failure surface 500 instead of reporting a half-deleted doc as "deleted".
    # R1: every store call runs in a threadpool — delete_by_doc walks LanceDB +
    # jieba BM25 synchronously and would freeze the event loop on large docs.
    async with _kb_lock(kb_id):
        try:
            await run_in_threadpool(vec_store.delete_by_doc, doc_path)
        except Exception as e:
            logger.error("delete_document: vector delete_by_doc failed for %s: %s", doc_id, e)
            raise HTTPException(500, f"vector delete failed: {e}") from e
        try:
            await run_in_threadpool(meta_store.delete_chunks_by_doc, doc_id)
            await run_in_threadpool(meta_store.delete_document, doc_id)
        except Exception as e:
            logger.error("delete_document: metadata delete failed for %s: %s", doc_id, e)
            raise HTTPException(500, f"metadata delete failed: {e}") from e

        kb.file_count = max(0, kb.file_count - 1)
        kb.chunk_count = max(0, kb.chunk_count - chunk_count)
        _get_kb_manager().update(kb_id, file_count=kb.file_count, chunk_count=kb.chunk_count)

    logger.info("deleted doc %s from kb %s: %d chunks removed", doc_id, kb_id, chunk_count)
    return {"doc_id": doc_id, "status": "deleted", "chunks_removed": chunk_count}


_doc_write_dep = Depends(require_kb_action("write", resolve_doc_path=True))


@router.put("/bases/{kb_id}/documents/{doc_id}", dependencies=[_doc_write_dep])
async def replace_document(kb_id: str, doc_id: str, data: dict[str, Any]) -> dict[str, Any]:
    kb = _get_base(kb_id)
    meta_store = _get_meta_store(kb_id)
    vec_store = _get_vector_store(kb_id)

    doc = await run_in_threadpool(meta_store.get_document, doc_id)
    if not doc:
        raise HTTPException(404, f"Document '{doc_id}' not found")

    new_file_path = data.get("file_path", "")
    if not new_file_path:
        raise HTTPException(400, "file_path is required for replacement")

    # L11: parse + chunk + embed the NEW file BEFORE deleting the old index.
    # Before: delete-then-parse left an orphan doc row with zero chunks (and
    # zero vectors) if the new file failed to parse — destroying the existing
    # index irrecoverably. Now a parse/embed failure leaves the old index intact.
    result = await _doc_parser.parse(new_file_path)
    if result.error:
        raise HTTPException(400, result.error)

    chunker = Chunker(strategy=kb.config.chunk_strategy, chunk_size=kb.config.chunk_size)
    chunks = await chunker.chunk(result)
    if not chunks:
        raise HTTPException(400, "replacement file produced no chunks (keeping existing index)")

    contextualize = data.get("contextualize", True)
    _check_embed_model(kb_id)
    embed = _get_embed_client()
    contextualizer = Contextualizer(enabled=contextualize, api_key=embed.api_key)
    chunk_dicts = [{"id": f"tmp_{i}", "text": c.text} for i, c in enumerate(chunks)]
    try:
        chunk_dicts = await contextualizer.contextualize(chunk_dicts, result.content)
    except LLMUnavailable as e:
        logger.warning("replace_document: contextualization unavailable, replacing without context: %s", e)
        for cd in chunk_dicts:
            cd.setdefault("context", "")
    embed_texts = []
    for cd, c in zip(chunk_dicts, chunks):
        ctx = cd.get("context", "")
        embed_texts.append((ctx + " " + c.text).strip() if ctx else c.text)
    vectors = await embed.embed_batch(embed_texts)

    # New content prepared successfully — now swap. Old index is removed, new
    # index written. On a vector/bm25 write failure the old index is already
    # gone (unavoidable for a replace); _write_doc_to_stores rolls back the new
    # write so the KB stays consistent (old gone, new gone, metadata clean).
    old_path = doc.get("file_path", "")
    old_chunks = await run_in_threadpool(meta_store.get_chunks_by_doc, doc_id)
    old_chunk_count = len(old_chunks)
    # R1: wrap the old-index teardown in a threadpool — delete_by_doc walks
    # LanceDB + jieba BM25 synchronously.
    await run_in_threadpool(vec_store.delete_by_doc, old_path)
    await run_in_threadpool(meta_store.delete_chunks_by_doc, doc_id)
    await run_in_threadpool(meta_store.delete_document, doc_id)

    # P4-1 + 硬伤B: single write path, vectors-first. doc_id reused for replace.
    ok, err, rollback_failed = await _write_doc_to_stores(
        vec_store=vec_store,
        meta_store=meta_store,
        doc_id=doc_id,
        doc_path=result.file_path,
        doc_name=result.file_name,
        doc_type=result.doc_type.value,
        file_size=result.metadata.get("size", 0),
        metadata=result.metadata,
        chunks=chunks,
        vectors=vectors,
        chunk_dicts=chunk_dicts,
        chars=result.chars,
    )
    if not ok:
        # H2: 503 on rollback_failed, else 500. On replace the old index is
        # already gone, so an orphan here means new vectors landed unowned.
        code = 503 if rollback_failed else 500
        msg = err
        if rollback_failed:
            msg = f"{err} [ROLLBACK FAILED — orphan data likely durable, grep log for ORPHAN doc_id={doc_id}]"
        raise HTTPException(code, msg)

    # L12: guard the read-modify-write of chunk_count under the per-KB lock.
    async with _kb_lock(kb_id):
        kb.chunk_count = max(0, kb.chunk_count - old_chunk_count + len(chunks))
        _get_kb_manager().update(kb_id, file_count=kb.file_count, chunk_count=kb.chunk_count)

    logger.info("replaced doc %s in kb %s: %d -> %d chunks", doc_id, kb_id, old_chunk_count, len(chunks))
    return {
        "doc_id": doc_id,
        "status": "replaced",
        "old_chunks": old_chunk_count,
        "new_chunks": len(chunks),
        "chars": result.chars,
    }


_doc_read_dep = Depends(require_kb_action("read", resolve_doc_path=True))


@router.get("/bases/{kb_id}/documents/{doc_id}/status", dependencies=[_doc_read_dep])
async def document_status(kb_id: str, doc_id: str) -> dict[str, Any]:
    _get_base(kb_id)
    meta_store = _get_meta_store(kb_id)

    doc = await run_in_threadpool(meta_store.get_document, doc_id)
    if not doc:
        task = get_tasks().get(doc_id)
        if task and task.get("kb_id") == kb_id:
            return {
                "doc_id": doc_id,
                "status": task.get("status", "unknown"),
                "progress": task.get("progress"),
                "error": task.get("error"),
            }
        raise HTTPException(404, f"Document '{doc_id}' not found")

    chunks = await run_in_threadpool(meta_store.get_chunks_by_doc, doc_id)
    return {
        "doc_id": doc_id,
        "status": "indexed",
        "file_path": doc.get("file_path", ""),
        "file_name": doc.get("file_name", ""),
        "chunk_count": len(chunks),
        "chars": doc.get("chars", 0),
    }


@router.get("/bases/{kb_id}/documents", dependencies=[Depends(require_kb_action("read"))])
async def list_documents(kb_id: str) -> list[dict[str, Any]]:
    _get_base(kb_id)
    meta_store = _get_meta_store(kb_id)
    # R1: list_documents scans the SQLite documents table synchronously — wrap
    # so a large KB doesn't block the event loop (concurrent search/list queue).
    return await run_in_threadpool(meta_store.list_documents)


@router.post("/bases/{kb_id}/scan", dependencies=[Depends(require_kb_action("write"))])
async def scan_directory(kb_id: str, data: dict[str, Any]) -> dict[str, Any]:
    kb = _get_base(kb_id)
    dir_path = data.get("dir_path", "")
    if not dir_path:
        raise HTTPException(400, "dir_path is required")
    # F15: LFI guard on the scanned directory root.
    _check_ingest_root(dir_path)

    # D4: scan file cap was hardcoded 1000; RuntimeConfig makes it env-tunable.
    from ..engine.runtime_config import get_runtime_config

    results = await _doc_parser.parse_directory(
        dir_path, recursive=True, max_files=get_runtime_config().scan_max_files
    )
    _check_embed_model(kb_id)
    embed = _get_embed_client()
    vec_store = _get_vector_store(kb_id)
    meta_store = _get_meta_store(kb_id)
    contextualize = data.get("contextualize", True)
    contextualizer = Contextualizer(enabled=contextualize, api_key=embed.api_key)

    total_chunks = 0
    total_chars = 0
    file_count = 0
    errors = []

    for result in results:
        if result.error:
            errors.append({"file": result.file_name, "error": result.error})
            continue

        chunker = Chunker(strategy=kb.config.chunk_strategy, chunk_size=kb.config.chunk_size)
        chunks = await chunker.chunk(result)
        if not chunks:
            continue

        chunk_dicts = [{"id": f"tmp_{i}", "text": c.text} for i, c in enumerate(chunks)]
        try:
            chunk_dicts = await contextualizer.contextualize(chunk_dicts, result.content)
        except LLMUnavailable as e:
            logger.warning(
                "scan: contextualization unavailable for %s, indexing without context: %s",
                result.file_name,
                e,
            )
            for cd in chunk_dicts:
                cd.setdefault("context", "")

        embed_texts = []
        for cd, c in zip(chunk_dicts, chunks):
            ctx = cd.get("context", "")
            embed_texts.append((ctx + " " + c.text).strip() if ctx else c.text)
        vectors = await embed.embed_batch(embed_texts)

        doc_id = uuid.uuid4().hex[:16]
        # P4-1 + 硬伤B: single write path, vectors-first, unified rollback.
        # On failure this file is rolled back clean; continue scanning the rest.
        ok, err, rollback_failed = await _write_doc_to_stores(
            vec_store=vec_store,
            meta_store=meta_store,
            doc_id=doc_id,
            doc_path=result.file_path,
            doc_name=result.file_name,
            doc_type=result.doc_type.value,
            file_size=result.metadata.get("size", 0),
            metadata=result.metadata,
            chunks=chunks,
            vectors=vectors,
            chunk_dicts=chunk_dicts,
            chars=result.chars,
        )
        if not ok:
            errors.append({
                "file": result.file_name,
                "error": err,
                "rollback_failed": rollback_failed,
            })
            continue
        total_chunks += len(chunks)
        total_chars += result.chars
        file_count += 1

    # L12: per-KB lock around the count read-modify-write.
    async with _kb_lock(kb_id):
        kb.file_count += file_count
        kb.chunk_count += total_chunks
        _get_kb_manager().update(kb_id, file_count=kb.file_count, chunk_count=kb.chunk_count)

    return {
        "files_indexed": file_count,
        "total_chunks": total_chunks,
        "total_chars": total_chars,
        "errors": errors,
    }


def _file_hash(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return ""


async def _watch_loop(watch_id: str) -> None:
    watch = get_watches().get(watch_id)
    if not watch:
        return
    kb_id = watch["kb_id"]
    file_paths = watch["file_paths"]
    interval = watch.get("poll_interval", 30)
    hashes = watch["hashes"]
    logger.info("watch %s: started, %d files, interval=%ds", watch_id, len(file_paths), interval)
    while watch.get("active", False):
        await asyncio.sleep(interval)
        if not watch.get("active", False):
            break
        changed = []
        for fp in file_paths:
            current = _file_hash(fp)
            previous = hashes.get(fp, "")
            if current and current != previous:
                changed.append(fp)
                hashes[fp] = current
        if changed:
            logger.info("watch %s: %d files changed, re-indexing", watch_id, len(changed))
            try:
                for fp in changed:
                    result = await _index_document(kb_id, fp)
                    if "error" in result:
                        logger.warning("watch %s: re-index failed for %s: %s", watch_id, fp, result["error"])
                watch["last_reindex"] = uuid.uuid4().hex[:8]
                watch["changes_detected"] = watch.get("changes_detected", 0) + len(changed)
                logger.info("watch %s: re-indexed %d files", watch_id, len(changed))
            except Exception as e:
                logger.error("watch %s: re-index error: %s", watch_id, e, exc_info=True)
    logger.info("watch %s: stopped", watch_id)


@router.post("/bases/{kb_id}/watch", dependencies=[Depends(require_kb_action("write"))])
async def watch_files(kb_id: str, data: dict[str, Any]) -> dict[str, Any]:
    _get_base(kb_id)
    file_paths = data.get("file_paths", [])
    poll_interval = max(data.get("poll_interval", 30), 10)
    if not file_paths:
        raise HTTPException(400, "file_paths is required")
    # F15: every watched file must sit under the ingest root (re-index reads it).
    valid_paths = []
    for fp in file_paths:
        if not os.path.isfile(fp):
            continue
        _check_ingest_root(fp)
        valid_paths.append(fp)
    if not valid_paths:
        raise HTTPException(400, "No valid file paths provided")
    hashes = {}
    for fp in valid_paths:
        h = _file_hash(fp)
        if h:
            hashes[fp] = h
    # R3: per-KB concurrency cap. Without it a client loop-calling /watch
    # spawned an unbounded number of _watch_loop polling tasks → FD/CPU
    # exhaustion. Reject over-cap with 429 (Rate Limit) + the current count so
    # the caller can back off.
    cap = _watch_cap()
    active_for_kb = [w for w in get_watches().values() if w["kb_id"] == kb_id and w.get("active")]
    if len(active_for_kb) >= cap:
        raise HTTPException(
            429,
            f"watch cap reached for KB '{kb_id}': {len(active_for_kb)}/{cap} active watches. "
            f"Unwatch one before adding another, or raise FUSION_RAG_WATCH_CAP.",
        )
    watch_id = uuid.uuid4().hex[:12]
    watch = {
        "watch_id": watch_id,
        "kb_id": kb_id,
        "file_paths": valid_paths,
        "poll_interval": poll_interval,
        "hashes": hashes,
        "active": True,
        "changes_detected": 0,
    }
    get_watches()[watch_id] = watch
    get_watches()[watch_id]["_task"] = asyncio.create_task(_watch_loop(watch_id))
    # R3: persist the registry so a restart restores watches instead of
    # silently dropping every monitor.
    _persist_watches(get_watches())
    logger.info("watch: created watch_id=%s, %d files", watch_id, len(valid_paths))
    return {"watch_id": watch_id, "file_count": len(valid_paths), "poll_interval": poll_interval}


@router.post("/bases/{kb_id}/unwatch", dependencies=[Depends(require_kb_action("write"))])
async def unwatch_files(kb_id: str, data: dict[str, Any]) -> dict[str, Any]:
    watch_id = data.get("watch_id", "")
    watches = get_watches()
    watch = watches.get(watch_id)
    if not watch or watch["kb_id"] != kb_id:
        raise HTTPException(404, f"Watch not found: {watch_id}")
    watch["active"] = False
    changes = watch.get("changes_detected", 0)
    del watches[watch_id]
    # R3: re-persist so a restart does not re-spawn a just-unwatched monitor.
    _persist_watches(watches)
    return {"stopped": True, "watch_id": watch_id, "changes_detected": changes}


@router.get("/bases/{kb_id}/watch/status", dependencies=[Depends(require_kb_action("read"))])
async def watch_status(kb_id: str) -> dict[str, Any]:
    active = [w for w in get_watches().values() if w["kb_id"] == kb_id and w.get("active")]
    return {
        "active_watches": len(active),
        "watches": [
            {
                "watch_id": w["watch_id"],
                "file_count": len(w["file_paths"]),
                "changes_detected": w.get("changes_detected", 0),
                "last_reindex": w.get("last_reindex"),
            }
            for w in active
        ],
    }
