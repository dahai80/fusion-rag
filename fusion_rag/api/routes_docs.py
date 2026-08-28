from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from .._validators import ValidationError, validate_identifier, validate_path_under_root
from ..engine.chunker import Chunker
from ..engine.contextualizer import Contextualizer
from ..engine.document import DocumentParser, DocumentType, ParseResult
from ..engine.llm_errors import LLMUnavailable
from ..store.metadata_store import MetadataStore
from ..store.vector_store import VectorStore
from .app_state import get_embed_client, get_kb_locks, get_kb_manager, get_tasks, get_watches
from .auth import verify_api_key

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
    return VectorStore(kb.vector_path, backend_type=backend_type)


def _get_meta_store(kb_id: str) -> MetadataStore:
    kb = _get_base(kb_id)
    return MetadataStore(kb.metadata_path)


async def _index_document(kb_id: str, file_path: str, contextualize: bool = True) -> dict:
    kb = _get_base(kb_id)
    # F15: LFI guard — reject paths escaping the configured ingest root.
    _check_ingest_root(file_path)
    result = await _doc_parser.parse(file_path)
    if result.error:
        return {"error": result.error}

    chunker = Chunker(strategy=kb.config.chunk_strategy, chunk_size=kb.config.chunk_size)
    chunks = await chunker.chunk(result)
    if not chunks:
        return {"error": "no chunks produced"}

    embed = _get_embed_client()
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
    # L7: prepare ALL records before any write, then order the three stores
    # (metadata → vectors → bm25) and roll back in reverse on partial failure.
    # Before: a failure mid-sequence left orphan chunks (vectors w/o metadata
    # or metadata w/o vectors), corrupting retrieval and delete.
    meta_store.add_document(
        doc_id, result.file_path, result.file_name, result.doc_type.value, result.metadata.get("size", 0)
    )
    records = []
    for i, (chunk, vector, cd) in enumerate(zip(chunks, vectors, chunk_dicts)):
        chunk_id = f"{doc_id}_{i}"
        meta_store.add_chunk(chunk_id, doc_id, result.file_path, i, chunk.text, chunk.tokens)
        records.append(
            {
                "id": chunk_id,
                "vector": vector,
                "text": chunk.text,
                "doc_path": result.file_path,
                "doc_name": result.file_name,
                "doc_type": result.doc_type.value,
                "chunk_index": i,
                "metadata": result.metadata,
                "context": cd.get("context", ""),
            }
        )
    try:
        vec_store.add_batch(records)
    except Exception as e:
        # L7: vectors failed — roll back metadata writes (doc + chunks).
        logger.error("index_document: vector add_batch failed, rolling back metadata for doc %s: %s", doc_id, e)
        meta_store.delete_chunks_by_doc(doc_id)
        meta_store.delete_document(doc_id)
        return {"error": f"vector index write failed: {e}"}
    try:
        vec_store.bm25.add_documents(records)
    except Exception as e:
        # L7: bm25 failed — roll back vectors + metadata for this doc.
        logger.error("index_document: bm25 add failed, rolling back vectors+metadata for doc %s: %s", doc_id, e)
        vec_store.delete_by_doc(result.file_path)
        meta_store.delete_chunks_by_doc(doc_id)
        meta_store.delete_document(doc_id)
        return {"error": f"bm25 index write failed: {e}"}
    meta_store.update_chunk_count(doc_id, len(chunks), result.chars)

    # L12: per-KB lock so concurrent ingest can't both read N and write N+1.
    async with _kb_lock(kb_id):
        kb.file_count += 1
        kb.chunk_count += len(chunks)
        _get_kb_manager().update(kb_id, file_count=kb.file_count, chunk_count=kb.chunk_count)

    return {"doc_id": doc_id, "chunks": len(chunks), "chars": result.chars}


@router.post("/bases/{kb_id}/documents", dependencies=[Depends(verify_api_key)])
async def upload_document(kb_id: str, data: dict[str, Any]) -> dict[str, Any]:
    file_path = data.get("file_path", "")
    if not file_path:
        raise HTTPException(400, "file_path is required")
    result = await _index_document(kb_id, file_path, contextualize=data.get("contextualize", True))
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@router.post("/bases/{kb_id}/documents/batch", dependencies=[Depends(verify_api_key)])
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
                errors.append({"file": fp, "error": result["error"]})
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


@router.post("/bases/{kb_id}/documents/ingest", dependencies=[Depends(verify_api_key)])
async def ingest_content(kb_id: str, data: dict[str, Any]) -> dict[str, Any]:
    kb = _get_base(kb_id)
    content = data.get("content", "")
    if not content:
        raise HTTPException(400, "content is required")
    content_type = data.get("content_type", "text")
    metadata = data.get("metadata", {})
    doc_name = data.get("doc_name", f"inline_{uuid.uuid4().hex[:8]}.{content_type}")
    doc_path = f"inline://{doc_name}"

    type_map = {
        "markdown": DocumentType.MARKDOWN,
        "html": DocumentType.HTML,
        "csv": DocumentType.TXT,
        "text": DocumentType.TXT,
    }
    doc_type_enum = type_map.get(content_type, DocumentType.TXT)
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
    meta_store.add_document(doc_id, doc_path, doc_name, content_type, len(content.encode("utf-8")), metadata=metadata)

    records = []
    for i, (chunk, vector, cd) in enumerate(zip(chunks, vectors, chunk_dicts)):
        chunk_id = f"{doc_id}_{i}"
        meta_store.add_chunk(chunk_id, doc_id, doc_path, i, chunk.text, chunk.tokens, metadata=metadata)
        records.append(
            {
                "id": chunk_id,
                "vector": vector,
                "text": chunk.text,
                "doc_path": doc_path,
                "doc_name": doc_name,
                "doc_type": content_type,
                "chunk_index": i,
                "metadata": metadata,
                "context": cd.get("context", ""),
            }
        )

    vec_store.add_batch(records)
    vec_store.bm25.add_documents(records)
    meta_store.update_chunk_count(doc_id, len(chunks), len(content))

    # L12: per-KB lock around the count read-modify-write.
    async with _kb_lock(kb_id):
        kb.file_count += 1
        kb.chunk_count += len(chunks)
        _get_kb_manager().update(kb_id, file_count=kb.file_count, chunk_count=kb.chunk_count)

    logger.info("ingested inline content %s into kb %s: %d chunks", doc_id, kb_id, len(chunks))
    return {"doc_id": doc_id, "chunks": len(chunks), "chars": len(content)}


@router.delete("/bases/{kb_id}/documents/{doc_id}", dependencies=[Depends(verify_api_key)])
async def delete_document(kb_id: str, doc_id: str) -> dict[str, Any]:
    kb = _get_base(kb_id)
    meta_store = _get_meta_store(kb_id)
    vec_store = _get_vector_store(kb_id)

    doc = meta_store.get_document(doc_id)
    if not doc:
        raise HTTPException(404, f"Document '{doc_id}' not found")

    doc_path = doc.get("file_path", "")
    chunks = meta_store.get_chunks_by_doc(doc_id)
    chunk_count = len(chunks)

    vec_store.delete_by_doc(doc_path)
    meta_store.delete_chunks_by_doc(doc_id)
    meta_store.delete_document(doc_id)

    kb.file_count = max(0, kb.file_count - 1)
    kb.chunk_count = max(0, kb.chunk_count - chunk_count)
    _get_kb_manager().update(kb_id, file_count=kb.file_count, chunk_count=kb.chunk_count)

    logger.info("deleted doc %s from kb %s: %d chunks removed", doc_id, kb_id, chunk_count)
    return {"doc_id": doc_id, "status": "deleted", "chunks_removed": chunk_count}


@router.put("/bases/{kb_id}/documents/{doc_id}", dependencies=[Depends(verify_api_key)])
async def replace_document(kb_id: str, doc_id: str, data: dict[str, Any]) -> dict[str, Any]:
    kb = _get_base(kb_id)
    meta_store = _get_meta_store(kb_id)
    vec_store = _get_vector_store(kb_id)

    doc = meta_store.get_document(doc_id)
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
    # gone (unavoidable for a replace), but the doc row + new chunks are
    # written so the state is consistent (new doc, fully or partially indexed).
    old_path = doc.get("file_path", "")
    old_chunks = meta_store.get_chunks_by_doc(doc_id)
    old_chunk_count = len(old_chunks)
    vec_store.delete_by_doc(old_path)
    meta_store.delete_chunks_by_doc(doc_id)
    meta_store.delete_document(doc_id)

    meta_store.add_document(
        doc_id, result.file_path, result.file_name, result.doc_type.value, result.metadata.get("size", 0)
    )

    records = []
    for i, (chunk, vector, cd) in enumerate(zip(chunks, vectors, chunk_dicts)):
        chunk_id = f"{doc_id}_{i}"
        meta_store.add_chunk(chunk_id, doc_id, result.file_path, i, chunk.text, chunk.tokens)
        records.append(
            {
                "id": chunk_id,
                "vector": vector,
                "text": chunk.text,
                "doc_path": result.file_path,
                "doc_name": result.file_name,
                "doc_type": result.doc_type.value,
                "chunk_index": i,
                "metadata": result.metadata,
                "context": cd.get("context", ""),
            }
        )

    vec_store.add_batch(records)
    vec_store.bm25.add_documents(records)
    meta_store.update_chunk_count(doc_id, len(chunks), result.chars)

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


@router.get("/bases/{kb_id}/documents/{doc_id}/status")
async def document_status(kb_id: str, doc_id: str) -> dict[str, Any]:
    _get_base(kb_id)
    meta_store = _get_meta_store(kb_id)

    doc = meta_store.get_document(doc_id)
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

    chunks = meta_store.get_chunks_by_doc(doc_id)
    return {
        "doc_id": doc_id,
        "status": "indexed",
        "file_path": doc.get("file_path", ""),
        "file_name": doc.get("file_name", ""),
        "chunk_count": len(chunks),
        "chars": doc.get("chars", 0),
    }


@router.get("/bases/{kb_id}/documents")
async def list_documents(kb_id: str) -> list[dict[str, Any]]:
    _get_base(kb_id)
    meta_store = _get_meta_store(kb_id)
    return meta_store.list_documents()


@router.post("/bases/{kb_id}/scan", dependencies=[Depends(verify_api_key)])
async def scan_directory(kb_id: str, data: dict[str, Any]) -> dict[str, Any]:
    kb = _get_base(kb_id)
    dir_path = data.get("dir_path", "")
    if not dir_path:
        raise HTTPException(400, "dir_path is required")
    # F15: LFI guard on the scanned directory root.
    _check_ingest_root(dir_path)

    results = await _doc_parser.parse_directory(dir_path, recursive=True, max_files=1000)
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
        meta_store.add_document(
            doc_id, result.file_path, result.file_name, result.doc_type.value, result.metadata.get("size", 0)
        )

        records = []
        for i, (chunk, vector, cd) in enumerate(zip(chunks, vectors, chunk_dicts)):
            chunk_id = f"{doc_id}_{i}"
            meta_store.add_chunk(chunk_id, doc_id, result.file_path, i, chunk.text, chunk.tokens)
            records.append(
                {
                    "id": chunk_id,
                    "vector": vector,
                    "text": chunk.text,
                    "doc_path": result.file_path,
                    "doc_name": result.file_name,
                    "doc_type": result.doc_type.value,
                    "chunk_index": i,
                    "metadata": result.metadata,
                    "context": cd.get("context", ""),
                }
            )

        vec_store.add_batch(records)
        vec_store.bm25.add_documents(records)
        meta_store.update_chunk_count(doc_id, len(chunks), result.chars)
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


@router.post("/bases/{kb_id}/watch", dependencies=[Depends(verify_api_key)])
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
    logger.info("watch: created watch_id=%s, %d files", watch_id, len(valid_paths))
    return {"watch_id": watch_id, "file_count": len(valid_paths), "poll_interval": poll_interval}


@router.post("/bases/{kb_id}/unwatch", dependencies=[Depends(verify_api_key)])
async def unwatch_files(kb_id: str, data: dict[str, Any]) -> dict[str, Any]:
    watch_id = data.get("watch_id", "")
    watches = get_watches()
    watch = watches.get(watch_id)
    if not watch or watch["kb_id"] != kb_id:
        raise HTTPException(404, f"Watch not found: {watch_id}")
    watch["active"] = False
    changes = watch.get("changes_detected", 0)
    del watches[watch_id]
    return {"stopped": True, "watch_id": watch_id, "changes_detected": changes}


@router.get("/bases/{kb_id}/watch/status", dependencies=[Depends(verify_api_key)])
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
