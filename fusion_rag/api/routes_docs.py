from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from ..embed.client import EmbeddingClient
from ..engine.chunker import Chunker
from ..engine.contextualizer import Contextualizer
from ..engine.document import DocumentParser, DocumentType, ParseResult
from ..engine.knowledge_base import KnowledgeBaseManager
from ..store.metadata_store import MetadataStore
from ..store.vector_store import VectorStore
from .auth import verify_api_key

logger = logging.getLogger(__name__)

router = APIRouter(tags=["documents"])

_kb_manager: KnowledgeBaseManager | None = None
_embed_client: EmbeddingClient | None = None
_doc_parser = DocumentParser()
_tasks: dict[str, dict[str, Any]] = {}
_watches: dict[str, dict[str, Any]] = {}


def set_doc_context(kb_manager: KnowledgeBaseManager, embed_client: EmbeddingClient) -> None:
    global _kb_manager, _embed_client
    _kb_manager = kb_manager
    _embed_client = embed_client


def _get_kb_manager() -> KnowledgeBaseManager:
    if _kb_manager is None:
        raise HTTPException(503, "Knowledge base manager not initialized")
    return _kb_manager


def _get_embed_client() -> EmbeddingClient:
    if _embed_client is None:
        raise HTTPException(503, "Embedding client not initialized")
    return _embed_client


def _get_base(kb_id: str):
    try:
        return _get_kb_manager().get(kb_id)
    except KeyError:
        raise HTTPException(404, f"Knowledge base '{kb_id}' not found")


def _get_vector_store(kb_id: str) -> VectorStore:
    kb = _get_base(kb_id)
    return VectorStore(kb.vector_path)


def _get_meta_store(kb_id: str) -> MetadataStore:
    kb = _get_base(kb_id)
    return MetadataStore(kb.metadata_path)


async def _index_document(kb_id: str, file_path: str, contextualize: bool = True) -> dict:
    kb = _get_base(kb_id)
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
    chunk_dicts = await contextualizer.contextualize(chunk_dicts, result.content)

    embed_texts = []
    for cd, c in zip(chunk_dicts, chunks):
        ctx = cd.get("context", "")
        embed_texts.append((ctx + " " + c.text).strip() if ctx else c.text)
    vectors = await embed.embed_batch(embed_texts)

    vec_store = _get_vector_store(kb_id)
    meta_store = _get_meta_store(kb_id)

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

    old_path = doc.get("file_path", "")
    old_chunks = meta_store.get_chunks_by_doc(doc_id)
    old_chunk_count = len(old_chunks)
    vec_store.delete_by_doc(old_path)
    meta_store.delete_chunks_by_doc(doc_id)

    result = await _doc_parser.parse(new_file_path)
    if result.error:
        raise HTTPException(400, result.error)

    chunker = Chunker(strategy=kb.config.chunk_strategy, chunk_size=kb.config.chunk_size)
    chunks = await chunker.chunk(result)

    contextualize = data.get("contextualize", True)
    embed = _get_embed_client()
    contextualizer = Contextualizer(enabled=contextualize, api_key=embed.api_key)
    chunk_dicts = [{"id": f"tmp_{i}", "text": c.text} for i, c in enumerate(chunks)]
    chunk_dicts = await contextualizer.contextualize(chunk_dicts, result.content)
    embed_texts = []
    for cd, c in zip(chunk_dicts, chunks):
        ctx = cd.get("context", "")
        embed_texts.append((ctx + " " + c.text).strip() if ctx else c.text)
    vectors = await embed.embed_batch(embed_texts)

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
        task = _tasks.get(doc_id)
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
        chunk_dicts = await contextualizer.contextualize(chunk_dicts, result.content)

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
    watch = _watches.get(watch_id)
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


@router.post("/bases/{kb_id}/watch")
async def watch_files(kb_id: str, data: dict[str, Any]) -> dict[str, Any]:
    _get_base(kb_id)
    file_paths = data.get("file_paths", [])
    poll_interval = max(data.get("poll_interval", 30), 10)
    if not file_paths:
        raise HTTPException(400, "file_paths is required")
    valid_paths = [fp for fp in file_paths if os.path.isfile(fp)]
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
    _watches[watch_id] = watch
    _watches[watch_id]["_task"] = asyncio.create_task(_watch_loop(watch_id))
    logger.info("watch: created watch_id=%s, %d files", watch_id, len(valid_paths))
    return {"watch_id": watch_id, "file_count": len(valid_paths), "poll_interval": poll_interval}


@router.post("/bases/{kb_id}/unwatch")
async def unwatch_files(kb_id: str, data: dict[str, Any]) -> dict[str, Any]:
    watch_id = data.get("watch_id", "")
    watch = _watches.get(watch_id)
    if not watch or watch["kb_id"] != kb_id:
        raise HTTPException(404, f"Watch not found: {watch_id}")
    watch["active"] = False
    changes = watch.get("changes_detected", 0)
    del _watches[watch_id]
    return {"stopped": True, "watch_id": watch_id, "changes_detected": changes}


@router.get("/bases/{kb_id}/watch/status")
async def watch_status(kb_id: str) -> dict[str, Any]:
    active = [w for w in _watches.values() if w["kb_id"] == kb_id and w.get("active")]
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
