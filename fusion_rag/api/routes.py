"""Fusion-RAG HTTP API — FastAPI routes for knowledge base operations.

All model calls go through fusion-mlx HTTP API (/v1/embeddings, /v1/chat/completions).
No direct MLX imports.
"""

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
from ..engine.document import DocumentParser
from ..engine.knowledge_base import KnowledgeBaseManager
from ..engine.query_rewriter import QueryRewriter
from ..engine.reranker import HybridSearch, Reranker
from ..store.metadata_store import MetadataStore
from ..store.vector_store import VectorStore
from .auth import verify_api_key

# Async task tracking for indexing status (#14)
_tasks: dict[str, dict[str, Any]] = {}

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/kb", tags=["knowledge-base"])

# Global instances (set by server.py)
_kb_manager: KnowledgeBaseManager | None = None
_embed_client: EmbeddingClient | None = None
_doc_parser = DocumentParser()


def set_kb_context(kb_manager: KnowledgeBaseManager, embed_client: EmbeddingClient) -> None:
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


# ── Knowledge Base CRUD ──


@router.get("/bases")
async def list_knowledge_bases() -> list[dict[str, Any]]:
    """List all knowledge bases."""
    return _get_kb_manager().list()


@router.post("/bases", dependencies=[Depends(verify_api_key)])
async def create_knowledge_base(data: dict[str, Any]) -> dict[str, Any]:
    """Create a new knowledge base."""
    name = data.get("name", "")
    if not name:
        raise HTTPException(400, "name is required")
    description = data.get("description", "")
    chunk_strategy = data.get("chunk_strategy", "semantic")
    embedding_model = data.get("embedding_model", "BGE-M3")
    kb = _get_kb_manager().create(
        name=name, description=description,
        chunk_strategy=chunk_strategy, embedding_model=embedding_model,
    )
    return {"id": kb.id, "name": kb.config.name, "status": "created"}


@router.get("/bases/{kb_id}")
async def get_knowledge_base(kb_id: str) -> dict[str, Any]:
    """Get knowledge base details."""
    try:
        return _get_kb_manager().get(kb_id).to_dict()
    except KeyError:
        raise HTTPException(404, f"Knowledge base '{kb_id}' not found")


@router.delete("/bases/{kb_id}", dependencies=[Depends(verify_api_key)])
async def delete_knowledge_base(kb_id: str) -> dict[str, str]:
    """Delete a knowledge base."""
    if _get_kb_manager().delete(kb_id):
        return {"id": kb_id, "status": "deleted"}
    raise HTTPException(404, f"Knowledge base '{kb_id}' not found")


# ── Document Operations ──


@router.post("/bases/{kb_id}/documents", dependencies=[Depends(verify_api_key)])
async def upload_document(kb_id: str, data: dict[str, Any]) -> dict[str, Any]:
    """Upload and index a single document."""
    kb = _get_base(kb_id)
    file_path = data.get("file_path", "")
    if not file_path:
        raise HTTPException(400, "file_path is required")

    # Parse document
    result = await _doc_parser.parse(file_path)
    if result.error:
        raise HTTPException(400, result.error)

    # Chunk
    chunker = Chunker(strategy=kb.config.chunk_strategy, chunk_size=kb.config.chunk_size)
    chunks = await chunker.chunk(result)

    # Contextualize (optional, default enabled)
    contextualize = data.get("contextualize", True)
    contextualizer = Contextualizer(enabled=contextualize)
    chunk_dicts = [{"id": f"tmp_{i}", "text": c.text} for i, c in enumerate(chunks)]
    chunk_dicts = await contextualizer.contextualize(chunk_dicts, result.content)

    # Embed (use context+text when available)
    embed = _get_embed_client()
    embed_texts = []
    for cd, c in zip(chunk_dicts, chunks):
        ctx = cd.get("context", "")
        embed_texts.append((ctx + " " + c.text).strip() if ctx else c.text)
    vectors = await embed.embed_batch(embed_texts)

    # Store vectors
    vec_store = _get_vector_store(kb_id)
    meta_store = _get_meta_store(kb_id)

    doc_id = uuid.uuid4().hex[:16]
    meta_store.add_document(doc_id, result.file_path, result.file_name,
                            result.doc_type.value, result.metadata.get("size", 0))

    records = []
    for i, (chunk, vector, cd) in enumerate(zip(chunks, vectors, chunk_dicts)):
        chunk_id = f"{doc_id}_{i}"
        meta_store.add_chunk(chunk_id, doc_id, result.file_path, i, chunk.text, chunk.tokens)
        records.append({
            "id": chunk_id,
            "vector": vector,
            "text": chunk.text,
            "doc_path": result.file_path,
            "doc_name": result.file_name,
            "doc_type": result.doc_type.value,
            "chunk_index": i,
            "metadata": result.metadata,
            "context": cd.get("context", ""),
        })

    vec_store.add_batch(records)
    # Update BM25 index
    vec_store.bm25.add_documents(records)
    meta_store.update_chunk_count(doc_id, len(chunks), result.chars)

    # Update KB stats
    kb.file_count += 1
    kb.chunk_count += len(chunks)
    _get_kb_manager().update(kb_id, file_count=kb.file_count, chunk_count=kb.chunk_count)

    return {"doc_id": doc_id, "chunks": len(chunks), "chars": result.chars}


# #18: Batch file upload
@router.post("/bases/{kb_id}/documents/batch", dependencies=[Depends(verify_api_key)])
async def batch_upload_documents(kb_id: str, data: dict[str, Any]) -> dict[str, Any]:
    """Upload and index multiple documents in batch."""
    kb = _get_base(kb_id)
    file_paths = data.get("file_paths", [])
    if not file_paths:
        raise HTTPException(400, "file_paths is required")

    embed = _get_embed_client()
    vec_store = _get_vector_store(kb_id)
    meta_store = _get_meta_store(kb_id)
    contextualize = data.get("contextualize", True)
    contextualizer = Contextualizer(enabled=contextualize)

    total_chunks = 0
    total_chars = 0
    indexed = []
    errors = []

    for fp in file_paths:
        if not os.path.isfile(fp):
            errors.append({"file": fp, "error": "file not found"})
            continue
        try:
            result = await _doc_parser.parse(fp)
            if result.error:
                errors.append({"file": fp, "error": result.error})
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
            meta_store.add_document(doc_id, result.file_path, result.file_name,
                                    result.doc_type.value, result.metadata.get("size", 0))
            records = []
            for i, (chunk, vector, cd) in enumerate(zip(chunks, vectors, chunk_dicts)):
                chunk_id = f"{doc_id}_{i}"
                meta_store.add_chunk(chunk_id, doc_id, result.file_path, i, chunk.text, chunk.tokens)
                records.append({
                    "id": chunk_id, "vector": vector, "text": chunk.text,
                    "doc_path": result.file_path, "doc_name": result.file_name,
                    "doc_type": result.doc_type.value, "chunk_index": i,
                    "metadata": result.metadata, "context": cd.get("context", ""),
                })
            vec_store.add_batch(records)
            vec_store.bm25.add_documents(records)
            meta_store.update_chunk_count(doc_id, len(chunks), result.chars)
            indexed.append({"doc_id": doc_id, "file": fp, "chunks": len(chunks)})
            total_chunks += len(chunks)
            total_chars += result.chars
        except Exception as e:
            logger.error("batch upload: failed %s: %s", fp, e)
            errors.append({"file": fp, "error": str(e)})

    kb.file_count += len(indexed)
    kb.chunk_count += total_chunks
    _get_kb_manager().update(kb_id, file_count=kb.file_count, chunk_count=kb.chunk_count)

    return {
        "indexed": len(indexed),
        "total_chunks": total_chunks,
        "total_chars": total_chars,
        "documents": indexed,
        "errors": errors,
    }


# #11: Document delete API
@router.delete("/bases/{kb_id}/documents/{doc_id}", dependencies=[Depends(verify_api_key)])
async def delete_document(kb_id: str, doc_id: str) -> dict[str, Any]:
    """Delete a document and its associated chunks/vectors."""
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


# #13: Document replace API (delete + re-index, preserving doc_id)
@router.put("/bases/{kb_id}/documents/{doc_id}", dependencies=[Depends(verify_api_key)])
async def replace_document(kb_id: str, doc_id: str, data: dict[str, Any]) -> dict[str, Any]:
    """Replace a document: delete old vectors, re-index from new file_path."""
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
    contextualizer = Contextualizer(enabled=contextualize)
    chunk_dicts = [{"id": f"tmp_{i}", "text": c.text} for i, c in enumerate(chunks)]
    chunk_dicts = await contextualizer.contextualize(chunk_dicts, result.content)

    embed = _get_embed_client()
    embed_texts = []
    for cd, c in zip(chunk_dicts, chunks):
        ctx = cd.get("context", "")
        embed_texts.append((ctx + " " + c.text).strip() if ctx else c.text)
    vectors = await embed.embed_batch(embed_texts)

    meta_store.delete_document(doc_id)
    meta_store.add_document(doc_id, result.file_path, result.file_name,
                            result.doc_type.value, result.metadata.get("size", 0))

    records = []
    for i, (chunk, vector, cd) in enumerate(zip(chunks, vectors, chunk_dicts)):
        chunk_id = f"{doc_id}_{i}"
        meta_store.add_chunk(chunk_id, doc_id, result.file_path, i, chunk.text, chunk.tokens)
        records.append({
            "id": chunk_id, "vector": vector, "text": chunk.text,
            "doc_path": result.file_path, "doc_name": result.file_name,
            "doc_type": result.doc_type.value, "chunk_index": i,
            "metadata": result.metadata, "context": cd.get("context", ""),
        })

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


# #14: Document indexing status
@router.get("/bases/{kb_id}/documents/{doc_id}/status")
async def document_status(kb_id: str, doc_id: str) -> dict[str, Any]:
    """Get document indexing status."""
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


# List documents in a KB (#14)
@router.get("/bases/{kb_id}/documents")
async def list_documents(kb_id: str) -> list[dict[str, Any]]:
    """List all documents in a knowledge base."""
    _get_base(kb_id)
    meta_store = _get_meta_store(kb_id)
    return meta_store.list_documents()


@router.post("/bases/{kb_id}/scan", dependencies=[Depends(verify_api_key)])
async def scan_directory(kb_id: str, data: dict[str, Any]) -> dict[str, Any]:
    """Scan a directory and index all supported files."""
    kb = _get_base(kb_id)
    dir_path = data.get("dir_path", "")
    if not dir_path:
        raise HTTPException(400, "dir_path is required")

    results = await _doc_parser.parse_directory(dir_path, recursive=True, max_files=1000)
    embed = _get_embed_client()
    vec_store = _get_vector_store(kb_id)
    meta_store = _get_meta_store(kb_id)
    contextualize = data.get("contextualize", True)
    contextualizer = Contextualizer(enabled=contextualize)

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

        # Contextualize
        chunk_dicts = [{"id": f"tmp_{i}", "text": c.text} for i, c in enumerate(chunks)]
        chunk_dicts = await contextualizer.contextualize(chunk_dicts, result.content)

        # Embed with context
        embed_texts = []
        for cd, c in zip(chunk_dicts, chunks):
            ctx = cd.get("context", "")
            embed_texts.append((ctx + " " + c.text).strip() if ctx else c.text)
        vectors = await embed.embed_batch(embed_texts)

        doc_id = uuid.uuid4().hex[:16]
        meta_store.add_document(doc_id, result.file_path, result.file_name,
                                result.doc_type.value, result.metadata.get("size", 0))

        records = []
        for i, (chunk, vector, cd) in enumerate(zip(chunks, vectors, chunk_dicts)):
            chunk_id = f"{doc_id}_{i}"
            meta_store.add_chunk(chunk_id, doc_id, result.file_path, i, chunk.text, chunk.tokens)
            records.append({
                "id": chunk_id, "vector": vector, "text": chunk.text,
                "doc_path": result.file_path, "doc_name": result.file_name,
                "doc_type": result.doc_type.value, "chunk_index": i,
                "metadata": result.metadata,
                "context": cd.get("context", ""),
            })

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


# ── File Watching ──

_watches: dict[str, dict[str, Any]] = {}


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
                kb = _get_base(kb_id)
                embed = _get_embed_client()
                vec_store = _get_vector_store(kb_id)
                meta_store = _get_meta_store(kb_id)
                contextualizer = Contextualizer(enabled=True)
                for fp in changed:
                    results = await _doc_parser.parse_directory(os.path.dirname(fp), recursive=False, max_files=1)
                    for result in results:
                        if result.error:
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
                        meta_store.add_document(doc_id, result.file_path, result.file_name,
                                                result.doc_type.value, result.metadata.get("size", 0))
                        records = []
                        for i, (chunk, vector, cd) in enumerate(zip(chunks, vectors, chunk_dicts)):
                            chunk_id = f"{doc_id}_{i}"
                            meta_store.add_chunk(chunk_id, doc_id, result.file_path, i, chunk.text, chunk.tokens)
                            records.append({
                                "id": chunk_id, "vector": vector, "text": chunk.text,
                                "doc_path": result.file_path, "doc_name": result.file_name,
                                "doc_type": result.doc_type.value, "chunk_index": i,
                                "metadata": result.metadata, "context": cd.get("context", ""),
                            })
                        vec_store.add_batch(records)
                        vec_store.bm25.add_documents(records)
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


# ── Search & RAG ──


@router.post("/bases/{kb_id}/search")
async def search(kb_id: str, data: dict[str, Any]) -> list[dict[str, Any]]:
    """Search for relevant document chunks.

    Supports:
    - Vector search (default)
    - Hybrid search (BM25+Vector RRF) via hybrid=true (#15)
    - Reranking via rerank=true (#16)
    - Folder-level filter via folder_prefix (#12)
    """
    kb = _get_base(kb_id)
    query = data.get("query", "")
    if not query:
        raise HTTPException(400, "query is required")

    top_k = data.get("top_k", kb.config.max_results)
    threshold = data.get("threshold", kb.config.similarity_threshold)
    folder_prefix = data.get("folder_prefix")  # #12
    use_hybrid = data.get("hybrid", False)  # #15
    use_rerank = data.get("rerank", False)  # #16
    hybrid_alpha = data.get("hybrid_alpha", 0.7)  # #15
    hybrid_method = data.get("hybrid_method", "rrf")  # #15

    embed = _get_embed_client()
    vec_store = _get_vector_store(kb_id)

    # Query rewrite (optional)
    rewrite_mode = data.get("rewrite_mode")
    if rewrite_mode:
        rewriter = QueryRewriter(enabled=True)
        rewritten = await rewriter.rewrite(query, mode=rewrite_mode)
        if isinstance(rewritten, list):
            all_results = []
            for q in rewritten:
                qv = await embed.embed(q)
                if qv and not all(v == 0.0 for v in qv):
                    all_results.extend(vec_store.search(qv, top_k=top_k, threshold=threshold))
            seen = {}
            for r in all_results:
                rid = r.get("id", "")
                if rid not in seen or r.get("score", 0) > seen[rid].get("score", 0):
                    seen[rid] = r
            results = sorted(seen.values(), key=lambda x: x.get("score", 0), reverse=True)[:top_k]
            results = _apply_search_filters(results, folder_prefix)
            if use_rerank:
                results = await _do_rerank(query, results, top_k)
            return results
        query = rewritten

    query_vector = await embed.embed(query)
    if not query_vector or all(v == 0.0 for v in query_vector):
        raise HTTPException(500, "Embedding failed")

    # #15: Hybrid search (BM25 + Vector with RRF fusion)
    if use_hybrid:
        hs = HybridSearch(vec_store, alpha=hybrid_alpha, method=hybrid_method)
        filters = {}
        if folder_prefix:
            filters["folder_prefix"] = folder_prefix
        results = await hs.search(
            query_vector, query,
            top_k=top_k, threshold=threshold,
            filters=filters if filters else None,
        )
    else:
        results = vec_store.search(query_vector, top_k=top_k, threshold=threshold)
        results = _apply_search_filters(results, folder_prefix)

    # #16: Reranking
    if use_rerank:
        results = await _do_rerank(query, results, top_k)

    return results


def _apply_search_filters(results: list[dict], folder_prefix: str | None) -> list[dict]:
    """Apply folder prefix filter to search results (#12)."""
    if not folder_prefix:
        return results
    return [r for r in results if r.get("doc_path", "").startswith(folder_prefix)]


async def _do_rerank(query: str, results: list[dict], top_k: int) -> list[dict]:
    """Apply LLM reranking to search results (#16)."""
    if not results:
        return results
    try:
        mlx_base = _get_embed_client().base_url.replace("/v1", "")
        reranker = Reranker(mlx_url=mlx_base)
        return await reranker.rerank(query, results, top_k=top_k)
    except Exception as e:
        logger.error("rerank failed: %s", e)
        return results[:top_k]


@router.post("/bases/{kb_id}/ask")
async def ask(kb_id: str, data: dict[str, Any]) -> dict[str, Any]:
    """RAG: Retrieve relevant chunks and generate an answer.

    Supports:
    - Parameterized model/max_tokens/temperature (#17)
    - Hybrid search via hybrid=true (#15)
    - Reranking via rerank=true (#16)
    - Folder-level filter via folder_prefix (#12)
    """
    kb = _get_base(kb_id)
    question = data.get("question", "")
    if not question:
        raise HTTPException(400, "question is required")

    top_k = data.get("top_k", kb.config.max_results)
    # #17: Parameterized model config
    model = data.get("model", "qwen3.5-9b")
    max_tokens = data.get("max_tokens", 4096)
    temperature = data.get("temperature", 0.3)
    # #15/#16/#12: Search enhancements
    use_hybrid = data.get("hybrid", False)
    use_rerank = data.get("rerank", False)
    folder_prefix = data.get("folder_prefix")

    # 1. Retrieve
    embed = _get_embed_client()
    vec_store = _get_vector_store(kb_id)

    # Query rewrite (condense with conversation history)
    rewrite_mode = data.get("rewrite_mode")
    history = data.get("history")
    if rewrite_mode or history:
        rewriter = QueryRewriter(enabled=True)
        mode = rewrite_mode or ("condense" if history else "hyde")
        rewritten = await rewriter.rewrite(question, history=history, mode=mode)
        if isinstance(rewritten, str) and rewritten:
            question = rewritten

    query_vector = await embed.embed(question)
    if not query_vector or all(v == 0.0 for v in query_vector):
        raise HTTPException(500, "Embedding failed")

    # Search with optional hybrid + folder filter
    if use_hybrid:
        hs = HybridSearch(vec_store)
        filters = {}
        if folder_prefix:
            filters["folder_prefix"] = folder_prefix
        chunks = await hs.search(
            query_vector, question,
            top_k=top_k, threshold=kb.config.similarity_threshold,
            filters=filters if filters else None,
        )
    else:
        chunks = vec_store.search(query_vector, top_k=top_k, threshold=kb.config.similarity_threshold)
        if folder_prefix:
            chunks = [c for c in chunks if c.get("doc_path", "").startswith(folder_prefix)]

    if not chunks:
        return {"answer": "No relevant documents found.", "sources": []}

    # #16: Reranking
    if use_rerank:
        chunks = await _do_rerank(question, chunks, top_k)

    # 2. Build context
    context = "\n\n".join(
        f"[{c['doc_name']}] {c['text'][:2000]}"
        for c in chunks
    )

    # 3. Generate answer via fusion-mlx chat (#17: parameterized)
    answer = await _generate_answer(question, context, chunks,
                                    model=model, max_tokens=max_tokens,
                                    temperature=temperature)

    return answer


async def _generate_answer(question: str, context: str,
                           chunks: list[dict], *,
                           model: str = "qwen3.5-9b",
                           max_tokens: int = 4096,
                           temperature: float = 0.3) -> dict[str, Any]:
    """Generate an answer using fusion-mlx's chat API."""
    mlx_base = _get_embed_client().base_url.replace("/v1", "")
    mlx_url = f"{mlx_base}/v1/chat/completions"

    system_prompt = (
        "You are a knowledge base assistant. Answer the user's question based "
        "on the provided context. If the context doesn't contain the answer, "
        "say so. Always cite the source document names."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]

    import httpx
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(mlx_url, json={
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            })
            resp.raise_for_status()
            data = resp.json()
            answer_text = data["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error("RAG answer generation failed: %s", e)
        answer_text = f"Failed to generate answer: {e}"

    # Build sources
    seen = set()
    sources = []
    for c in chunks:
        doc_name = c.get("doc_name", "unknown")
        if doc_name not in seen:
            seen.add(doc_name)
            sources.append({
                "doc_name": doc_name,
                "doc_path": c.get("doc_path", ""),
                "score": c.get("score", 0),
                "snippet": c.get("text", "")[:200],
            })

    return {"answer": answer_text, "sources": sources}


# ── System ──


@router.get("/status")
async def status() -> dict[str, Any]:
    """Get service status."""
    kb_count = _get_kb_manager().count if _kb_manager else 0
    embed_ok = await _get_embed_client().health() if _embed_client else False
    return {
        "status": "ok",
        "knowledge_bases": kb_count,
        "embedding_available": embed_ok,
    }


@router.get("/bases/{kb_id}/stats")
async def kb_stats(kb_id: str) -> dict[str, Any]:
    """Get knowledge base statistics."""
    kb = _get_base(kb_id)
    meta_store = _get_meta_store(kb_id)
    vec_store = _get_vector_store(kb_id)
    return {
        "id": kb.id,
        "name": kb.config.name,
        "documents": meta_store.doc_count(),
        "chunks": meta_store.chunk_count(),
        "vectors": vec_store.count(),
        "file_count": kb.file_count,
        "chunk_count": kb.chunk_count,
    }


# ── Helpers ──


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
