"""Fusion-RAG HTTP API — FastAPI routes for knowledge base operations.

All model calls go through fusion-mlx HTTP API (/v1/embeddings, /v1/chat/completions).
No direct MLX imports.
"""

from __future__ import annotations

import logging
import uuid
import os
import asyncio
import hashlib
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from ..embed.client import EmbeddingClient
from ..engine.chunker import Chunker
from ..engine.contextualizer import Contextualizer
from ..engine.document import DocumentParser
from ..engine.knowledge_base import KnowledgeBaseManager
from ..engine.query_rewriter import QueryRewriter
from ..store.metadata_store import MetadataStore
from ..store.vector_store import VectorStore
from .auth import verify_api_key

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
    asyncio.create_task(_watch_loop(watch_id))
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
    """Search for relevant document chunks using vector similarity."""
    kb = _get_base(kb_id)
    query = data.get("query", "")
    if not query:
        raise HTTPException(400, "query is required")

    top_k = data.get("top_k", kb.config.max_results)
    threshold = data.get("threshold", kb.config.similarity_threshold)

    embed = _get_embed_client()
    vec_store = _get_vector_store(kb_id)

    # Query rewrite (optional)
    rewrite_mode = data.get("rewrite_mode")
    if rewrite_mode:
        rewriter = QueryRewriter(enabled=True)
        rewritten = await rewriter.rewrite(query, mode=rewrite_mode)
        if isinstance(rewritten, list):
            # For expand mode, embed original + variants and merge results
            all_results = []
            for q in rewritten:
                qv = await embed.embed(q)
                if qv and not all(v == 0.0 for v in qv):
                    all_results.extend(vec_store.search(qv, top_k=top_k, threshold=threshold))
            # Deduplicate by id, keep highest score
            seen = {}
            for r in all_results:
                rid = r.get("id", "")
                if rid not in seen or r.get("score", 0) > seen[rid].get("score", 0):
                    seen[rid] = r
            results = sorted(seen.values(), key=lambda x: x.get("score", 0), reverse=True)[:top_k]
            return results
        query = rewritten

    # callers: test_search_with_mocked_embed expects 500 when embed fails
    # API: search() -> list[dict]; embed returns zero vector on failure
    # schema: query_vector is list[float], zero-vector means embed failed
    # user instruction: "bug/问题/需求 修改完成，遇到报错的用例，不管和自己的代码是否相关，都要定位修复"
    query_vector = await embed.embed(query)
    if not query_vector or all(v == 0.0 for v in query_vector):
        raise HTTPException(500, "Embedding failed")

    results = vec_store.search(query_vector, top_k=top_k, threshold=threshold)
    return results


@router.post("/bases/{kb_id}/ask")
async def ask(kb_id: str, data: dict[str, Any]) -> dict[str, Any]:
    """RAG: Retrieve relevant chunks and generate an answer."""
    kb = _get_base(kb_id)
    question = data.get("question", "")
    if not question:
        raise HTTPException(400, "question is required")

    top_k = data.get("top_k", kb.config.max_results)

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
    chunks = vec_store.search(query_vector, top_k=top_k, threshold=kb.config.similarity_threshold)

    if not chunks:
        return {"answer": "No relevant documents found.", "sources": []}

    # 2. Build context
    context = "\n\n".join(
        f"[{c['doc_name']}] {c['text'][:2000]}"
        for c in chunks
    )

    # 3. Generate answer via fusion-mlx chat
    answer = await _generate_answer(question, context, chunks)

    return answer


async def _generate_answer(question: str, context: str,
                           chunks: list[dict]) -> dict[str, Any]:
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
                "model": "qwen3.5-9b",
                "messages": messages,
                "max_tokens": 4096,
                "temperature": 0.3,
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
