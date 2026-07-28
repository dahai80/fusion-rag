"""Fusion-RAG HTTP API — FastAPI routes for knowledge base operations.

All model calls go through fusion-mlx HTTP API (/v1/embeddings, /v1/chat/completions).
No direct MLX imports.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from ..engine.knowledge_base import KnowledgeBaseManager, KnowledgeBaseConfig
from ..engine.document import DocumentParser, ParseResult
from ..engine.chunker import Chunker, Chunk
from ..embed.client import EmbeddingClient
from ..store.vector_store import VectorStore
from ..store.metadata_store import MetadataStore

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


@router.post("/bases")
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


@router.delete("/bases/{kb_id}")
async def delete_knowledge_base(kb_id: str) -> dict[str, str]:
    """Delete a knowledge base."""
    if _get_kb_manager().delete(kb_id):
        return {"id": kb_id, "status": "deleted"}
    raise HTTPException(404, f"Knowledge base '{kb_id}' not found")


# ── Document Operations ──


@router.post("/bases/{kb_id}/documents")
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

    # Embed
    embed = _get_embed_client()
    vectors = await embed.embed_batch([c.text for c in chunks])

    # Store vectors
    vec_store = _get_vector_store(kb_id)
    meta_store = _get_meta_store(kb_id)

    doc_id = uuid.uuid4().hex[:16]
    meta_store.add_document(doc_id, result.file_path, result.file_name,
                            result.doc_type.value, result.metadata.get("size", 0))

    records = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
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
        })

    vec_store.add_batch(records)
    meta_store.update_chunk_count(doc_id, len(chunks), result.chars)

    # Update KB stats
    kb.file_count += 1
    kb.chunk_count += len(chunks)
    _get_kb_manager().update(kb_id, file_count=kb.file_count, chunk_count=kb.chunk_count)

    return {"doc_id": doc_id, "chunks": len(chunks), "chars": result.chars}


@router.post("/bases/{kb_id}/scan")
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

        vectors = await embed.embed_batch([c.text for c in chunks])
        doc_id = uuid.uuid4().hex[:16]
        meta_store.add_document(doc_id, result.file_path, result.file_name,
                                result.doc_type.value, result.metadata.get("size", 0))

        records = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            chunk_id = f"{doc_id}_{i}"
            meta_store.add_chunk(chunk_id, doc_id, result.file_path, i, chunk.text, chunk.tokens)
            records.append({
                "id": chunk_id, "vector": vector, "text": chunk.text,
                "doc_path": result.file_path, "doc_name": result.file_name,
                "doc_type": result.doc_type.value, "chunk_index": i,
                "metadata": result.metadata,
            })

        vec_store.add_batch(records)
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

    query_vector = await embed.embed(query)
    if not query_vector:
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

    query_vector = await embed.embed(question)
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