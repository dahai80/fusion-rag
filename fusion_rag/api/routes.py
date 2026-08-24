from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from fusion_core.http_client import get_async_client, with_retry

from .._validators import validate_identifier
from ..embed.client import EmbeddingClient
from ..engine.knowledge_base import KnowledgeBaseManager
from ..engine.llm_errors import LLMUnavailable
from ..engine.reranker import Reranker
from ..store.metadata_store import MetadataStore
from ..store.vector_store import VectorStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/kb", tags=["knowledge-base"])

_kb_manager: KnowledgeBaseManager | None = None
_embed_client: EmbeddingClient | None = None

_DEFAULT_SYSTEM_PROMPT = (
    "You are a knowledge base assistant. Answer the user's question based "
    "on the provided context. If the context doesn't contain the answer, "
    "say so. Always cite the source document names."
)


def set_kb_context(kb_manager: KnowledgeBaseManager, embed_client: EmbeddingClient) -> None:
    global _kb_manager, _embed_client
    _kb_manager = kb_manager
    _embed_client = embed_client

    from .routes_admin import set_admin_kb_manager as _set_admin
    from .routes_docs import set_doc_context as _set_doc
    from .routes_kb import set_kb_context as _set_kb
    from .routes_project import set_project_context as _set_proj

    _set_kb(kb_manager, embed_client)
    _set_doc(kb_manager, embed_client)
    _set_admin(kb_manager)
    _set_proj(kb_manager)


def _get_kb_manager() -> KnowledgeBaseManager:
    if _kb_manager is None:
        raise HTTPException(503, "Knowledge base manager not initialized")
    return _kb_manager


def _get_embed_client() -> EmbeddingClient:
    if _embed_client is None:
        raise HTTPException(503, "Embedding client not initialized")
    return _embed_client


def _get_base(kb_id: str):
    # F12: confine kb_id to the identifier charset before it reaches any path
    # construction (vector_path / metadata_path). Without this, a kb_id of
    # "../etc" traverses out of the stores root.
    try:
        validate_identifier(kb_id, field="kb_id")
    except ValueError:
        raise HTTPException(400, f"Invalid kb_id: {kb_id}")
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


def _get_kb_storage_path(kb_id: str) -> str:
    kb = _get_base(kb_id)
    return kb.vector_path.rsplit("/vectors", 1)[0] if "/vectors" in kb.vector_path else kb.vector_path


def _apply_search_filters(
    results: list[dict], folder_prefix: str | None, metadata_filter: dict | None = None
) -> list[dict]:
    filtered = results
    if folder_prefix:
        filtered = [r for r in filtered if r.get("doc_path", "").startswith(folder_prefix)]
    if metadata_filter:

        def _match_meta(r: dict) -> bool:
            meta = r.get("metadata", {})
            if isinstance(meta, str):
                import json as _json

                try:
                    meta = _json.loads(meta)
                except Exception:
                    meta = {}
            return all(meta.get(k) == v for k, v in metadata_filter.items())

        filtered = [r for r in filtered if _match_meta(r)]
    return filtered


async def _do_rerank(query: str, results: list[dict], top_k: int) -> list[dict]:
    if not results:
        return results
    try:
        mlx_base = _get_embed_client().base_url.replace("/v1", "")
        reranker = Reranker(mlx_url=mlx_base)
        return await reranker.rerank(query, results, top_k=top_k)
    except LLMUnavailable as e:
        # L1: rerank is an enhancement. On LLM failure fall back to original
        # retrieval order — logged, not silent. Narrow catch: programmer bugs
        # (AttributeError etc.) must surface as 500, not be masked as rerank
        # failure.
        logger.warning("rerank LLM unavailable, returning original order: %s", e)
        return results[:top_k]


async def _generate_answer(
    question: str,
    context: str,
    chunks: list[dict],
    *,
    model: str = "qwen3.5-9b",
    max_tokens: int = 4096,
    temperature: float = 0.3,
    system_prompt: str = "",
) -> dict[str, Any]:
    import os as _os

    embed = _get_embed_client()
    mlx_base = embed.base_url.replace("/v1", "")
    mlx_url = f"{mlx_base}/v1/chat/completions"
    mlx_base_url = f"{mlx_base}/v1"
    headers = {}
    if embed.api_key:
        headers["Authorization"] = f"Bearer {embed.api_key}"
    if not system_prompt:
        system_prompt = _os.environ.get("FUSION_RAG_SYSTEM_PROMPT", _DEFAULT_SYSTEM_PROMPT)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]

    try:
        client = get_async_client(mlx_base_url, timeout=60.0)
        resp = await with_retry(
            lambda: client.post(
                mlx_url,
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                headers=headers,
            ),
            retries=2,
        )
        resp.raise_for_status()
        data = resp.json()
        answer_text = data["choices"][0]["message"]["content"]
        if not answer_text or not answer_text.strip():
            logger.warning("RAG answer empty content, question=%s", question[:50])
            raise ValueError("empty_content")
    except Exception as e:
        # L9: do not return "Failed to generate answer: {e}" with HTTP 200 —
        # that makes a down upstream (fusion-mlx) look like a successful (if
        # unhelpful) answer. Propagate so the ask endpoint maps to 502.
        logger.error("RAG answer generation failed: %s", e)
        raise LLMUnavailable("answer generation failed") from e

    seen = set()
    sources = []
    for c in chunks:
        doc_name = c.get("doc_name", "unknown")
        if doc_name not in seen:
            seen.add(doc_name)
            sources.append(
                {
                    "doc_name": doc_name,
                    "doc_path": c.get("doc_path", ""),
                    "score": c.get("score", 0),
                    "snippet": c.get("text", "")[:200],
                }
            )
    return {"answer": answer_text, "sources": sources}


# ── System endpoints (stay here) ──


@router.get("/status")
async def status() -> dict[str, Any]:
    kb_count = _get_kb_manager().count if _kb_manager else 0
    embed_ok = await _get_embed_client().health() if _embed_client else False
    return {
        "status": "ok",
        "knowledge_bases": kb_count,
        "embedding_available": embed_ok,
    }


@router.get("/bases/{kb_id}/stats")
async def kb_stats(kb_id: str) -> dict[str, Any]:
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


# ── Mount sub-routers ──

from .routes_admin import router as _admin_router
from .routes_docs import router as _docs_router
from .routes_kb import router as _kb_router
from .routes_project import router as _project_router
from .routes_search import router as _search_router

router.include_router(_kb_router)
router.include_router(_docs_router)
router.include_router(_search_router)
router.include_router(_admin_router)
router.include_router(_project_router)
