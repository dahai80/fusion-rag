from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import APIRouter, HTTPException
from fusion_core.http_client import get_async_client, with_retry

from .._validators import validate_identifier
from ..engine.llm_errors import LLMUnavailable
from ..engine.reranker import Reranker
from ..store.metadata_store import MetadataStore
from ..store.vector_store import VectorStore
from .app_state import get_embed_client, get_kb_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/kb", tags=["knowledge-base"])

_DEFAULT_SYSTEM_PROMPT = (
    "You are a knowledge base assistant. Answer the user's question based "
    "on the provided context. If the context doesn't contain the answer, "
    "say so. Always cite the source document names."
)


# 硬伤1: per-app state lives on app.state (populated by init_app_state in the
# lifespan) and is read per-request via a contextvar. These no-arg accessors
# delegate to app_state so existing callers (_get_base, _do_rerank,
# _generate_answer, and the `from .routes import _get_embed_client` in
# routes_search) keep working without a Request parameter.
_get_kb_manager = get_kb_manager
_get_embed_client = get_embed_client


def _get_base(kb_id: str):
    # F12: confine kb_id to the identifier charset before it reaches any path
    # construction (vector_path / metadata_path). Without this, a kb_id of
    # "../etc" traverses out of the stores root.
    try:
        validate_identifier(kb_id, field="kb_id")
    except ValueError:
        raise HTTPException(400, f"Invalid kb_id: {kb_id}")
    # Issue #61: scope by the request's authoritative tenant when isolation is
    # on — a tenant-A caller addressing tenant-B's kb_id gets 404, not the KB.
    from .tenant import tenant_scope

    tenant, require_match = tenant_scope()
    try:
        return _get_kb_manager().get(kb_id, tenant=tenant, require_tenant_match=require_match)
    except KeyError:
        raise HTTPException(404, f"Knowledge base '{kb_id}' not found")


def _get_vector_store(kb_id: str) -> VectorStore:
    kb = _get_base(kb_id)
    backend_type = os.environ.get("FUSION_RAG_STORE_BACKEND", "local")
    # 硬伤A: reuse one pooled backend handle per KB (see app_state
    # get_or_create_vec_store). The prior per-request VectorStore leaked
    # handles and reopened the fusion_store env each call (EnvAlreadyOpened).
    from .app_state import get_or_create_vec_store

    return get_or_create_vec_store(kb.vector_path, backend_type)


def _get_meta_store(kb_id: str) -> MetadataStore:
    kb = _get_base(kb_id)
    # A-P1-1: reuse the pooled MetadataStore (one conn per metadata_path) via
    # app_state. The prior per-request MetadataStore(kb.metadata_path) opened a
    # fresh sqlite conn each call and never closed it — FD leak across a long
    # run. The pool closes all conns from the lifespan shutdown.
    from .app_state import get_meta_store

    return get_meta_store(kb.metadata_path)


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


def _normalize_evidence(results: list[dict]) -> list[dict]:
    # Issue #77: every retrieved chunk carries chunk_id / doc_id / page at the
    # top level so consumers (envipermit) can bind extracted values to stable
    # evidence anchors. doc_id falls back to the chunk-id prefix ("{doc_id}_{i}"
    # format) for rows ingested before the field was stamped into chunk
    # metadata; page is 0 when the source has no page info.
    # Issue #85: OCR-sourced chunks also surface ocr_confidence + slice
    # evidence (ocr_task_id/ocr_slice_id/ocr_slice_url) + low_confidence flag
    # against RuntimeConfig.low_ocr_confidence. Issue #86: status is stamped at
    # ingest; the authoritative live status is resolved from the documents
    # table by _apply_status_filter (PATCH can flip it after ingest).
    from ..engine.runtime_config import get_runtime_config

    cfg = get_runtime_config()
    for r in results:
        if not isinstance(r, dict):
            continue
        meta = r.get("metadata") or {}
        if isinstance(meta, str):
            import json as _json

            try:
                meta = _json.loads(meta)
            except Exception:
                meta = {}
        chunk_id = r.get("id", "")
        doc_id = meta.get("doc_id") or (chunk_id.rsplit("_", 1)[0] if "_" in chunk_id else chunk_id)
        r["chunk_id"] = chunk_id
        r["doc_id"] = doc_id
        r["page"] = meta.get("page", 0)
        r["status"] = meta.get("status", "active")
        if "ocr_confidence" in meta:
            conf = meta.get("ocr_confidence")
            r["ocr_confidence"] = conf
            r["low_confidence"] = conf is not None and conf < cfg.low_ocr_confidence
        if meta.get("ocr_slice_id"):
            r["slice_ref"] = {
                "task_id": meta.get("ocr_task_id", ""),
                "slice_id": meta.get("ocr_slice_id", ""),
                "url": meta.get("ocr_slice_url", ""),
            }
    return results


async def _apply_status_filter(
    kb_id: str, results: list[dict], status_filter: str
) -> tuple[list[dict], dict[str, str]]:
    """Issue #86: filter retrieved chunks by document status (active|deprecated).

    status_filter: "active" (default — drops deprecated docs), "deprecated"
    (only deprecated), "all" (no filter). Status is resolved from the
    documents table (the PATCH endpoint flips it there), NOT from the
    ingest-time chunk metadata — a doc deprecated after ingest must vanish
    from default search without re-indexing. Returns (kept, status_map) so
    /ask can mark deprecated context without a second lookup."""
    from fastapi.concurrency import run_in_threadpool

    doc_ids = [r.get("doc_id", "") for r in results if isinstance(r, dict)]
    status_map = await run_in_threadpool(_get_meta_store(kb_id).get_document_status_map, doc_ids)
    if status_filter == "deprecated":
        kept = [r for r in results if status_map.get(r.get("doc_id", "")) == "deprecated"]
    elif status_filter == "all":
        kept = list(results)
    else:
        kept = [r for r in results if status_map.get(r.get("doc_id", ""), "active") == "active"]
    for r in kept:
        if isinstance(r, dict):
            r["status"] = status_map.get(r.get("doc_id", ""), "active")
    return kept, status_map


async def _do_rerank(
    query: str,
    results: list[dict],
    top_k: int,
    *,
    backend: str = "",
    model: str = "",
) -> list[dict]:
    if not results:
        return results
    from ..engine.runtime_config import get_runtime_config

    cfg = get_runtime_config()
    backend = (backend or cfg.rerank_backend or "llm").strip().lower()
    model = model or cfg.rerank_model
    embed = _get_embed_client()
    mlx_base = embed.base_url.replace("/v1", "")
    # Issue #70: cross_encoder backend uses fusion-mlx POST /v1/rerank (real
    # cross-encoder, e.g. bge-reranker-v2-m3). Falls back to the legacy
    # LLM-prompt Reranker on LLMUnavailable, then to original order — so a
    # missing/unreachable rerank model degrades, never crashes.
    # Issue #72: pass embed.api_key so rerank honors the same auth as the
    # embedding/LLM backend — a non-default gateway URL+key no longer 401s.
    if backend == "cross_encoder":
        from ..engine.cross_encoder_reranker import CrossEncoderReranker

        try:
            reranker = CrossEncoderReranker(mlx_base_url=mlx_base, model=model, api_key=embed.api_key)
            return await reranker.rerank(query, results, top_k=top_k)
        except LLMUnavailable as e:
            logger.warning(
                "cross-encoder rerank unavailable (model=%s), falling back to llm rerank: %s",
                model or "(default)",
                e,
            )
            # Fall through to the LLM-prompt reranker as a second-tier fallback.
    try:
        reranker = Reranker(mlx_url=mlx_base, api_key=embed.api_key)
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
        import time as _time

        from ..engine.metrics import record_llm_latency

        _llm_start = _time.perf_counter()
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
            total_deadline=30.0,
        )
        record_llm_latency("generate", (_time.perf_counter() - _llm_start) * 1000)
        resp.raise_for_status()
        data = resp.json()
        answer_text = data["choices"][0]["message"]["content"]
        if not answer_text or not answer_text.strip():
            # O-P1-3: question is PII — drop snippet, keep the warning signal.
            logger.warning("RAG answer empty content (question len=%d)", len(question))
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
                    # Issue #77: evidence anchors on /ask sources too.
                    "doc_id": c.get("doc_id", ""),
                    "chunk_id": c.get("chunk_id", ""),
                    "page": c.get("page", 0),
                }
            )
    # Issue #85: per-chunk citations (NOT deduped — every cited chunk is an
    # anchor) + has_low_confidence so a consumer can flag OCR-shaky answers.
    from ..engine.runtime_config import get_runtime_config

    cfg = get_runtime_config()
    citations = []
    has_low_confidence = False
    for c in chunks:
        conf = c.get("ocr_confidence")
        if conf is not None and conf < cfg.low_ocr_confidence:
            has_low_confidence = True
        citation = {
            "doc_id": c.get("doc_id", ""),
            "doc_name": c.get("doc_name", "unknown"),
            "chunk_id": c.get("chunk_id", ""),
            "page": c.get("page", 0),
            "score": c.get("score", 0),
            "status": c.get("status", "active"),
        }
        if conf is not None:
            citation["ocr_confidence"] = conf
            citation["low_confidence"] = conf < cfg.low_ocr_confidence
        if c.get("slice_ref"):
            citation["slice_ref"] = c["slice_ref"]
        citations.append(citation)
    return {"answer": answer_text, "sources": sources, "citations": citations, "has_low_confidence": has_low_confidence}


# ── System endpoints (stay here) ──


@router.get("/status")
async def status() -> dict[str, Any]:
    # 硬伤1: read via contextvar accessors. /status is a readiness probe, so
    # if the app state isn't bound yet (startup race / direct call) degrade
    # gracefully to "not ready" rather than 503-ing the probe.
    try:
        # Issue #61: report the tenant-scoped KB count when isolation is on, so
        # a tenant's /status reflects only its own KBs, not the whole fleet.
        from .tenant import tenant_scope

        _tenant, _require = tenant_scope()
        kb_count = len(_get_kb_manager().list(tenant=_tenant, require_tenant_match=_require))
    except HTTPException:
        kb_count = 0
    try:
        embed_ok = await _get_embed_client().health()
    except HTTPException:
        embed_ok = False
    except Exception as e:
        logger.warning("/status: embed health check failed: %s", e)
        embed_ok = False
    return {
        "status": "ok",
        "knowledge_bases": kb_count,
        "embedding_available": embed_ok,
    }


# ── Mount sub-routers ──

from .routes_admin import router as _admin_router
from .routes_docs import router as _docs_router
from .routes_graph import router as _graph_router
from .routes_kb import router as _kb_router
from .routes_project import router as _project_router
from .routes_search import router as _search_router
from .routes_store import router as _store_router

router.include_router(_kb_router)
router.include_router(_docs_router)
router.include_router(_search_router)
router.include_router(_admin_router)
router.include_router(_project_router)
router.include_router(_store_router)
router.include_router(_graph_router)
