"""Issue #70 — cross-encoder rerank via fusion-mlx POST /v1/rerank.

Real cross-encoder scoring (bge-reranker-v2-m3) re-ranks the top-N vector
candidates, returning the final top_k. Replaces the LLM-prompt-scoring
`Reranker` (engine/reranker.py) as the precision stage when a rerank model is
configured. fusion-mlx exposes a Cohere/Jina-compatible /v1/rerank endpoint;
this module is a plain async HTTP client to it — no MLX import (respects the
"no direct MLX imports" + "only modify own project" rules).

callers: routes._do_rerank (backend selection), routes_search /search + /ask
API: CrossEncoderReranker.rerank(query, documents, top_k) -> list[dict]
schema: documents list[dict] with id/text/score keys; returns same dicts with
        score stamped to the cross-encoder relevance_score, reordered desc.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from fusion_core.http_client import get_async_client, with_retry

from .llm_errors import LLMUnavailable

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "bge-reranker-v2-m3"
_RERANK_TIMEOUT = 10.0
_MAX_DOC_CHARS = 2000


def _mlx_base_url() -> str:
    raw = os.environ.get("FUSION_MLX_URL", "http://127.0.0.1:11432/v1").strip()
    return raw.rstrip("/")


def _mlx_api_key() -> str:
    return os.environ.get("FUSION_MLX_API_KEY", "").strip()


class CrossEncoderReranker:
    """Async client for fusion-mlx /v1/rerank (cross-encoder reranking).

    rerank(query, documents, top_k) -> documents reordered by cross-encoder
    relevance_score (desc), truncated to top_k. Raises LLMUnavailable on
    network/5xx/unreachable so the route layer falls back to original order
    (same contract as the LLM-prompt Reranker). 400/404 (model not found /
    not a reranker) also raise LLMUnavailable — operator misconfig degrades to
    fallback, not a crash.
    """

    def __init__(
        self,
        mlx_base_url: str = "",
        model: str = "",
        api_key: str = "",
        timeout: float = _RERANK_TIMEOUT,
    ):
        self._base = (mlx_base_url or _mlx_base_url())
        self._model = model or os.environ.get("FUSION_RAG_RERANK_MODEL", "").strip() or _DEFAULT_MODEL
        self._api_key = api_key or _mlx_api_key()
        self._timeout = timeout

    async def rerank(
        self, query: str, documents: list[dict[str, Any]], top_k: int = 5
    ) -> list[dict[str, Any]]:
        if not documents:
            return []
        texts = [str(d.get("text", ""))[:_MAX_DOC_CHARS] for d in documents]
        payload = {
            "model": self._model,
            "query": query,
            "documents": texts,
            "top_n": top_k,
            "return_documents": False,
        }
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        try:
            results = await self._call_rerank(payload, headers)
        except LLMUnavailable:
            raise
        except Exception as e:
            logger.warning("CrossEncoderReranker.rerank failed: %s", e)
            raise LLMUnavailable() from e
        ranked = []
        for r in results[:top_k]:
            idx = r.get("index")
            if idx is None or idx < 0 or idx >= len(documents):
                continue
            doc = dict(documents[idx])
            doc["score"] = float(r.get("relevance_score", 0.0))
            ranked.append(doc)
        return ranked

    async def _call_rerank(self, payload: dict, headers: dict) -> list[dict]:
        client = get_async_client(self._base, timeout=self._timeout)
        try:
            resp = await with_retry(
                lambda: client.post(f"{self._base}/rerank", json=payload, headers=headers),
                retries=2,
                total_deadline=15.0,
            )
        except Exception as e:
            logger.warning("CrossEncoderReranker: /v1/rerank unreachable (%s): %s", self._base, e)
            raise LLMUnavailable() from e
        if resp.status_code == 200:
            try:
                data = resp.json()
            except Exception as e:
                logger.warning("CrossEncoderReranker: bad JSON from /v1/rerank: %s", e)
                raise LLMUnavailable() from e
            return data.get("results", [])
        if resp.status_code in (400, 404):
            logger.warning(
                "CrossEncoderReranker: /v1/rerank %s (model=%s) — misconfig, fallback",
                resp.status_code,
                self._model,
            )
            raise LLMUnavailable()
        logger.warning("CrossEncoderReranker: unexpected /v1/rerank status %s", resp.status_code)
        raise LLMUnavailable()

    async def aclose(self) -> None:
        # Pool-managed client; nothing to close here.
        return None
