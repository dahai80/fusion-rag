"""Reranker and HybridSearch — batch reranking and multi-strategy fusion.

callers: routes.py search/ask endpoints, KnowledgeBase pipeline
API: Reranker.rerank(query, documents, top_k), HybridSearch.search(query_vector, query_text, ...)
schema: documents list[dict] with id/text/score keys, HybridSearch supports alpha and rrf fusion methods
user instruction: "按照你的方案和计划落地所有phase阶段的需求"
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

import httpx
from fastapi.concurrency import run_in_threadpool
from fusion_core.http_client import get_async_client, with_retry

from .llm_errors import LLMUnavailable

logger = logging.getLogger(__name__)


class Reranker:
    """Reranks search results by relevance to query.

    Uses batch LLM scoring — single API call for all documents.
    """

    def __init__(self, mlx_url: str = "http://127.0.0.1:11432/v1", model: str = "qwen3.5-9b", batch_size: int = 20):
        self.mlx_url = mlx_url.rstrip("/")
        self.model = model
        self.batch_size = batch_size

    async def _get_client(self) -> httpx.AsyncClient:
        # A8: pool is the single source of truth (dedup + is_closed check).
        # Drop the self._client check-then-assign cache that could hold a
        # pool-evicted/closed reference.
        return get_async_client(self.mlx_url, timeout=30.0)

    async def aclose(self) -> None:
        # Pool-managed: nothing to close here.
        return None

    async def rerank(self, query: str, documents: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
        if not documents:
            return []
        # P-P2-5: prior loop scored batches SEQUENTIALLY — N batches = N serial
        # LLM round-trips, rerank latency = sum not max. Fan the batches out
        # concurrently behind a semaphore so the LLM backend (sync-friendly,
        # pool-shared) isn't flooded. Batch boundaries preserved so each scored
        # batch maps back to its docs in order.
        batches = [documents[i : i + self.batch_size] for i in range(0, len(documents), self.batch_size)]
        sem = asyncio.Semaphore(4)

        async def _score_batch(batch: list[dict[str, Any]]) -> list[float]:
            async with sem:
                return await self._batch_score(query, batch)

        score_lists = await asyncio.gather(*[_score_batch(b) for b in batches], return_exceptions=True)
        all_scored = []
        for batch, result in zip(batches, score_lists):
            if isinstance(result, Exception):
                # L1: a single batch failure propagates LLMUnavailable from
                # _batch_score; gather captured it. Re-raise so the route's
                # rerank-fallback (original order) engages, not a partial rerank
                # that silently drops the failed batch's docs.
                raise result
            for doc, score in zip(batch, result):
                doc["score"] = score
                all_scored.append(doc)
        all_scored.sort(key=lambda x: x["score"], reverse=True)
        return all_scored[:top_k]

    async def _batch_score(self, query: str, docs: list[dict[str, Any]]) -> list[float]:
        doc_list = "\n".join(f"[{i}] {doc.get('text', '')[:500]}" for i, doc in enumerate(docs))
        prompt = (
            f"Rate the relevance of each document to the query "
            f"on a scale of 0.0 to 10.0.\n\n"
            f"Query: {query}\n\n"
            f"Documents:\n{doc_list}\n\n"
            f"Output ONLY a JSON array of {len(docs)} scores, e.g. [8.5, 3.2, ...]:"
        )
        try:
            client = await self._get_client()
            import time as _time

            from .metrics import record_llm_latency

            _llm_start = _time.perf_counter()
            resp = await with_retry(
                lambda: client.post(
                    f"{self.mlx_url}/chat/completions",
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 200,
                        "temperature": 0.0,
                    },
                ),
                retries=2,
                total_deadline=15.0,
            )
            record_llm_latency("rerank", (_time.perf_counter() - _llm_start) * 1000)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            if not content:
                logger.warning("Reranker empty content (query len=%d)", len(query))
                raise ValueError("empty_content")
            return self._parse_scores(content, len(docs))
        except Exception as e:
            # L1: do NOT return a 5.0 magic array — that makes total LLM
            # failure look like a successful rerank with confident scores.
            # Propagate so the route layer maps to a logged fallback (original
            # order) rather than a silent fabricated rerank.
            logger.warning("Batch rerank failed (propagating, no fabricated scores): %s", e)
            raise LLMUnavailable() from e

    def _parse_scores(self, content: str, expected: int) -> list[float]:
        try:
            scores = json.loads(content)
            if isinstance(scores, list):
                return [float(s) for s in scores[:expected]]
        except Exception as e:
            logger.warning("Failed to parse rerank scores as JSON: %s", e)
        nums = re.findall(r"\d+\.?\d*", content)
        if len(nums) >= expected:
            return [float(n) for n in nums[:expected]]
        logger.warning("Could not parse rerank scores, using defaults")
        return [5.0] * expected

    async def _score_relevance(self, client: httpx.AsyncClient, query: str, text: str) -> float:
        """Single-doc fallback scoring (0-10)."""
        prompt = (
            f"Rate the relevance of the following document to the query "
            f"on a scale of 0 to 10 (0=completely irrelevant, 10=perfect match).\n\n"
            f"Query: {query}\n\n"
            f"Document: {text[:1500]}\n\n"
            f"Relevance score (0-10):"
        )
        try:
            resp = await client.post(
                f"{self.mlx_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 10,
                    "temperature": 0.0,
                },
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            if not content:
                # O-P1-3: query is PII — drop snippet.
                logger.warning("Reranker _score_relevance empty content (query len=%d)", len(query))
                raise ValueError("empty_content")
            return float(content[:4])
        except Exception as e:
            # M2/L1: do NOT return a 5.0 magic midpoint — that makes total LLM
            # failure look like a confident mediocre score and (per M2) gets
            # locked in by a test asserting 5.0, so a future fix to fail loudly
            # would be blocked by the test. Propagate so the caller knows the
            # single-doc fallback is unavailable.
            logger.warning("Reranker _score_relevance failed (propagating, no 5.0 fallback): %s", e)
            raise LLMUnavailable("single-doc relevance scoring failed") from e


class HybridSearch:
    """Combines vector similarity and keyword search with weighted or RRF fusion."""

    def __init__(self, vector_store, alpha: float = 0.7, method: str = "rrf"):
        # P1-6: default RRF (rank-based) not alpha. vector score ∈ [0,1] (cosine
        # 1-dist) while BM25 Okapi is unbounded (can exceed 10) — alpha-weighting
        # them raw lets BM25 dominate and makes alpha meaningless. RRF is
        # scale-free so it is the correct default. Alpha is still available on
        # request, with BM25 min-max normalized first (see _alpha_fusion).
        self.vector_store = vector_store
        self.alpha = alpha
        self.method = method

    async def search(
        self,
        query_vector: list[float],
        query_text: str,
        top_k: int = 10,
        threshold: float = 0.0,
        filters: dict | None = None,
    ) -> list[dict[str, Any]]:
        # P-P1-1: vector_store.search / keyword_search are sync (LanceDB + BM25).
        # Calling them inline in an async handler blocks the event loop for the
        # full search duration. Push both off the loop thread; run them
        # concurrently (independent reads) so the two backends don't serialize.
        vector_results, keyword_results = await asyncio.gather(
            run_in_threadpool(self.vector_store.search, query_vector, top_k=top_k * 2, threshold=0.0),
            run_in_threadpool(self.vector_store.keyword_search, query_text, top_k=top_k * 2),
        )

        if filters:
            vector_results = self._apply_filters(vector_results, filters)
            keyword_results = self._apply_filters(keyword_results, filters)

        if self.method == "rrf":
            return self._rrf_fusion(vector_results, keyword_results, top_k, threshold)
        return self._alpha_fusion(vector_results, keyword_results, top_k, threshold)

    def _alpha_fusion(
        self, vector_results: list[dict], keyword_results: list[dict], top_k: int, threshold: float
    ) -> list[dict[str, Any]]:
        # P1-6: vector scores are cosine 1-dist ∈ [0,1]; BM25 Okapi scores are
        # unbounded (can exceed 10). Adding them raw with an alpha weight lets
        # BM25 dominate and makes alpha meaningless. Min-max normalize both to
        # [0,1] before weighting so the two scales are commensurable. A single-
        # result list normalizes to 1.0 (no spread to compute — keep its value).
        def _normalize(results: list[dict]) -> list[dict]:
            if not results:
                return results
            vals = [r.get("score", 0.0) for r in results]
            lo, hi = min(vals), max(vals)
            span = hi - lo
            if span <= 0:
                return [{**r, "score": 1.0} for r in results]
            return [{**r, "score": (r.get("score", 0.0) - lo) / span} for r in results]

        vector_results = _normalize(vector_results)
        keyword_results = _normalize(keyword_results)

        scores: dict[str, float] = {}
        for r in vector_results:
            rid = r.get("id", "")
            scores[rid] = scores.get(rid, 0) + self.alpha * r.get("score", 0)

        for r in keyword_results:
            rid = r.get("id", "")
            scores[rid] = scores.get(rid, 0) + (1 - self.alpha) * r.get("score", 0)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ranked = [{"id": rid, "score": s} for rid, s in ranked if s >= threshold]

        result_map = {r["id"]: r for r in vector_results}
        for r in keyword_results:
            if r["id"] not in result_map:
                result_map[r["id"]] = r
        for r in ranked:
            if r["id"] in result_map:
                r.update(result_map[r["id"]])
        return ranked[:top_k]

    @staticmethod
    def _rrf_fusion(
        vector_results: list[dict], keyword_results: list[dict], top_k: int, threshold: float, k: int = 60
    ) -> list[dict[str, Any]]:
        scores: dict[str, float] = {}
        for rank, r in enumerate(vector_results, 1):
            rid = r.get("id", "")
            scores[rid] = scores.get(rid, 0) + 1.0 / (k + rank)
        for rank, r in enumerate(keyword_results, 1):
            rid = r.get("id", "")
            scores[rid] = scores.get(rid, 0) + 1.0 / (k + rank)

        result_map = {r["id"]: r for r in vector_results}
        for r in keyword_results:
            if r["id"] not in result_map:
                result_map[r["id"]] = r

        logger.debug(
            "RRF fusion: threshold=%.4f ignored (RRF scores are rank-based, not cosine-scaled)",
            threshold,
        )
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for rid, s in ranked:
            entry = dict(result_map.get(rid, {"id": rid}))
            entry["score"] = s
            results.append(entry)
        return results[:top_k]

    @staticmethod
    def _apply_filters(results: list[dict], filters: dict) -> list[dict]:
        # P1-7: folder_prefix is NOT a metadata field — it is a top-level
        # doc_path prefix match. The prior loop only checked `if key in meta`,
        # so folder_prefix was never found in metadata → silently passed every
        # row → a folder-filtered hybrid search returned the whole KB. The
        # non-hybrid path (routes._apply_search_filters) already does
        # doc_path.startswith; this must match. Other keys stay metadata lookups
        # (unchanged behavior for a key absent from metadata).
        filtered = []
        for r in results:
            match = True
            for key, value in filters.items():
                if key == "folder_prefix":
                    if not str(r.get("doc_path", "")).startswith(value):
                        match = False
                        break
                    continue
                meta = r.get("metadata", {})
                if key in meta:
                    if isinstance(value, (list, tuple)):
                        if meta[key] not in value:
                            match = False
                            break
                    elif meta[key] != value:
                        match = False
                        break
            if match:
                filtered.append(r)
        return filtered
