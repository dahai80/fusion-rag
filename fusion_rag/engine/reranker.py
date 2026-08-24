"""Reranker and HybridSearch — batch reranking and multi-strategy fusion.

callers: routes.py search/ask endpoints, KnowledgeBase pipeline
API: Reranker.rerank(query, documents, top_k), HybridSearch.search(query_vector, query_text, ...)
schema: documents list[dict] with id/text/score keys, HybridSearch supports alpha and rrf fusion methods
user instruction: "按照你的方案和计划落地所有phase阶段的需求"
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx
from fusion_core.http_client import get_async_client, with_retry

logger = logging.getLogger(__name__)


class Reranker:
    """Reranks search results by relevance to query.

    Uses batch LLM scoring — single API call for all documents.
    """

    def __init__(self, mlx_url: str = "http://127.0.0.1:11432/v1", model: str = "qwen3.5-9b", batch_size: int = 20):
        self.mlx_url = mlx_url.rstrip("/")
        self.model = model
        self.batch_size = batch_size
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = get_async_client(self.mlx_url, timeout=30.0)
            logger.debug("Pooled httpx.AsyncClient via fusion_core, base=%s", self.mlx_url)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            logger.debug("Releasing reference to pooled httpx.AsyncClient (pool-managed, not closed)")
        self._client = None

    async def rerank(self, query: str, documents: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
        if not documents:
            return []
        all_scored = []
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i : i + self.batch_size]
            scores = await self._batch_score(query, batch)
            for doc, score in zip(batch, scores):
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
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            if not content:
                logger.warning("Reranker empty content, query=%s", query[:50])
                raise ValueError("empty_content")
            return self._parse_scores(content, len(docs))
        except Exception as e:
            logger.warning("Batch rerank failed, fallback to original scores: %s", e)
            return [doc.get("score", 5.0) for doc in docs]

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
            return float(content[:4])
        except Exception:
            return 5.0


class HybridSearch:
    """Combines vector similarity and keyword search with weighted or RRF fusion."""

    def __init__(self, vector_store, alpha: float = 0.7, method: str = "alpha"):
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
        vector_results = self.vector_store.search(query_vector, top_k=top_k * 2, threshold=0.0)
        keyword_results = self.vector_store.keyword_search(query_text, top_k=top_k * 2)

        if filters:
            vector_results = self._apply_filters(vector_results, filters)
            keyword_results = self._apply_filters(keyword_results, filters)

        if self.method == "rrf":
            return self._rrf_fusion(vector_results, keyword_results, top_k, threshold)
        return self._alpha_fusion(vector_results, keyword_results, top_k, threshold)

    def _alpha_fusion(
        self, vector_results: list[dict], keyword_results: list[dict], top_k: int, threshold: float
    ) -> list[dict[str, Any]]:
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
        filtered = []
        for r in results:
            meta = r.get("metadata", {})
            match = True
            for key, value in filters.items():
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
