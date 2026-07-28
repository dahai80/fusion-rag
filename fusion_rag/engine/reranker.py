"""Reranker — reranks search results to improve precision using fusion-mlx."""
from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class Reranker:
    """Reranks search results by relevance to query.

    Uses fusion-mlx's chat API to score document-query relevance.
    """

    def __init__(self, mlx_url: str = "http://localhost:11434/v1", model: str = "qwen3.5-9b"):
        self.mlx_url = mlx_url.rstrip("/")
        self.model = model

    async def rerank(self, query: str, documents: list[dict[str, Any]],
                     top_k: int = 5) -> list[dict[str, Any]]:
        """Rerank documents by relevance to query.

        Args:
            query: The search query.
            documents: List of documents with 'id' and 'text' keys.
            top_k: Number of top results to return.

        Returns:
            Documents sorted by relevance score, with score added.
        """
        if not documents:
            return []

        scored = []
        async with httpx.AsyncClient(timeout=30.0) as client:
            for doc in documents:
                text = doc.get("text", "")[:2000]
                score = await self._score_relevance(client, query, text)
                doc["score"] = score
                scored.append(doc)

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    async def _score_relevance(self, client: httpx.AsyncClient, query: str, text: str) -> float:
        """Score the relevance of a document to a query (0-10)."""
        prompt = (
            f"Rate the relevance of the following document to the query "
            f"on a scale of 0 to 10 (0=completely irrelevant, 10=perfect match).\n\n"
            f"Query: {query}\n\n"
            f"Document: {text[:1500]}\n\n"
            f"Relevance score (0-10):"
        )
        try:
            resp = await client.post(f"{self.mlx_url}/chat/completions", json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 10,
                "temperature": 0.0,
            })
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            return float(content[:4])
        except Exception:
            return 5.0  # Neutral score on failure


class HybridSearch:
    """Combines vector similarity and keyword search with weighted fusion."""

    def __init__(self, vector_store, alpha: float = 0.7):
        self.vector_store = vector_store
        self.alpha = alpha

    async def search(self, query_vector: list[float], query_text: str,
                     top_k: int = 10, threshold: float = 0.0,
                     filters: dict | None = None) -> list[dict[str, Any]]:
        """Hybrid search with weighted fusion of vector and keyword results.

        Args:
            query_vector: Vector embedding of the query.
            query_text: Raw query text for keyword search.
            top_k: Number of results to return.
            threshold: Minimum score threshold.
            filters: Optional metadata filters.

        Returns:
            Ranked results with fused scores.
        """
        # Vector search
        vector_results = self.vector_store.search(query_vector, top_k=top_k * 2, threshold=0.0)
        # Keyword search
        keyword_results = self.vector_store.keyword_search(query_text, top_k=top_k * 2)

        # Apply filters if provided
        if filters:
            vector_results = self._apply_filters(vector_results, filters)
            keyword_results = self._apply_filters(keyword_results, filters)

        # Fuse scores
        scores: dict[str, float] = {}
        for r in vector_results:
            rid = r.get("id", "")
            scores[rid] = scores.get(rid, 0) + self.alpha * r.get("score", 0)

        for r in keyword_results:
            rid = r.get("id", "")
            scores[rid] = scores.get(rid, 0) + (1 - self.alpha) * r.get("score", 0)

        # Sort by fused score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ranked = [{"id": rid, "score": s} for rid, s in ranked if s >= threshold]

        # Enrich with full data from vector results
        result_map = {r["id"]: r for r in vector_results}
        for r in ranked:
            if r["id"] in result_map:
                r.update(result_map[r["id"]])
        return ranked[:top_k]

    @staticmethod
    def _apply_filters(results: list[dict], filters: dict) -> list[dict]:
        """Apply metadata filters to results."""
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