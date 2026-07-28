"""Retrieval strategies — MMR, context compression, and fusion retrievers."""
from __future__ import annotations

import math
from typing import Any


class MMRRetriever:
    """Maximum Marginal Relevance retriever — balances relevance and diversity.

    Selects documents that are both relevant to the query and diverse from each other.
    """

    def __init__(self, vector_store, lambda_param: float = 0.7, k: int = 10):
        self.vector_store = vector_store
        self.lambda_param = lambda_param  # 0.7 = more relevance, 0.3 = more diversity
        self.k = k

    async def search(self, query_vector: list[float], top_k: int = 10) -> list[dict[str, Any]]:
        """MMR search: select diverse top-k results."""
        candidates = self.vector_store.search(query_vector, top_k=self.k * 3)
        if not candidates:
            return []

        selected = []
        remaining = list(candidates)

        while len(selected) < top_k and remaining:
            mmr_scores = []
            for candidate in remaining:
                # Relevance score
                rel_score = candidate.get("score", 0)

                # Diversity penalty: max similarity to already selected
                div_penalty = 0.0
                if selected:
                    for sel in selected:
                        sim = self._cosine_sim(
                            candidate.get("vector", []),
                            sel.get("vector", []),
                        )
                        div_penalty = max(div_penalty, sim)

                mmr = self.lambda_param * rel_score - (1 - self.lambda_param) * div_penalty
                mmr_scores.append(mmr)

            # Select the best
            best_idx = mmr_scores.index(max(mmr_scores))
            best = remaining.pop(best_idx)
            best["mmr_score"] = mmr_scores[best_idx]
            selected.append(best)

        return selected[:top_k]

    @staticmethod
    def _cosine_sim(a: list[float], b: list[float]) -> float:
        if not a or not b:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)


class ContextCompressionRetriever:
    """Compresses retrieved documents to fit within a token budget."""

    def __init__(self, base_retriever, max_tokens: int = 2048):
        self.base_retriever = base_retriever
        self.max_tokens = max_tokens

    async def search(self, query_vector: list[float], query_text: str = "",
                     top_k: int = 10) -> list[dict[str, Any]]:
        """Retrieve and compress results to fit token budget."""
        results = await self.base_retriever.search(query_vector, top_k=top_k)
        return self._compress(results, query_text)

    def _compress(self, results: list[dict], query: str) -> list[dict]:
        """Compress results by truncating and removing low-value content."""
        total_tokens = 0
        compressed = []
        for r in results:
            text = r.get("text", "")
            tokens = len(text) // 4
            if total_tokens + tokens > self.max_tokens:
                allowed_chars = (self.max_tokens - total_tokens) * 4
                r["text"] = text[:allowed_chars] + "..."
                r["compressed"] = True
                compressed.append(r)
                break
            total_tokens += tokens
            compressed.append(r)
        return compressed


class FusionRetriever:
    """Combines multiple retrievers with weighted scoring."""

    def __init__(self, retrievers: list[tuple[str, Any, float]]):
        self.retrievers = retrievers  # [(name, retriever, weight), ...]

    async def search(self, query_vector: list[float], query_text: str = "",
                     top_k: int = 10) -> list[dict[str, Any]]:
        """Fuse results from multiple retrievers with weighted scores."""
        all_scores: dict[str, float] = {}
        all_results: dict[str, dict] = {}

        for name, retriever, weight in self.retrievers:
            results = await retriever.search(query_vector, query_text, top_k=top_k)
            for r in results:
                rid = r.get("id", "")
                all_scores[rid] = all_scores.get(rid, 0) + weight * r.get("score", 0)
                if rid not in all_results:
                    all_results[rid] = r

        ranked = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        return [all_results[rid] for rid, _ in ranked[:top_k]]
