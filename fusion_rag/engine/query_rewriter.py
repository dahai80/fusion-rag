"""QueryRewriter — LLM-powered query expansion and refinement.

callers: routes.py search/ask endpoints, HybridSearch pipeline
API: QueryRewriter.rewrite(query, history=None, mode="hyde") -> str | list[str]
schema: input query str, output rewritten query or list of expanded queries
user instruction: "按照你的方案和计划落地所有phase阶段的需求"
"""

from __future__ import annotations

import logging

import httpx
from fusion_core.http_client import get_async_client, with_retry

from .llm_errors import LLMUnavailable

logger = logging.getLogger(__name__)

HYDE_PROMPT = (
    "Write a detailed passage that answers the following question. "
    "Write as if you are explaining to someone who needs accurate information. "
    "Do not include the question itself, only the answer passage.\n\n"
    "Question: {query}"
)

EXPAND_PROMPT = (
    "Given the following query, generate 3 different reformulations that capture "
    "the same information need but use different phrasing and terminology. "
    "Each reformulation should be on its own line, numbered 1-3. "
    "Do not include any other text.\n\n"
    "Query: {query}"
)

CONDENSE_PROMPT = (
    "Given the conversation history and the latest question, "
    "reformulate the question to be a standalone question that captures "
    "the full context. Output ONLY the standalone question, nothing else.\n\n"
    "{history}\n\nLatest question: {query}"
)


class QueryRewriter:
    """Rewrites queries using LLM for better retrieval."""

    def __init__(self, mlx_url: str = "http://127.0.0.1:11432/v1", model: str = "qwen3.5-9b", enabled: bool = True):
        self.mlx_url = mlx_url.rstrip("/")
        self.model = model
        self.enabled = enabled

    async def _get_client(self) -> httpx.AsyncClient:
        # A8: get_async_client is the single source of truth — it dedups by
        # loop+base_url and skips a closed entry. The old self._client cache
        # was a check-then-assign with no lock and could hold a reference to a
        # client the pool had since LRU-evicted and closed. Drop the cache;
        # resolve from the pool every call.
        return get_async_client(self.mlx_url, timeout=30.0)

    async def aclose(self) -> None:
        # Pool-managed: nothing to close here. Kept for callers that release
        # per-instance (pool eviction handles actual aclose).
        return None

    async def rewrite(
        self, query: str, history: list[dict[str, str]] | None = None, mode: str = "hyde"
    ) -> str | list[str]:
        if not self.enabled or not query.strip():
            return query
        try:
            if mode == "hyde":
                return await self._hyde(query)
            if mode == "expand":
                return await self._expand(query)
            if mode == "condense":
                return await self._condense(query, history or [])
            logger.warning("Unknown rewrite mode '%s', returning original query", mode)
            return query
        except Exception as e:
            # L1: do not silently return the original query — that makes LLM
            # failure look like a successful (no-op) rewrite, degrading
            # retrieval quality invisibly. Propagate so the route layer logs
            # the degradation and decides: proceed with original query or 503.
            logger.warning("Query rewrite failed (mode=%s, propagating): %s", mode, e)
            raise LLMUnavailable(f"query rewrite failed (mode={mode})") from e

    async def _hyde(self, query: str) -> str:
        prompt = HYDE_PROMPT.format(query=query)
        client = await self._get_client()
        resp = await with_retry(
            lambda: client.post(
                f"{self.mlx_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 300,
                    "temperature": 0.3,
                },
            ),
            retries=2,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        if content:
            logger.info("HyDE rewrite: %s -> %s", query[:50], content[:50])
        return content if content else query

    async def _expand(self, query: str) -> list[str]:
        prompt = EXPAND_PROMPT.format(query=query)
        client = await self._get_client()
        resp = await with_retry(
            lambda: client.post(
                f"{self.mlx_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 200,
                    "temperature": 0.5,
                },
            ),
            retries=2,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        variants = self._parse_variants(content)
        if variants:
            result = [query] + variants
            logger.info("Query expansion: %d variants for '%s'", len(variants), query[:50])
            return result
        return [query]

    async def _condense(self, query: str, history: list[dict[str, str]]) -> str:
        if not history:
            return query
        hist_text = "\n".join(f"{h.get('role', 'user')}: {h.get('content', '')}" for h in history[-6:])
        prompt = CONDENSE_PROMPT.format(history=hist_text, query=query)
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
        if content:
            logger.info("Condensed: %s -> %s", query[:50], content[:50])
        return content if content else query

    @staticmethod
    def _parse_variants(content: str) -> list[str]:
        variants = []
        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue
            for prefix in ("1.", "2.", "3.", "1)", "2)", "3)"):
                if line.startswith(prefix):
                    line = line[len(prefix) :].strip()
                    break
            if line:
                variants.append(line)
        return variants[:3]
