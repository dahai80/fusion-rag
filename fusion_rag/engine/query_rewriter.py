"""QueryRewriter — LLM-powered query expansion and refinement.

callers: routes.py search/ask endpoints, HybridSearch pipeline
API: QueryRewriter.rewrite(query, history=None, mode="hyde") -> str | list[str]
schema: input query str, output rewritten query or list of expanded queries
user instruction: "按照你的方案和计划落地所有phase阶段的需求"
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

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

    def __init__(self, mlx_url: str = "http://localhost:11434/v1",
                 model: str = "qwen3.5-9b", enabled: bool = True):
        self.mlx_url = mlx_url.rstrip("/")
        self.model = model
        self.enabled = enabled

    async def rewrite(self, query: str, history: list[dict[str, str]] | None = None,
                      mode: str = "hyde") -> str | list[str]:
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
            logger.warning("Query rewrite failed (mode=%s): %s", mode, e)
            return query

    async def _hyde(self, query: str) -> str:
        prompt = HYDE_PROMPT.format(query=query)
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{self.mlx_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 300,
                    "temperature": 0.3,
                },
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            if content:
                logger.info("HyDE rewrite: %s -> %s", query[:50], content[:50])
            return content if content else query

    async def _expand(self, query: str) -> list[str]:
        prompt = EXPAND_PROMPT.format(query=query)
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{self.mlx_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 200,
                    "temperature": 0.5,
                },
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
        hist_text = "\n".join(
            f"{h.get('role', 'user')}: {h.get('content', '')}"
            for h in history[-6:]
        )
        prompt = CONDENSE_PROMPT.format(history=hist_text, query=query)
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{self.mlx_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 200,
                    "temperature": 0.0,
                },
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
                    line = line[len(prefix):].strip()
                    break
            if line:
                variants.append(line)
        return variants[:3]
