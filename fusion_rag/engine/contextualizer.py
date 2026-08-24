"""Contextualizer — Anthropic Contextual Retrieval for chunk contextualization.

callers: KnowledgeBase intake flow, routes.py upload_document/scan_directory
API: Contextualizer.contextualize(chunks, doc_text) -> list[dict]
schema: chunks gain "context" field (str), embedded as context+text
user instruction: "按照你的方案和计划落地所有phase阶段的需求"
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from fusion_core.http_client import get_async_client, with_retry

logger = logging.getLogger(__name__)

CONTEXT_PROMPT = (
    "<document>\n{doc_text}\n</document>\n"
    "Here is the chunk we want to situate within the whole document\n"
    "<chunk>\n{chunk_text}\n</chunk>\n"
    "Please give a short succinct context to situate this chunk within the "
    "overall document for the purposes of improving search retrieval of the chunk. "
    "Answer only with the succinct context and nothing else."
)


class Contextualizer:
    """Generates context for chunks using Anthropic's Contextual Retrieval approach."""

    def __init__(
        self,
        mlx_url: str = "http://127.0.0.1:11432/v1",
        model: str = "qwen3.5-9b",
        enabled: bool = True,
        api_key: str = "",
    ):
        self.mlx_url = mlx_url.rstrip("/")
        self.model = model
        self.enabled = enabled
        self.api_key = api_key
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = get_async_client(self.mlx_url, timeout=30.0)
            logger.debug("Pooled httpx.AsyncClient via fusion_core, base=%s", self.mlx_url)
        return self._client

    def _auth_headers(self) -> dict[str, str]:
        if self.api_key:
            return {"Authorization": f"Bearer {self.api_key}"}
        return {}

    async def aclose(self) -> None:
        if self._client is not None:
            logger.debug("Releasing reference to pooled httpx.AsyncClient (pool-managed, not closed)")
        self._client = None

    async def contextualize(self, chunks: list[dict[str, Any]], doc_text: str) -> list[dict[str, Any]]:
        if not self.enabled or not doc_text or not chunks:
            return chunks
        doc_truncated = doc_text[:8000]
        client = await self._get_client()
        for chunk in chunks:
            try:
                context = await self._generate_context(client, chunk.get("text", ""), doc_truncated)
                chunk["context"] = context
            except Exception as e:
                logger.warning(
                    "Context generation failed for chunk %s: %s",
                    chunk.get("id", "?"),
                    e,
                )
                chunk["context"] = ""
        return chunks

    async def _generate_context(self, client: httpx.AsyncClient, chunk_text: str, doc_text: str) -> str:
        prompt = CONTEXT_PROMPT.format(
            doc_text=doc_text,
            chunk_text=chunk_text,
        )
        resp = await with_retry(
            lambda: client.post(
                f"{self.mlx_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 100,
                    "temperature": 0.0,
                },
                headers=self._auth_headers(),
            ),
            retries=2,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        if not content:
            logger.warning("Contextualizer empty content, chunk_text=%s", chunk_text[:50])
            raise ValueError("empty_content")
        return content[:200]
