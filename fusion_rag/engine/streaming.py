"""Streaming SSE support and metadata extraction for RAG responses."""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
from fusion_core.http_client import get_async_client, with_retry

from .llm_errors import LLMUnavailable

logger = logging.getLogger(__name__)


class SSEStreamer:
    """Server-Sent Events streaming for RAG responses."""

    @staticmethod
    async def stream_response(question: str, context: str, mlx_url: str = "http://127.0.0.1:11432/v1") -> str:
        """Stream a RAG response as SSE events. Keeps raw httpx.stream (with_retry does not apply to SSE streams)."""
        messages = [
            {"role": "system", "content": "Answer based on the context. Cite sources."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ]
        events = []
        async with (
            httpx.AsyncClient(timeout=60.0) as client,
            client.stream(
                "POST",
                f"{mlx_url}/chat/completions",
                json={
                    "model": "qwen3.5-9b",
                    "messages": messages,
                    "max_tokens": 4096,
                    "stream": True,
                },
            ) as resp,
        ):
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data != "[DONE]":
                        try:
                            chunk = json.loads(data)
                            content = chunk["choices"][0]["delta"].get("content", "")
                            if content:
                                events.append(f"data: {json.dumps({'content': content})}\n\n")
                        except (json.JSONDecodeError, KeyError):
                            pass
        events.append("data: [DONE]\n\n")
        return "".join(events)


class MetadataExtractor:
    """Automatically extracts metadata from documents using LLM."""

    def __init__(self, mlx_url: str = "http://127.0.0.1:11432/v1"):
        self.mlx_url = mlx_url.rstrip("/")

    async def _get_client(self) -> httpx.AsyncClient:
        # A8: pool is the single source of truth (dedup + is_closed check).
        # Drop the self._client check-then-assign cache that could hold a
        # pool-evicted/closed reference.
        return get_async_client(self.mlx_url, timeout=30.0)

    async def aclose(self) -> None:
        # Pool-managed: nothing to close here.
        return None

    async def extract(self, text: str, doc_name: str = "") -> dict[str, Any]:
        """Extract metadata from document text."""
        prompt = (
            f"Extract metadata from the following document. "
            f"Return ONLY a JSON object with these fields:\n"
            f"- title: document title\n"
            f"- author: author if mentioned\n"
            f"- date: date if mentioned\n"
            f"- language: detected language\n"
            f"- topics: 2-5 key topics as array\n"
            f"- summary: one sentence summary\n\n"
            f"Document: {text[:2000]}\n\n"
            f"JSON:"
        )
        try:
            client = await self._get_client()
            resp = await with_retry(
                lambda: client.post(
                    f"{self.mlx_url}/chat/completions",
                    json={
                        "model": "qwen3.5-9b",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 512,
                        "temperature": 0.1,
                    },
                ),
                retries=2,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            if not content or not content.strip():
                logger.warning("MetadataExtractor empty content, doc_name=%s", doc_name)
                raise ValueError("empty_content")
            import re

            match = re.search(r"\{.*\}", content, re.DOTALL)
            if not match:
                logger.warning("MetadataExtractor no JSON in content, doc_name=%s", doc_name)
                raise ValueError("no_json")
            return json.loads(match.group())
        except Exception as e:
            # L1: do not return fabricated default metadata — that makes LLM
            # failure look like a successful extraction with placeholder
            # fields. Propagate so the caller knows extraction is unavailable.
            logger.warning("Metadata extraction failed (propagating, no fabricated metadata): %s", e)
            raise LLMUnavailable("metadata extraction failed") from e
