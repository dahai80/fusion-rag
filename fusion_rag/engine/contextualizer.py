"""Contextualizer — Anthropic Contextual Retrieval for chunk contextualization.

callers: KnowledgeBase intake flow, routes.py upload_document/scan_directory
API: Contextualizer.contextualize(chunks, doc_text) -> list[dict]
schema: chunks gain "context" field (str), embedded as context+text
user instruction: "按照你的方案和计划落地所有phase阶段的需求"
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
from fusion_core.http_client import get_async_client, with_retry

from .llm_errors import LLMUnavailable

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
        max_context_chars: int = 300,
    ):
        self.mlx_url = mlx_url.rstrip("/")
        self.model = model
        self.enabled = enabled
        self.api_key = api_key
        self.max_context_chars = max_context_chars

    async def _get_client(self) -> httpx.AsyncClient:
        # A8: pool is the single source of truth (dedup + is_closed check).
        # Drop the self._client check-then-assign cache that could hold a
        # pool-evicted/closed reference.
        return get_async_client(self.mlx_url, timeout=30.0)

    def _auth_headers(self) -> dict[str, str]:
        if self.api_key:
            return {"Authorization": f"Bearer {self.api_key}"}
        return {}

    async def aclose(self) -> None:
        # Pool-managed: nothing to close here.
        return None

    async def contextualize(self, chunks: list[dict[str, Any]], doc_text: str) -> list[dict[str, Any]]:
        if not self.enabled or not doc_text or not chunks:
            return chunks
        # L17: anchor the context window around each chunk instead of always
        # truncating to the document head — a chunk at offset 15000 otherwise
        # gets context from the (unrelated) document beginning.
        client = await self._get_client()
        # P-P1-3: prior loop called _generate_context per chunk SEQUENTIALLY —
        # N chunks = N serial LLM round-trips, contextualize latency = sum not
        # max. A 200-chunk doc at ~0.5s/call blocked ingest for 100s. Fan the
        # per-chunk context calls out concurrently behind a semaphore so the
        # LLM backend isn't flooded (it serializes anyway, but the round-trips
        # overlap instead of queuing one behind another).
        sem = asyncio.Semaphore(4)

        async def _one(chunk: dict[str, Any]) -> str:
            async with sem:
                windowed = self._window_doc_text(doc_text, chunk.get("text", ""))
                return await self._generate_context(client, chunk.get("text", ""), windowed)

        contexts = await asyncio.gather(*[_one(c) for c in chunks], return_exceptions=True)
        failures = 0
        for chunk, result in zip(chunks, contexts):
            if isinstance(result, Exception):
                # L1: record the failure. If EVERY chunk fails the LLM is down
                # — propagate LLMUnavailable so the caller knows contextual
                # retrieval silently degraded to plain retrieval. Per-chunk
                # partial failures (some ok, some not) keep empty context but
                # are counted so the signal is visible.
                failures += 1
                logger.warning(
                    "Context generation failed for chunk %s: %s",
                    chunk.get("id", "?"),
                    result,
                )
                chunk["context"] = ""
            else:
                chunk["context"] = result
        if failures and failures == len(chunks):
            logger.warning("Contextualizer: all %d chunks failed — LLM unavailable", failures)
            raise LLMUnavailable("contextualization failed for all chunks")
        if failures:
            logger.info("Contextualizer: %d/%d chunks had no context (partial)", failures, len(chunks))
        return chunks

    @staticmethod
    def _window_doc_text(doc_text: str, chunk_text: str, half_window: int = 4000) -> str:
        # L17: take ±half_window chars around the chunk's first occurrence so
        # deep chunks get locally relevant context, not the document head.
        if len(doc_text) <= half_window * 2:
            return doc_text
        idx = doc_text.find(chunk_text[:64])
        if idx < 0:
            return doc_text[: half_window * 2]
        start = max(0, idx - half_window)
        end = min(len(doc_text), idx + half_window)
        return doc_text[start:end]

    async def _generate_context(self, client: httpx.AsyncClient, chunk_text: str, doc_text: str) -> str:
        prompt = CONTEXT_PROMPT.format(
            doc_text=doc_text,
            chunk_text=chunk_text,
        )
        import time as _time

        from .metrics import record_llm_latency

        _llm_start = _time.perf_counter()
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
            total_deadline=15.0,
        )
        record_llm_latency("contextualize", (_time.perf_counter() - _llm_start) * 1000)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        if not content:
            # O-P1-3: chunk_text is document PII — keep the warning signal but
            # drop the text snippet so INFO+ logs carry no chunk content.
            logger.warning("Contextualizer empty content (chunk len=%d)", len(chunk_text))
            raise ValueError("empty_content")
        # L17: configurable output cap (was hard 200). Long chunks benefit from
        # more context; callers can pass max_context_chars to tune.
        return content[: self.max_context_chars]
