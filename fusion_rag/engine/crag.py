"""CorrectiveRAG — retrieval grader + bounded rewrite-retry loop (issue #127, R5).

callers: routes_search.ask (opt-in via crag_rounds param / FUSION_RAG_CRAG_ROUNDS)
API: CorrectiveRAG.run(question, retrieve_fn) -> (chunks, trace)
pattern: CRAG (arXiv 2401.15884) / LangGraph agentic RAG — grade retrieved
chunks against the query with a structured LLM judge; when the round is
insufficient, rewrite the question (QueryRewriter) and re-retrieve; merge
round results (dedupe by chunk id, first-seen order wins). Bounded by
max_rounds; every grader/rewrite failure degrades to the current result
(enhancement never breaks retrieval).
user instruction: "按照你的方案和计划落地所有phase阶段的需求"
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from fusion_core.http_client import get_async_client, with_retry

from .llm_errors import LLMUnavailable

logger = logging.getLogger(__name__)

GRADE_PROMPT = (
    "You are a strict relevance grader for a retrieval system.\n"
    "For the user query below, decide for EACH chunk whether it contains "
    "information that directly helps answer the query. A chunk that is "
    "topically related but does NOT help answer the specific question must be "
    "judged irrelevant. When unsure, judge irrelevant.\n\n"
    "Query: {query}\n\n"
    "{chunks}\n\n"
    'Respond ONLY with JSON: {{"relevant": [chunk numbers]}}'
)


class RetrievalGrader:
    """LLM judge that scores retrieved chunks against the query (structured JSON)."""

    def __init__(
        self,
        mlx_url: str = "http://127.0.0.1:11432/v1",
        model: str = "qwen3.5-9b",
        api_key: str = "",
        max_chunks: int = 10,
        chunk_chars: int = 500,
    ):
        self.mlx_url = mlx_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.max_chunks = max_chunks
        self.chunk_chars = chunk_chars

    def _auth_headers(self) -> dict[str, str]:
        if self.api_key:
            return {"Authorization": f"Bearer {self.api_key}"}
        return {}

    async def _get_client(self):
        return get_async_client(self.mlx_url, timeout=30.0)

    def _parse_relevant(self, content: str, n: int) -> set[int]:
        # Tolerant JSON extraction: grade output is a single JSON object but
        # small models wrap it in prose or fences. Unparseable / out-of-range
        # entries degrade to "nothing relevant" (retry triggers) only when the
        # whole object parses; a parse failure raises LLMUnavailable so the
        # caller logs the degradation instead of silently retrying.
        text = content.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise LLMUnavailable(f"grader output not JSON: {content[:80]!r}")
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError as e:
            raise LLMUnavailable("grader output unparseable JSON") from e
        ids = data.get("relevant", []) if isinstance(data, dict) else []
        out = set()
        for i in ids:
            if isinstance(i, int) and 0 <= i < n:
                out.add(i)
        return out

    async def grade(self, query: str, chunks: list[dict[str, Any]]) -> set[int]:
        """Return indices of chunks judged relevant. Raises LLMUnavailable on failure."""
        if not chunks:
            return set()
        listed = "\n".join(
            f"{i}. {c.get('text', '')[: self.chunk_chars]}" for i, c in enumerate(chunks[: self.max_chunks])
        )
        prompt = GRADE_PROMPT.format(query=query, chunks=listed)
        client = await self._get_client()
        try:
            resp = await with_retry(
                lambda: client.post(
                    f"{self.mlx_url}/chat/completions",
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 200,
                        "temperature": 0.0,
                    },
                    headers=self._auth_headers(),
                ),
                retries=2,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
        except LLMUnavailable:
            raise
        except Exception as e:
            raise LLMUnavailable(f"grader LLM call failed: {e}") from e
        relevant = self._parse_relevant(content or "", min(len(chunks), self.max_chunks))
        logger.debug("grader: query=%r relevant=%s of %d chunks", query[:50], sorted(relevant), len(chunks))
        return relevant


class CorrectiveRAG:
    """Bounded grade → rewrite → re-retrieve loop over a retrieval callable."""

    def __init__(
        self,
        grader: RetrievalGrader,
        rewriter: Any,
        max_rounds: int = 2,
        min_relevant: int = 1,
        rewrite_mode: str = "expand",
    ):
        self.grader = grader
        self.rewriter = rewriter
        self.max_rounds = max(0, max_rounds)
        self.min_relevant = max(1, min_relevant)
        self.rewrite_mode = rewrite_mode

    async def run(
        self,
        question: str,
        retrieve_fn: Callable[[str], Awaitable[list[dict[str, Any]]]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        # Round 0 = the caller's normal single-shot retrieval. Each retry round
        # rewrites the question and re-retrieves; merged chunks dedupe by chunk
        # id (first-seen wins — round-0 ordering is the fused ranking). Trace
        # entries land in the /ask response and the citation-audit rewrite
        # chain (one entry per rewritten query).
        chunks = await retrieve_fn(question)
        trace: list[dict[str, Any]] = [
            {"round": 0, "query": question, "n_chunks": len(chunks), "action": "initial"}
        ]
        if not self.max_rounds or not chunks:
            return chunks, trace

        seen: set[str] = set()
        merged: list[dict[str, Any]] = []
        for c in chunks:
            cid = c.get("chunk_id") or c.get("id") or ""
            if cid and cid in seen:
                continue
            seen.add(cid)
            merged.append(c)
        query = question

        for rnd in range(1, self.max_rounds + 1):
            try:
                relevant = await self.grader.grade(query, merged)
            except LLMUnavailable as e:
                logger.warning("CRAG grader unavailable at round %d, keeping current results: %s", rnd, e)
                trace.append({"round": rnd, "action": "grader_unavailable", "error": str(e)})
                break
            n_rel = len(relevant)
            if n_rel >= self.min_relevant:
                trace.append(
                    {"round": rnd, "action": "sufficient", "n_relevant": n_rel, "n_chunks": len(merged)}
                )
                break
            try:
                rewritten = await self.rewriter.rewrite(query, mode=self.rewrite_mode)
            except LLMUnavailable as e:
                logger.warning("CRAG rewrite unavailable at round %d, keeping current results: %s", rnd, e)
                trace.append({"round": rnd, "action": "rewrite_unavailable", "error": str(e)})
                break
            if not isinstance(rewritten, str) or not rewritten.strip() or rewritten == query:
                trace.append({"round": rnd, "action": "rewrite_noop", "n_relevant": n_rel})
                break
            query = rewritten
            new_chunks = await retrieve_fn(query)
            added = 0
            for c in new_chunks:
                cid = c.get("chunk_id") or c.get("id") or ""
                if cid and cid in seen:
                    continue
                seen.add(cid)
                merged.append(c)
                added += 1
            trace.append(
                {
                    "round": rnd,
                    "action": "retried",
                    "query": query,
                    "n_relevant": n_rel,
                    "n_new": added,
                    "n_chunks": len(merged),
                }
            )
            logger.info(
                "CRAG round %d: %d relevant of %d, rewrote query, +%d new chunks",
                rnd,
                n_rel,
                len(merged) - added,
                added,
            )
        return merged, trace
