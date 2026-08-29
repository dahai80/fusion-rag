"""Multi-turn RAG — conversational retrieval with history and token management.

callers: routes.py ask endpoint, KnowledgeBase RAG pipeline
API: MultiTurnRAG.ask(question, context, session_id), .clear_history(), .token_count()
schema: history list[dict], token_budget int, sessions dict
user instruction: "按照你的方案和计划落地所有phase阶段的需求"
"""

from __future__ import annotations

import logging
import re
from typing import Any

import httpx
from fusion_core.http_client import get_async_client, with_retry

from .llm_errors import LLMUnavailable

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN_EN = 4.0
CHARS_PER_TOKEN_ZH = 2.0


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    chinese_chars = len(re.findall(r"[一-鿿]", text))
    other_chars = len(text) - chinese_chars
    return int(chinese_chars / CHARS_PER_TOKEN_ZH + other_chars / CHARS_PER_TOKEN_EN) + 1


class MultiTurnRAG:
    """Multi-turn conversational RAG with history tracking and token budget."""

    def __init__(
        self,
        mlx_url: str = "http://127.0.0.1:11432/v1",
        model: str = "qwen3.5-9b",
        token_budget: int | None = None,
        max_history_turns: int | None = None,
        system_prompt: str = "",
    ):
        # D4: token_budget/max_history default from RuntimeConfig (env-overridable)
        # when the caller omits them. Was hardcoded 8192 / 10 with no override.
        from .runtime_config import get_runtime_config

        cfg = get_runtime_config()
        if token_budget is None:
            token_budget = cfg.rag_token_budget
        if max_history_turns is None:
            max_history_turns = cfg.rag_max_history_turns
        self.mlx_url = mlx_url.rstrip("/")
        self.model = model
        self.token_budget = token_budget
        self.max_history_turns = max_history_turns
        self._system_prompt = system_prompt
        self._history: list[dict] = []
        self._sessions: dict[str, list[dict]] = {}

    async def _get_client(self) -> httpx.AsyncClient:
        # A8: pool is the single source of truth (dedup + is_closed check).
        # Drop the self._client check-then-assign cache that could hold a
        # pool-evicted/closed reference.
        return get_async_client(self.mlx_url, timeout=60.0)

    async def aclose(self) -> None:
        # Pool-managed: nothing to close here.
        return None

    async def ask(self, question: str, context: str = "", session_id: str = "") -> dict[str, Any]:
        history = self._resolve_history(session_id)
        # L4: trim BEFORE building messages — otherwise this turn consumes an
        # unbounded history (the token-budget break in _build_messages is the
        # only guard) and trimming only takes effect next turn.
        self._trim_history(history)
        messages = self._build_messages(question, context, history)

        try:
            client = await self._get_client()
            resp = await with_retry(
                lambda: client.post(
                    f"{self.mlx_url}/chat/completions",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "max_tokens": 4096,
                        "temperature": 0.3,
                    },
                ),
                retries=2,
                total_deadline=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
            answer = data["choices"][0]["message"]["content"]
            if not answer or not answer.strip():
                logger.warning("MultiTurnRAG empty content, question=%s", question[:50])
                raise ValueError("empty_content")
            usage = data.get("usage", {})
        except Exception as e:
            # L1/L3: do NOT return "Error: {e}" as a 200 answer — that makes an
            # LLM failure look like a valid (if terse) response AND gets written
            # into history as an assistant message, poisoning subsequent turns.
            # Propagate so the caller maps to an error response; history stays
            # unchanged (no user/assistant pair appended on failure).
            logger.error("MultiTurnRAG ask failed (propagating, not poisoning history): %s", e)
            raise LLMUnavailable("multi-turn ask failed") from e

        # Success — record the turn. Only real answers enter history.
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})

        result = {
            "answer": answer,
            "history_length": len(history) // 2,
            "token_estimate": self._history_token_count(history),
        }
        if usage:
            result["prompt_tokens"] = usage.get("prompt_tokens", 0)
            result["completion_tokens"] = usage.get("completion_tokens", 0)
            result["total_tokens"] = usage.get("total_tokens", 0)
        return result

    def token_count(self, session_id: str = "") -> int:
        history = self._resolve_history(session_id)
        return self._history_token_count(history)

    def clear_history(self, session_id: str = "") -> None:
        if session_id and session_id in self._sessions:
            del self._sessions[session_id]
        elif not session_id:
            self._history.clear()
            self._sessions.clear()

    def _resolve_history(self, session_id: str) -> list[dict]:
        if session_id:
            if session_id not in self._sessions:
                self._sessions[session_id] = []
            return self._sessions[session_id]
        return self._history

    def _build_messages(self, question: str, context: str, history: list[dict]) -> list[dict]:
        import os

        messages = []
        if context:
            prompt = self._system_prompt or os.environ.get(
                "FUSION_RAG_SYSTEM_PROMPT",
                "You are a knowledge base assistant. Answer based on the provided context. "
                "Cite sources. If the context doesn't contain the answer, say so.",
            )
            messages.append({"role": "system", "content": prompt})

        # L2: explicit 3-segment build — system → history (chronological) → user.
        # Before: `messages.insert(len(messages) - 0 if not context else 1, h)`
        # — `len-0` is a no-op, so without context history was appended at the
        # END (then a user message appended after it), producing reversed
        # history plus a duplicated user question. Iterate history in forward
        # order, within the token budget, so the earliest affordable turns drop
        # first and the most-recent context is kept.
        remaining = self.token_budget - estimate_tokens(context) - 500
        # Keep the most-recent turns that fit: scan from newest to oldest to
        # find the cut, then append the kept slice in chronological order.
        kept_rev = []
        for h in reversed(history):
            h_tokens = estimate_tokens(h.get("content", ""))
            if remaining - h_tokens < 0:
                break
            kept_rev.append(h)
            remaining -= h_tokens
        for h in reversed(kept_rev):
            messages.append(h)

        if context:
            messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"})
        else:
            messages.append({"role": "user", "content": question})

        return messages

    def _trim_history(self, history: list[dict]) -> None:
        max_entries = self.max_history_turns * 2
        if len(history) > max_entries:
            del history[: len(history) - max_entries]

    @staticmethod
    def _history_token_count(history: list[dict]) -> int:
        return sum(estimate_tokens(h.get("content", "")) for h in history)


class DocumentChain:
    """Processes multiple documents with chain strategies."""

    # A7: fusion_core's client pool keys by loop+base_url only — the per-call
    # `timeout` passed to get_async_client is set by whichever call hits the
    # pool first for that base_url, then reused by every later caller on the
    # same loop. To keep that benign (no call needs a shorter timeout than the
    # pool's), every _call in this class uses one shared timeout. Tracked as
    # an upstream issue on fusion-core (pool key should include timeout).
    _CALL_TIMEOUT = 60.0

    @staticmethod
    async def _call(mlx_url: str, messages: list[dict], max_tokens: int, timeout: float | None = None) -> str:
        url = mlx_url.rstrip("/")
        # A7: collapse to the shared timeout so first-call-wins can't starve a
        # caller that asked for more. Callers may still pass a value; we ignore
        # a smaller one to avoid the pool-pin hazard (logged once).
        effective = DocumentChain._CALL_TIMEOUT
        if timeout is not None and timeout > effective:
            effective = timeout
        client = get_async_client(url, timeout=effective)
        resp = await with_retry(
            lambda: client.post(
                f"{url}/chat/completions",
                json={
                    "model": "qwen3.5-9b",
                    "messages": messages,
                    "max_tokens": max_tokens,
                },
            ),
            retries=2,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        if not content or not content.strip():
            logger.warning("DocumentChain empty content, messages=%d", len(messages))
            raise ValueError("empty_content")
        return content

    @staticmethod
    async def stuff(docs: list[str], query: str, mlx_url: str = "http://127.0.0.1:11432/v1") -> str:
        """Stuff all docs into a single prompt."""
        context = "\n\n".join(docs)
        return await DocumentChain._call(
            mlx_url,
            [{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}],
            4096,
        )

    @staticmethod
    async def refine(docs: list[str], query: str, mlx_url: str = "http://127.0.0.1:11432/v1") -> str:
        """Refine answer iteratively with each document."""
        answer = ""
        for doc in docs:
            prompt = f"Previous answer: {answer}\n\nNew document: {doc}\n\nQuestion: {query}\n\nRefine the answer."
            if not answer:
                prompt = f"Document: {doc}\n\nQuestion: {query}\n\nAnswer:"
            answer = await DocumentChain._call(
                mlx_url,
                [{"role": "user", "content": prompt}],
                4096,
            )
        return answer

    @staticmethod
    async def map_reduce(docs: list[str], query: str, mlx_url: str = "http://127.0.0.1:11432/v1") -> str:
        """Map each doc, then reduce to final answer."""
        import asyncio

        # P5: bound concurrent map-phase LLM calls. An unbounded gather fires
        # one chat/completions per doc (100 docs = 100 concurrent), and with
        # with_retry(retries=2) each can triple on a 429/5xx — a retry storm
        # that overloads MLX and cascades. Semaphore(4) caps in-flight maps;
        # the reduce call runs after the gate releases.
        sem = asyncio.Semaphore(4)

        async def map_doc(doc: str) -> str:
            async with sem:
                # A7: no per-call timeout — _call uses the shared _CALL_TIMEOUT
                # (60s) so the map phase can't pin the pool to a shorter value
                # that would starve the reduce call.
                return await DocumentChain._call(
                    mlx_url,
                    [{"role": "user", "content": f"Document: {doc}\n\nQuestion: {query}\n\nSummary:"}],
                    1024,
                )

        summaries = await asyncio.gather(*[map_doc(d) for d in docs])
        combined = "\n\n".join(summaries)

        return await DocumentChain._call(
            mlx_url,
            [{"role": "user", "content": f"Summaries:\n{combined}\n\nQuestion: {query}\n\nFinal answer:"}],
            4096,
        )
