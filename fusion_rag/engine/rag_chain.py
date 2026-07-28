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

    def __init__(self, mlx_url: str = "http://localhost:11434/v1",
                 model: str = "qwen3.5-9b",
                 token_budget: int = 8192,
                 max_history_turns: int = 10):
        self.mlx_url = mlx_url.rstrip("/")
        self.model = model
        self.token_budget = token_budget
        self.max_history_turns = max_history_turns
        self._history: list[dict] = []
        self._sessions: dict[str, list[dict]] = {}

    async def ask(self, question: str, context: str = "",
                  session_id: str = "") -> dict[str, Any]:
        history = self._resolve_history(session_id)
        messages = self._build_messages(question, context, history)

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(f"{self.mlx_url}/chat/completions", json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": 4096,
                    "temperature": 0.3,
                })
                resp.raise_for_status()
                data = resp.json()
                answer = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
        except Exception as e:
            logger.error("MultiTurnRAG ask failed: %s", e)
            answer = f"Error: {e}"
            usage = {}

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        self._trim_history(history)

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

    def _build_messages(self, question: str, context: str,
                        history: list[dict]) -> list[dict]:
        messages = []
        if context:
            messages.append({"role": "system", "content": (
                "You are a knowledge base assistant. Answer based on the provided context. "
                "Cite sources. If the context doesn't contain the answer, say so."
            )})

        # Add history within token budget (most recent first)
        remaining = self.token_budget - estimate_tokens(context) - 500
        for h in reversed(history):
            h_tokens = estimate_tokens(h.get("content", ""))
            if remaining - h_tokens < 0:
                break
            messages.insert(len(messages) - 0 if not context else 1, h)
            remaining -= h_tokens

        if context:
            messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"})
        else:
            messages.append({"role": "user", "content": question})

        return messages

    def _trim_history(self, history: list[dict]) -> None:
        max_entries = self.max_history_turns * 2
        if len(history) > max_entries:
            del history[:len(history) - max_entries]

    @staticmethod
    def _history_token_count(history: list[dict]) -> int:
        return sum(estimate_tokens(h.get("content", "")) for h in history)


class DocumentChain:
    """Processes multiple documents with chain strategies."""

    @staticmethod
    async def stuff(docs: list[str], query: str, mlx_url: str = "http://localhost:11434/v1") -> str:
        """Stuff all docs into a single prompt."""
        context = "\n\n".join(docs)
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(f"{mlx_url}/chat/completions", json={
                "model": "qwen3.5-9b",
                "messages": [{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}],
                "max_tokens": 4096,
            })
            return resp.json()["choices"][0]["message"]["content"]

    @staticmethod
    async def refine(docs: list[str], query: str, mlx_url: str = "http://localhost:11434/v1") -> str:
        """Refine answer iteratively with each document."""
        answer = ""
        for doc in docs:
            async with httpx.AsyncClient(timeout=60.0) as client:
                prompt = f"Previous answer: {answer}\n\nNew document: {doc}\n\nQuestion: {query}\n\nRefine the answer."
                if not answer:
                    prompt = f"Document: {doc}\n\nQuestion: {query}\n\nAnswer:"
                resp = await client.post(f"{mlx_url}/chat/completions", json={
                    "model": "qwen3.5-9b",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 4096,
                })
                answer = resp.json()["choices"][0]["message"]["content"]
        return answer

    @staticmethod
    async def map_reduce(docs: list[str], query: str, mlx_url: str = "http://localhost:11434/v1") -> str:
        """Map each doc, then reduce to final answer."""
        import asyncio
        async def map_doc(doc: str) -> str:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(f"{mlx_url}/chat/completions", json={
                    "model": "qwen3.5-9b",
                    "messages": [{"role": "user", "content": f"Document: {doc}\n\nQuestion: {query}\n\nSummary:"}],
                    "max_tokens": 1024,
                })
                return resp.json()["choices"][0]["message"]["content"]

        summaries = await asyncio.gather(*[map_doc(d) for d in docs])
        combined = "\n\n".join(summaries)

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(f"{mlx_url}/chat/completions", json={
                "model": "qwen3.5-9b",
                "messages": [{"role": "user", "content": f"Summaries:\n{combined}\n\nQuestion: {query}\n\nFinal answer:"}],
                "max_tokens": 4096,
            })
            return resp.json()["choices"][0]["message"]["content"]