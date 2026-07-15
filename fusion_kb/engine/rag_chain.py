"""Multi-turn RAG — conversational retrieval with history support."""
from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class MultiTurnRAG:
    """Multi-turn conversational RAG with history tracking."""

    def __init__(self, mlx_url: str = "http://localhost:11434/v1", model: str = "qwen3.5-9b"):
        self.mlx_url = mlx_url.rstrip("/")
        self.model = model
        self._history: list[dict] = []

    async def ask(self, question: str, context: str = "",
                  session_id: str = "") -> dict[str, Any]:
        """Ask a question with conversation history."""
        messages = []
        if context:
            messages.append({"role": "system", "content": (
                "You are a knowledge base assistant. Answer based on the provided context. "
                "Cite sources. If the context doesn't contain the answer, say so."
            )})
        # Add history
        for h in self._history:
            messages.append(h)
        # Add context
        if context:
            messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"})
        else:
            messages.append({"role": "user", "content": question})

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
        except Exception as e:
            answer = f"Error: {e}"

        # Update history
        self._history.append({"role": "user", "content": question})
        self._history.append({"role": "assistant", "content": answer})
        if len(self._history) > 20:
            self._history = self._history[-20:]

        return {"answer": answer, "history_length": len(self._history) // 2}

    def clear_history(self) -> None:
        self._history.clear()


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