"""Streaming SSE support and metadata extraction for RAG responses."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class SSEStreamer:
    """Server-Sent Events streaming for RAG responses."""

    @staticmethod
    async def stream_response(question: str, context: str, mlx_url: str = "http://localhost:11434/v1") -> str:
        """Stream a RAG response as SSE events."""
        import httpx

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

    def __init__(self, mlx_url: str = "http://localhost:11434/v1"):
        self.mlx_url = mlx_url

    async def extract(self, text: str, doc_name: str = "") -> dict[str, Any]:
        """Extract metadata from document text."""
        import httpx

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
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{self.mlx_url}/chat/completions",
                    json={
                        "model": "qwen3.5-9b",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 512,
                        "temperature": 0.1,
                    },
                )
                content = resp.json()["choices"][0]["message"]["content"]
                # Extract JSON from response
                import re

                match = re.search(r"\{.*\}", content, re.DOTALL)
                if match:
                    return json.loads(match.group())
        except Exception as e:
            logger.debug("Metadata extraction failed: %s", e)
        return {"title": doc_name, "language": "unknown", "topics": []}


class ResultCache:
    """SQLite-backed cache for RAG results."""

    def __init__(self, db_path: str = ""):
        from pathlib import Path

        if not db_path:
            db_path = str(Path.home() / ".fusion-rag" / "cache.db")
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self):
        import sqlite3

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS rag_cache (
                query_hash TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                context_hash TEXT NOT NULL,
                answer TEXT NOT NULL,
                sources TEXT DEFAULT '',
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_cache_lookup ON rag_cache(query_hash, context_hash);
        """)
        conn.commit()
        conn.close()

    def get(self, query: str, context: str = "") -> dict | None:
        import hashlib
        import json

        qh = hashlib.md5(query.encode()).hexdigest()
        ch = hashlib.md5(context.encode()).hexdigest() if context else ""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT answer, sources FROM rag_cache WHERE query_hash = ? AND context_hash = ?",
            (qh, ch),
        ).fetchone()
        conn.close()
        if row:
            return {"answer": row["answer"], "sources": json.loads(row["sources"])}
        return None

    def set(self, query: str, answer: str, context: str = "", sources: list | None = None):
        import hashlib
        import json
        import time

        qh = hashlib.md5(query.encode()).hexdigest()
        ch = hashlib.md5(context.encode()).hexdigest() if context else ""
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO rag_cache (query_hash, query, context_hash, answer, sources, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (qh, query, ch, answer, json.dumps(sources or []), time.time()),
        )
        conn.commit()
        conn.close()
