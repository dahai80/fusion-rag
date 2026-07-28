"""RAG Evaluator — automated evaluation framework for RAG pipeline quality.

callers: routes.py eval endpoint, CLI evaluation commands
API: RAGEvaluator.evaluate(kb_id, test_cases) -> EvaluationResult
schema: test_cases list[dict(query, expected_answer, expected_docs)],
        results with faithfulness/relevance/context_recall scores
user instruction: "按照你的方案和计划落地所有phase阶段的需求"
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

FAITHFULNESS_PROMPT = (
    "Given the following context and answer, determine if the answer is "
    "faithful to the context (i.e., all claims in the answer are supported by the context). "
    "Output a JSON object: {\"faithful\": true/false, \"reason\": \"...\"}\n\n"
    "Context: {context}\n\nAnswer: {answer}\n\nJSON:"
)

RELEVANCE_PROMPT = (
    "Rate the relevance of the following answer to the question on a scale of 0 to 10. "
    "Output ONLY a JSON object: {\"score\": <0-10>, \"reason\": \"...\"}\n\n"
    "Question: {query}\n\nAnswer: {answer}\n\nJSON:"
)


class RAGEvaluator:
    """Evaluates RAG pipeline quality with automated metrics."""

    def __init__(self, mlx_url: str = "http://localhost:11434/v1",
                 model: str = "qwen3.5-9b",
                 db_path: str = ""):
        self.mlx_url = mlx_url.rstrip("/")
        self.model = model
        if not db_path:
            db_path = str(Path.home() / ".fusion-rag" / "eval.db")
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS eval_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kb_id TEXT NOT NULL,
                query TEXT NOT NULL,
                expected_answer TEXT DEFAULT '',
                actual_answer TEXT DEFAULT '',
                faithfulness REAL DEFAULT 0,
                relevance REAL DEFAULT 0,
                context_recall REAL DEFAULT 0,
                retrieval_count INTEGER DEFAULT 0,
                latency_ms REAL DEFAULT 0,
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_eval_kb ON eval_results(kb_id);
        """)
        conn.commit()
        conn.close()

    async def evaluate(self, kb_id: str,
                       test_cases: list[dict[str, str]]) -> dict[str, Any]:
        """Run evaluation on test cases against a knowledge base."""
        from .routes import _get_kb_manager, _get_embed_client, _get_base
        from ..store.vector_store import VectorStore

        results = []
        for tc in test_cases:
            query = tc.get("query", "")
            expected = tc.get("expected_answer", "")
            expected_docs = tc.get("expected_docs", [])

            if not query:
                continue

            start = time.time()
            try:
                eval_result = await self._evaluate_single(
                    kb_id, query, expected, expected_docs)
                eval_result["latency_ms"] = (time.time() - start) * 1000
            except Exception as e:
                logger.warning("Evaluation failed for query '%s': %s", query[:50], e)
                eval_result = {
                    "query": query, "error": str(e),
                    "faithfulness": 0, "relevance": 0, "context_recall": 0,
                    "latency_ms": (time.time() - start) * 1000,
                }

            results.append(eval_result)
            self._store_result(kb_id, eval_result)

        if not results:
            return {"total": 0, "avg_faithfulness": 0, "avg_relevance": 0}

        avg_f = sum(r.get("faithfulness", 0) for r in results) / len(results)
        avg_r = sum(r.get("relevance", 0) for r in results) / len(results)
        avg_cr = sum(r.get("context_recall", 0) for r in results) / len(results)

        return {
            "total": len(results),
            "avg_faithfulness": round(avg_f, 3),
            "avg_relevance": round(avg_r, 3),
            "avg_context_recall": round(avg_cr, 3),
            "results": results,
        }

    async def _evaluate_single(self, kb_id: str, query: str,
                                expected: str,
                                expected_docs: list[str]) -> dict[str, Any]:
        """Evaluate a single query."""
        from .routes import _get_base, _get_embed_client, _generate_answer
        from ..store.vector_store import VectorStore

        kb = _get_base(kb_id)
        embed = _get_embed_client()
        vec_store = VectorStore(kb.vector_path)

        query_vector = await embed.embed(query)
        if not query_vector or all(v == 0.0 for v in query_vector):
            return {"query": query, "error": "Embedding failed",
                    "faithfulness": 0, "relevance": 0, "context_recall": 0}

        chunks = vec_store.search(query_vector, top_k=kb.config.max_results)
        context = "\n\n".join(f"[{c['doc_name']}] {c['text'][:2000]}" for c in chunks)
        answer_result = await _generate_answer(query, context, chunks)
        actual_answer = answer_result.get("answer", "")

        # Compute metrics
        faithfulness = await self._score_faithfulness(context, actual_answer)
        relevance = await self._score_relevance(query, actual_answer)
        context_recall = self._compute_context_recall(chunks, expected_docs)

        return {
            "query": query,
            "expected_answer": expected[:200],
            "actual_answer": actual_answer[:200],
            "retrieval_count": len(chunks),
            "faithfulness": faithfulness,
            "relevance": relevance,
            "context_recall": context_recall,
        }

    async def _score_faithfulness(self, context: str, answer: str) -> float:
        if not answer or not context:
            return 0.0
        prompt = FAITHFULNESS_PROMPT.format(context=context[:3000], answer=answer[:1000])
        try:
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
                match = re.search(r"\{.*\}", content, re.DOTALL)
                if match:
                    data = json.loads(match.group())
                    return 1.0 if data.get("faithful", False) else 0.0
        except Exception as e:
            logger.warning("Faithfulness scoring failed: %s", e)
        return 0.5

    async def _score_relevance(self, query: str, answer: str) -> float:
        if not answer:
            return 0.0
        prompt = RELEVANCE_PROMPT.format(query=query, answer=answer[:1000])
        try:
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
                match = re.search(r"\{.*\}", content, re.DOTALL)
                if match:
                    data = json.loads(match.group())
                    return min(data.get("score", 0) / 10.0, 1.0)
        except Exception as e:
            logger.warning("Relevance scoring failed: %s", e)
        return 0.5

    @staticmethod
    def _compute_context_recall(retrieved: list[dict],
                                 expected_docs: list[str]) -> float:
        if not expected_docs:
            return 1.0
        if not retrieved:
            return 0.0
        retrieved_names = {r.get("doc_name", "") for r in retrieved}
        hits = sum(1 for d in expected_docs if d in retrieved_names)
        return hits / len(expected_docs)

    def _store_result(self, kb_id: str, result: dict) -> None:
        conn = self._get_conn()
        try:
            conn.execute(
                """INSERT INTO eval_results
                   (kb_id, query, expected_answer, actual_answer,
                    faithfulness, relevance, context_recall,
                    retrieval_count, latency_ms, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (kb_id, result.get("query", ""),
                 result.get("expected_answer", ""), result.get("actual_answer", ""),
                 result.get("faithfulness", 0), result.get("relevance", 0),
                 result.get("context_recall", 0),
                 result.get("retrieval_count", 0), result.get("latency_ms", 0),
                 time.time()),
            )
            conn.commit()
        except Exception as e:
            logger.warning("Failed to store eval result: %s", e)
        finally:
            conn.close()

    def get_history(self, kb_id: str = "", limit: int = 50) -> list[dict]:
        conn = self._get_conn()
        try:
            if kb_id:
                rows = conn.execute(
                    "SELECT * FROM eval_results WHERE kb_id = ? ORDER BY created_at DESC LIMIT ?",
                    (kb_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM eval_results ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()
