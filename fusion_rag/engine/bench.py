import json
import logging
import sqlite3
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class BenchRunner:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None
        self._ensure_table()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    @contextmanager
    def _cursor(self):
        conn = self._get_conn()
        cur = conn.cursor()
        try:
            yield cur
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()

    def _ensure_table(self):
        with self._cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS bench_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kb_id TEXT NOT NULL,
                    test_name TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    details TEXT DEFAULT '{}',
                    created_at REAL NOT NULL
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_bench_kb
                ON bench_results(kb_id, test_name)
            """)

    async def run_search_bench(
        self,
        kb_id: str,
        vec_store,
        embed_client,
        queries: list[dict],
    ) -> dict:
        logger.info("Running search bench for kb %s with %d queries", kb_id, len(queries))
        results = []
        total_bm25_ms = 0.0
        total_vector_ms = 0.0
        total_hybrid_ms = 0.0

        for q in queries:
            query_text = q.get("query", "")
            expected_doc = q.get("expected_doc", "")
            # L18: cap top_k so a bench query can't trigger an unbounded vector
            # scan. 100 is well above any sane retrieval depth and keeps bench
            # runs bounded.
            top_k = q.get("top_k", 10)
            if not isinstance(top_k, int) or top_k <= 0:
                logger.warning("bench query has invalid top_k=%r, defaulting to 10", top_k)
                top_k = 10
            top_k = min(top_k, 100)

            t0 = time.monotonic()
            try:
                bm25_results = vec_store.keyword_search(query_text, top_k=top_k)
            except Exception as e:
                logger.error("BM25 search failed: %s", e)
                bm25_results = []
            bm25_ms = (time.monotonic() - t0) * 1000
            total_bm25_ms += bm25_ms

            t0 = time.monotonic()
            try:
                query_vector = await embed_client.embed(query_text)
                if query_vector and not all(v == 0.0 for v in query_vector):
                    vector_results = vec_store.search(query_vector, top_k=top_k)
                else:
                    vector_results = []
            except Exception as e:
                logger.error("Vector search failed: %s", e)
                vector_results = []
            vector_ms = (time.monotonic() - t0) * 1000
            total_vector_ms += vector_ms

            t0 = time.monotonic()
            try:
                from .reranker import HybridSearch

                hs = HybridSearch(vec_store)
                hybrid_results = await hs.search(
                    query_vector,
                    query_text,
                    top_k=top_k,
                )
            except Exception:
                hybrid_results = []
            hybrid_ms = (time.monotonic() - t0) * 1000
            total_hybrid_ms += hybrid_ms

            bm25_hit = any(r.get("doc_name", "") == expected_doc for r in bm25_results) if expected_doc else False
            vec_hit = any(r.get("doc_name", "") == expected_doc for r in vector_results) if expected_doc else False
            hybrid_hit = any(r.get("doc_name", "") == expected_doc for r in hybrid_results) if expected_doc else False

            results.append(
                {
                    "query": query_text,
                    "bm25_ms": round(bm25_ms, 2),
                    "vector_ms": round(vector_ms, 2),
                    "hybrid_ms": round(hybrid_ms, 2),
                    "bm25_hit": bm25_hit,
                    "vector_hit": vec_hit,
                    "hybrid_hit": hybrid_hit,
                }
            )

        n = max(len(queries), 1)
        summary = {
            "kb_id": kb_id,
            "query_count": len(queries),
            "avg_bm25_ms": round(total_bm25_ms / n, 2),
            "avg_vector_ms": round(total_vector_ms / n, 2),
            "avg_hybrid_ms": round(total_hybrid_ms / n, 2),
            "bm25_latency_target": "<100ms",
            "bm25_target_met": (total_bm25_ms / n) < 100.0,
            "queries": results,
        }

        self._save_result(kb_id, "search_bench", "avg_bm25_ms", total_bm25_ms / n, summary)
        self._save_result(kb_id, "search_bench", "avg_vector_ms", total_vector_ms / n, summary)
        self._save_result(kb_id, "search_bench", "avg_hybrid_ms", total_hybrid_ms / n, summary)

        logger.info(
            "Search bench complete: avg_bm25=%.1fms avg_vector=%.1fms avg_hybrid=%.1fms",
            total_bm25_ms / n,
            total_vector_ms / n,
            total_hybrid_ms / n,
        )
        return summary

    def _save_result(self, kb_id: str, test_name: str, metric_name: str, metric_value: float, details: dict):
        now = time.time()
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO bench_results (kb_id, test_name, metric_name, metric_value, details, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (kb_id, test_name, metric_name, metric_value, json.dumps(details, ensure_ascii=False), now),
            )

    def list_results(self, kb_id: str, test_name: str | None = None) -> list[dict]:
        with self._cursor() as cur:
            if test_name:
                cur.execute(
                    "SELECT * FROM bench_results WHERE kb_id = ? AND test_name = ? ORDER BY created_at DESC",
                    (kb_id, test_name),
                )
            else:
                cur.execute(
                    "SELECT * FROM bench_results WHERE kb_id = ? ORDER BY created_at DESC",
                    (kb_id,),
                )
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    def delete_results(self, kb_id: str) -> int:
        with self._cursor() as cur:
            cur.execute("DELETE FROM bench_results WHERE kb_id = ?", (kb_id,))
            return cur.rowcount
