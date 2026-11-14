# Issue #117 feature 2: schema'd RAG eval set + offline metric harness.
#
# Tables: eval_cases (query / gold chunks, seeded deterministically from the
# recall_bench corpus) + eval_runs (one row per metric per run). The runner
# is BM25-only and 100% offline — no embedding model, no fusion-mlx, no
# network — so CI can gate on it. Metrics: recall@5 / recall@10 / mrr@10.
#
# The corpus is recall_bench.generate_corpus(seed=SEED) — deterministic, so
# run-to-run deltas are real code changes, not corpus drift.

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import contextmanager
from typing import Any

from .recall_bench import (
    SEED,
    _map_at_k,
    _ndcg_at_k,
    build_chunk_records,
    generate_corpus,
)
from .sqlite_base import SqliteBase

logger = logging.getLogger(__name__)

EVAL_K_VALUES = (5, 10)
# R12 (issue #134): ndcg@10 + map@10 added — position-aware metrics that
# credit every relevant hit, not just the first (mrr) or raw counts (recall).
EVAL_METRICS = ("recall@5", "recall@10", "mrr@10", "ndcg@10", "map@10")


class EvalStore(SqliteBase):
    def __init__(self, db_path: str):
        self.db_path = db_path
        super().__init__()
        self._create_table()
        logger.info("EvalStore initialized with db=%s", db_path)

    @contextmanager
    def _cursor(self):
        conn = self._get_conn()
        with self._db_lock:
            cur = conn.cursor()
            try:
                yield cur
                conn.commit()
            except Exception:
                conn.rollback()
                logger.exception("DB operation failed, rolled back")
                raise
            finally:
                cur.close()

    def _create_table(self):
        with self._cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS eval_cases (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    gold_doc_id TEXT NOT NULL,
                    gold_chunk_ids TEXT NOT NULL,
                    note TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS eval_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    metric TEXT NOT NULL,
                    value REAL NOT NULL,
                    case_count INTEGER NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_eval_cases_query ON eval_cases(query)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_eval_runs_run ON eval_runs(run_id)")

    def seed_cases(self, cases: list[dict[str, Any]]) -> int:
        """Insert seed cases; rows with an existing query are skipped (idempotent)."""
        now = time.time()
        inserted = 0
        with self._cursor() as cur:
            for c in cases:
                cur.execute(
                    "INSERT OR IGNORE INTO eval_cases(query, gold_doc_id, gold_chunk_ids, note, created_at) "
                    "VALUES(?,?,?,?,?)",
                    (c["query"], c["gold_doc_id"], json.dumps(c["gold_chunk_ids"]), c.get("note", ""), now),
                )
                inserted += cur.rowcount
        logger.info("eval seed: %d new cases inserted (%d total submitted)", inserted, len(cases))
        return inserted

    def list_cases(self) -> list[dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute("SELECT id, query, gold_doc_id, gold_chunk_ids, note FROM eval_cases ORDER BY id")
            rows = [dict(r) for r in cur.fetchall()]
        for r in rows:
            r["gold_chunk_ids"] = json.loads(r["gold_chunk_ids"])
        return rows

    def case_count(self) -> int:
        with self._cursor() as cur:
            cur.execute("SELECT COUNT(*) AS n FROM eval_cases")
            return int(cur.fetchone()["n"])

    def record_run(self, metrics: dict[str, float], case_count: int) -> str:
        run_id = uuid.uuid4().hex[:12]
        now = time.time()
        with self._cursor() as cur:
            for metric, value in metrics.items():
                cur.execute(
                    "INSERT INTO eval_runs(run_id, metric, value, case_count, created_at) VALUES(?,?,?,?,?)",
                    (run_id, metric, float(value), case_count, now),
                )
        logger.info("eval run %s recorded: %s (cases=%d)", run_id, metrics, case_count)
        return run_id

    def latest_run(self) -> dict[str, Any] | None:
        with self._cursor() as cur:
            cur.execute("SELECT run_id, created_at FROM eval_runs ORDER BY id DESC LIMIT 1")
            row = cur.fetchone()
            if row is None:
                return None
            cur.execute(
                "SELECT metric, value, case_count FROM eval_runs WHERE run_id=?",
                (row["run_id"],),
            )
            rows = cur.fetchall()
            metrics = {r["metric"]: r["value"] for r in rows}
            case_count = rows[0]["case_count"] if rows else 0
            return {
                "run_id": row["run_id"],
                "created_at": row["created_at"],
                "metrics": metrics,
                "case_count": case_count,
            }


async def run_offline_eval(workdir: str) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """BM25-only offline eval over the deterministic seed corpus.

    Returns (aggregate_metrics, per_case_rows). Fully offline: BM25 index over
    real-chunker chunks, no embeddings, no LLM, no network."""
    from .bm25_index import BM25Index

    docs, queries = generate_corpus(seed=SEED)
    records, doc_to_chunks = await build_chunk_records(docs)
    bm25 = BM25Index(store_path=f"{workdir}/eval_bm25.db")
    bm25.add_documents(records)

    per_case: list[dict[str, Any]] = []
    sums = dict.fromkeys(EVAL_METRICS, 0.0)
    for query_text, gold_doc_id in queries:
        truth = set(doc_to_chunks.get(gold_doc_id, []))
        if not truth:
            continue
        retrieved = [r["id"] for r in bm25.search(query_text, top_k=10)]
        rec5 = sum(1 for rid in retrieved[:5] if rid in truth) / len(truth)
        rec10 = sum(1 for rid in retrieved[:10] if rid in truth) / len(truth)
        mrr10 = 0.0
        for rank, rid in enumerate(retrieved[:10], start=1):
            if rid in truth:
                mrr10 = 1.0 / rank
                break
        sums["recall@5"] += rec5
        sums["recall@10"] += rec10
        sums["mrr@10"] += mrr10
        sums["ndcg@10"] += _ndcg_at_k(retrieved, truth, 10)
        sums["map@10"] += _map_at_k(retrieved, truth, 10)
        per_case.append(
            {
                "query": query_text,
                "gold_doc_id": gold_doc_id,
                "recall@5": round(rec5, 4),
                "recall@10": round(rec10, 4),
                "mrr@10": round(mrr10, 4),
                "ndcg@10": round(_ndcg_at_k(retrieved, truth, 10), 4),
                "map@10": round(_map_at_k(retrieved, truth, 10), 4),
            }
        )
    n = len(per_case)
    metrics = {m: round(sums[m] / n, 4) if n else 0.0 for m in EVAL_METRICS}
    logger.info("offline eval done: %d cases, metrics=%s", n, metrics)
    return metrics, per_case


async def seed_default_cases(store: EvalStore) -> int:
    """Seed eval cases from the deterministic recall_bench corpus (idempotent)."""
    docs, queries = generate_corpus(seed=SEED)
    _, doc_to_chunks = await build_chunk_records(docs)
    cases = [
        {
            "query": q,
            "gold_doc_id": doc_id,
            "gold_chunk_ids": doc_to_chunks.get(doc_id, []),
            "note": "seed: recall_bench deterministic corpus",
        }
        for q, doc_id in queries
    ]
    return store.seed_cases(cases)
