"""Issue #132 (R10): BGE-M3 native sparse (lexical) weight store.

fusion-mlx PR #1627 exposes BGE-M3's sparse channel via POST /v1/embeddings
with "return_sparse": true — per input, {"token_id": weight} (relu + max-pool,
per FlagEmbedding encode_sparse). The sparse head loads lazily upstream; models
without the head return 400 or omit the field (caller falls back to BM25).

Storage: per-KB SQLite {storage_path}/sparse.db — chunk_id -> {token_id:
weight} JSON, written at ingest, read at query time for the third RRF fusion
channel in HybridSearch. Batched lookups by chunk_id (same #94 exact-chunk-id
discipline as vectors/BM25 teardown). SqliteBase gives WAL + the shared
thread-safe connection policy (defect-6).
"""

from __future__ import annotations

import json
import logging
import threading

from .sqlite_base import SqliteBase

logger = logging.getLogger(__name__)

_pool: dict[str, SparseIndex] = {}
_pool_lock = threading.Lock()


def get_sparse_index(db_path: str) -> SparseIndex:
    # Pooled like get_or_create_vec_store: one shared connection per KB file
    # instead of a new SQLite conn per request.
    with _pool_lock:
        idx = _pool.get(db_path)
        if idx is None:
            idx = SparseIndex(db_path)
            _pool[db_path] = idx
        return idx


def sparse_dot(query_weights: dict[str, float], doc_weights: dict[str, float]) -> float:
    """Dot product over the shared token-id space (same model at ingest + query)."""
    if not query_weights or not doc_weights:
        return 0.0
    total = 0.0
    for token, q_weight in query_weights.items():
        d_weight = doc_weights.get(token)
        if d_weight:
            total += q_weight * d_weight
    return total


class SparseIndex(SqliteBase):
    """chunk_id -> sparse lexical weights, per-KB SQLite (WAL via SqliteBase)."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        super().__init__()
        self._init_db()

    def _init_db(self) -> None:
        conn = self._get_conn()
        with self._db_lock:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS sparse_weights (
                       chunk_id TEXT PRIMARY KEY,
                       doc_id TEXT NOT NULL,
                       weights TEXT NOT NULL
                   )"""
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sparse_doc ON sparse_weights(doc_id)")
            conn.commit()

    def add_batch(self, rows: list[tuple[str, str, dict[str, float]]]) -> None:
        # rows: (chunk_id, doc_id, weights). Empty weights are skipped — a
        # zero-sparse row carries no signal and would only bloat lookups.
        payload = [
            (chunk_id, doc_id, json.dumps(weights, ensure_ascii=False))
            for chunk_id, doc_id, weights in rows
            if weights
        ]
        if not payload:
            return
        conn = self._get_conn()
        with self._db_lock:
            conn.executemany(
                "INSERT OR REPLACE INTO sparse_weights (chunk_id, doc_id, weights) VALUES (?, ?, ?)",
                payload,
            )
            conn.commit()

    def get_map(self, chunk_ids: list[str]) -> dict[str, dict[str, float]]:
        # Batched chunk_id -> weights lookup for the query-time candidate pool.
        ids = [c for c in dict.fromkeys(chunk_ids) if c]
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        conn = self._get_conn()
        rows = conn.execute(
            f"SELECT chunk_id, weights FROM sparse_weights WHERE chunk_id IN ({placeholders})",
            ids,
        ).fetchall()
        out: dict[str, dict[str, float]] = {}
        for r in rows:
            try:
                out[r["chunk_id"]] = json.loads(r["weights"])
            except Exception as e:
                logger.warning("sparse_weights row for %s unreadable: %s", r["chunk_id"], e)
        return out

    def delete_by_chunk_ids(self, chunk_ids: list[str]) -> int:
        ids = [c for c in dict.fromkeys(chunk_ids) if c]
        if not ids:
            return 0
        placeholders = ",".join("?" for _ in ids)
        conn = self._get_conn()
        with self._db_lock:
            cur = conn.execute(
                f"DELETE FROM sparse_weights WHERE chunk_id IN ({placeholders})", ids
            )
            conn.commit()
            return cur.rowcount

    def delete_by_doc(self, doc_id: str) -> int:
        conn = self._get_conn()
        with self._db_lock:
            cur = conn.execute("DELETE FROM sparse_weights WHERE doc_id = ?", (doc_id,))
            conn.commit()
            return cur.rowcount

    def count(self) -> int:
        conn = self._get_conn()
        row = conn.execute("SELECT COUNT(*) AS cnt FROM sparse_weights").fetchone()
        return row["cnt"] if row else 0
