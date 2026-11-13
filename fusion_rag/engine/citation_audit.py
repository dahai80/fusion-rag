# Issue #117 feature 1: first-class citation audit trail.
#
# Every retrieval call (when FUSION_RAG_CITATION_AUDIT=1) writes:
#   - one rag_queries row per query version: the original (depth 0, parent
#     NULL) plus one row per rewrite step chained via parent_id
#   - rag_citations rows for the retrieved chunks, linked to the FINAL query
#     version (last rewrite if any, else the original) with per-chunk
#     similarity score and rank
#
# Contract: an answer's citations can be reconstructed from the DB alone —
# rag_queries gives the query chain, rag_citations gives the evidence set
# that fed the generation. JSONL trajectory (trajectory_writer) stays as-is;
# this is the queryable relational tier.

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any

from .sqlite_base import SqliteBase

logger = logging.getLogger(__name__)


class CitationAuditStore(SqliteBase):
    def __init__(self, db_path: str):
        self.db_path = db_path
        super().__init__()
        self._create_table()
        logger.info("CitationAuditStore initialized with db=%s", db_path)

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
                CREATE TABLE IF NOT EXISTS rag_queries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kb_id TEXT NOT NULL,
                    query TEXT NOT NULL,
                    parent_id INTEGER,
                    depth INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS rag_citations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_id INTEGER NOT NULL,
                    chunk_id TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    similarity REAL NOT NULL,
                    rank INTEGER NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_rag_queries_kb ON rag_queries(kb_id, id)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_rag_citations_query ON rag_citations(query_id)"
            )

    def record_retrieval(
        self,
        kb_id: str,
        original_query: str,
        rewrites: list[str],
        results: list[dict[str, Any]],
    ) -> int:
        """Write one retrieval audit: query chain + citations of the final version.

        rewrites: ordered rewrite steps (empty list = no rewrite). Citations
        link to the LAST rewrite's row id, or the original row id when empty.
        Returns the root query row id.
        """
        now = time.time()
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO rag_queries(kb_id, query, parent_id, depth, created_at) VALUES(?,?,?,?,?)",
                (kb_id, original_query, None, 0, now),
            )
            root_id = int(cur.lastrowid)
            final_id = root_id
            for depth, rq in enumerate(rewrites, start=1):
                cur.execute(
                    "INSERT INTO rag_queries(kb_id, query, parent_id, depth, created_at) VALUES(?,?,?,?,?)",
                    (kb_id, rq, final_id, depth, now),
                )
                final_id = int(cur.lastrowid)
            rows = []
            for rank, r in enumerate(results, start=1):
                chunk_id = str(r.get("chunk_id") or r.get("id") or "")
                doc_id = str(r.get("doc_id") or "")
                if not chunk_id:
                    continue
                try:
                    score = float(r.get("score") or 0.0)
                except (TypeError, ValueError):
                    score = 0.0
                rows.append((final_id, chunk_id, doc_id, score, rank, now))
            if rows:
                cur.executemany(
                    "INSERT INTO rag_citations(query_id, chunk_id, doc_id, similarity, rank, "
                    "created_at) VALUES(?,?,?,?,?,?)",
                    rows,
                )
        logger.info(
            "citation audit: kb=%s root_query_id=%s rewrites=%d citations=%d",
            kb_id,
            root_id,
            len(rewrites),
            len(rows),
        )
        return root_id

    def get_trace(self, kb_id: str, query_id: int) -> dict[str, Any] | None:
        """Reconstruct one retrieval trace from the DB alone."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT id, kb_id, query, parent_id, depth, created_at FROM rag_queries WHERE id=? AND kb_id=?",
                (query_id, kb_id),
            )
            row = cur.fetchone()
            if row is None:
                return None
            root = dict(row)
            # Walk up to the retrieval root (depth 0), then follow the rewrite
            # chain down (each rewrite's parent is the previous version).
            root_id = root["id"]
            parent_id = root["parent_id"]
            while parent_id is not None:
                cur.execute(
                    "SELECT id, parent_id FROM rag_queries WHERE id=? AND kb_id=?",
                    (parent_id, kb_id),
                )
                prow = cur.fetchone()
                if prow is None:
                    break
                root_id = prow["id"]
                parent_id = prow["parent_id"]
            cur.execute(
                "SELECT id, query, parent_id, depth, created_at FROM rag_queries WHERE id=? AND kb_id=?",
                (root_id, kb_id),
            )
            chain = [dict(cur.fetchone())]
            cur_id = root_id
            while True:
                cur.execute(
                    "SELECT id, query, parent_id, depth, created_at FROM rag_queries "
                    "WHERE parent_id=? AND kb_id=? ORDER BY id",
                    (cur_id, kb_id),
                )
                child = cur.fetchone()
                if child is None:
                    break
                chain.append(dict(child))
                cur_id = child["id"]
            final_id = chain[-1]["id"]
            cur.execute(
                "SELECT chunk_id, doc_id, similarity, rank, created_at FROM rag_citations "
                "WHERE query_id=? ORDER BY rank",
                (final_id,),
            )
            citations = [dict(r) for r in cur.fetchall()]
        return {
            "query_id": root["id"],
            "kb_id": kb_id,
            "query_chain": chain,
            "final_query_id": final_id,
            "citations": citations,
        }

    def list_traces(self, kb_id: str, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        """List retrieval roots (depth 0) for a KB, newest first, with citation counts."""
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT q.id, q.query, q.created_at,
                       (SELECT COUNT(*) FROM rag_citations c
                         WHERE c.query_id = COALESCE(
                           (SELECT MAX(q2.id) FROM rag_queries q2 WHERE q2.id = q.id OR q2.parent_id = q.id), q.id))
                       AS citation_count,
                       (SELECT COUNT(*) FROM rag_queries q3 WHERE q3.parent_id = q.id) AS rewrite_count
                FROM rag_queries q
                WHERE q.kb_id=? AND q.depth=0
                ORDER BY q.id DESC LIMIT ? OFFSET ?
                """,
                (kb_id, int(limit), int(offset)),
            )
            return [dict(r) for r in cur.fetchall()]
