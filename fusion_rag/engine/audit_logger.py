import csv
import io
import json
import logging
import sqlite3
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kb_id TEXT NOT NULL,
    query TEXT NOT NULL,
    caller TEXT NOT NULL,
    results_count INTEGER DEFAULT 0,
    top_sources TEXT DEFAULT '[]',
    latency_ms REAL DEFAULT 0.0,
    metadata TEXT DEFAULT '{}',
    created_at REAL NOT NULL
)
"""


class AuditLogger:
    def __init__(self, db_path: str):
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_CREATE_TABLE_SQL)
        self._conn.commit()
        logger.info("AuditLogger initialized with db_path=%s", db_path)

    @contextmanager
    def _get_conn(self):
        yield self._conn

    @contextmanager
    def _cursor(self):
        cursor = self._conn.cursor()
        try:
            yield cursor
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cursor.close()

    def log_search(
        self,
        kb_id: str,
        query: str,
        caller: str,
        results_count: int,
        top_sources: list[dict],
        latency_ms: float,
        metadata: dict | None = None,
    ) -> int:
        now = time.time()
        sources_json = json.dumps(top_sources, ensure_ascii=False)
        meta_json = json.dumps(metadata or {}, ensure_ascii=False)
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO audit_log
                    (kb_id, query, caller, results_count, top_sources, latency_ms, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (kb_id, query, caller, results_count, sources_json, latency_ms, meta_json, now),
            )
            log_id = cur.lastrowid
        logger.info(
            "log_search: id=%d kb_id=%s caller=%s query=%.80s results=%d latency=%.1fms",
            log_id,
            kb_id,
            caller,
            query,
            results_count,
            latency_ms,
        )
        return log_id

    def query_logs(
        self,
        kb_id: str,
        limit: int = 50,
        offset: int = 0,
        caller: str | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> list[dict]:
        clauses = ["kb_id = ?"]
        params: list = [kb_id]
        if caller is not None:
            clauses.append("caller = ?")
            params.append(caller)
        if start_time is not None:
            clauses.append("created_at >= ?")
            params.append(start_time)
        if end_time is not None:
            clauses.append("created_at <= ?")
            params.append(end_time)
        where = " AND ".join(clauses)
        params.extend([limit, offset])
        sql = f"SELECT * FROM audit_log WHERE {where} ORDER BY created_at DESC LIMIT ? OFFSET ?"
        with self._cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
        results = []
        for row in rows:
            entry = dict(zip(columns, row))
            entry["top_sources"] = json.loads(entry["top_sources"])
            entry["metadata"] = json.loads(entry["metadata"])
            results.append(entry)
        logger.debug("query_logs: kb_id=%s returned %d rows", kb_id, len(results))
        return results

    def get_log(self, log_id: int) -> dict | None:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM audit_log WHERE id = ?", (log_id,))
            row = cur.fetchone()
        if row is None:
            return None
        columns = [desc[0] for desc in cur.description]
        entry = dict(zip(columns, row))
        entry["top_sources"] = json.loads(entry["top_sources"])
        entry["metadata"] = json.loads(entry["metadata"])
        return entry

    def count_logs(self, kb_id: str) -> int:
        with self._cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM audit_log WHERE kb_id = ?", (kb_id,))
            count = cur.fetchone()[0]
        return count

    def export_logs(self, kb_id: str, format: str = "json") -> str:
        rows = self.query_logs(kb_id, limit=100000, offset=0)
        if format == "csv":
            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow(["id", "kb_id", "query", "caller", "results_count", "latency_ms", "created_at"])
            for row in rows:
                writer.writerow(
                    [
                        row["id"],
                        row["kb_id"],
                        row["query"],
                        row["caller"],
                        row["results_count"],
                        row["latency_ms"],
                        row["created_at"],
                    ]
                )
            logger.info("export_logs: kb_id=%s format=csv rows=%d", kb_id, len(rows))
            return buf.getvalue()
        logger.info("export_logs: kb_id=%s format=json rows=%d", kb_id, len(rows))
        return json.dumps(rows, ensure_ascii=False, indent=2)

    def delete_old_logs(self, kb_id: str, before_time: float) -> int:
        with self._cursor() as cur:
            cur.execute(
                "DELETE FROM audit_log WHERE kb_id = ? AND created_at < ?",
                (kb_id, before_time),
            )
            deleted = cur.rowcount
        logger.info("delete_old_logs: kb_id=%s before=%.3f deleted=%d", kb_id, before_time, deleted)
        return deleted
