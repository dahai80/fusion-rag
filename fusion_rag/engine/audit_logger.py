import csv
import io
import json
import logging
import time
from contextlib import contextmanager

from .sqlite_base import SqliteBase

logger = logging.getLogger(__name__)

# P0-12: chars that spreadsheet apps treat as a formula trigger. A cell whose
# first char is one of these executes as a formula on open (Excel/Numbers/
# LibreOffice) — `=HYPERLINK(...)`, `=cmd|'/c calc'!A1`, `+...`, `-...`, `@...`.
# CSV is not a trusted format here: query/caller come from user input. Prefix
# a single quote to neutralize; spreadsheet apps display the value without the
# leading quote (it is the standard mitigation, e.g. OWASP CSV injection).
_CSV_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def _csv_safe(value: object) -> object:
    """Prefix a leading single quote on string cells that would be read as a
    formula. Non-strings and safe strings pass through unchanged."""
    if not isinstance(value, str) or not value:
        return value
    if value[0] in _CSV_FORMULA_PREFIXES:
        return "'" + value
    return value

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
);
CREATE INDEX IF NOT EXISTS idx_audit_kb_created ON audit_log(kb_id, created_at);
"""


class AuditLogger(SqliteBase):
    def __init__(self, db_path: str, retention_days: int = 30):
        # P2-7: inherit SqliteBase for the shared locked connection. The prior
        # self._conn was opened with check_same_thread=False but the _cursor
        # context manager took NO lock — under the async threadpool two threads
        # sharing self._conn interleaved commit/rollback (one thread's exception
        # rolled back the other's in-flight audit insert). SqliteBase exists
        # precisely for this; adopt it.
        self.db_path = db_path
        # P3-5: retention window in days. 0 = keep forever (legacy behavior).
        # Prune on construction so a long-running server doesn't grow audit.db
        # without bound; the (kb_id, created_at) index makes the delete cheap.
        self.retention_seconds = retention_days * 86400 if retention_days > 0 else 0
        super().__init__()
        conn = self._get_conn()
        conn.execute(_CREATE_TABLE_SQL)
        conn.commit()
        if self.retention_seconds:
            self.prune()
        logger.info("AuditLogger initialized with db_path=%s retention_days=%d", db_path, retention_days)

    @contextmanager
    def _cursor(self):
        conn = self._get_conn()
        with self._db_lock:
            cursor = conn.cursor()
            try:
                yield cursor
                conn.commit()
            except Exception:
                conn.rollback()
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

    def export_logs(self, kb_id: str, fmt: str = "json") -> str:
        # M7: param renamed from `format` to `fmt` to avoid shadowing the
        # builtin. Behavior unchanged.
        rows = self.query_logs(kb_id, limit=100000, offset=0)
        if fmt == "csv":
            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow(["id", "kb_id", "query", "caller", "results_count", "latency_ms", "created_at"])
            for row in rows:
                # P0-12: neutralize formula-injection on user-controlled string
                # cells (query, caller come straight from search input). id /
                # counts / latency / created_at are numeric — safe.
                writer.writerow(
                    [
                        row["id"],
                        _csv_safe(row["kb_id"]),
                        _csv_safe(row["query"]),
                        _csv_safe(row["caller"]),
                        row["results_count"],
                        row["latency_ms"],
                        row["created_at"],
                    ]
                )
            logger.info("export_logs: kb_id=%s fmt=csv rows=%d", kb_id, len(rows))
            return buf.getvalue()
        logger.info("export_logs: kb_id=%s fmt=json rows=%d", kb_id, len(rows))
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

    def prune(self) -> int:
        # P3-5: auto-prune all KBs older than the retention window. Called on
        # construction; can also be called periodically by the caller. Without
        # this audit.db grows without bound on a long-running server (one row
        # per search). The (kb_id, created_at) index makes the range delete
        # index-backed, not a full scan.
        if not self.retention_seconds:
            return 0
        cutoff = time.time() - self.retention_seconds
        with self._cursor() as cur:
            cur.execute("DELETE FROM audit_log WHERE created_at < ?", (cutoff,))
            deleted = cur.rowcount
        if deleted:
            logger.info("audit prune: deleted %d rows older than %d days", deleted, self.retention_seconds // 86400)
        return deleted

    def iter_logs(self, kb_id: str, batch_size: int = 5000):
        # P3-5: streaming export. export_logs fetches up to 100k rows into one
        # in-memory string — a 100k-row KB builds a multi-MB buffer per export.
        # Yield rows in batches so a caller can stream to a response/file
        # without holding the whole result set in memory. Uses keyset pagination
        # (id > last_id) — OFFSET deep-paging gets slower the deeper it goes.
        # kb_id is bind-param (not interpolated) so no identifier validation
        # needed — matches query_logs, which also binds kb_id unvalidated.
        last_id = 0
        while True:
            with self._read_cursor() as conn:
                rows = conn.execute(
                    "SELECT * FROM audit_log WHERE kb_id = ? AND id > ? ORDER BY id LIMIT ?",
                    (kb_id, last_id, batch_size),
                ).fetchall()
            if not rows:
                return
            for row in rows:
                entry = dict(row)
                entry["top_sources"] = json.loads(entry["top_sources"])
                entry["metadata"] = json.loads(entry["metadata"])
                last_id = entry["id"]
                yield entry
