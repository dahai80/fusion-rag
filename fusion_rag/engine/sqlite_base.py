"""硬伤6 unified SQLite concurrency base.

Before this, half the modules opened WAL + a fresh connection per call, the
other half kept one shared connection with no WAL and no
``check_same_thread=False``. Under the async server's threadpool, the latter
interleaved commit/rollback across threads -> ``database is locked`` and
transaction cross-contamination.

This base fixes the concurrency policy in one place: every shared connection
is opened with ``check_same_thread=False`` + WAL, and a ``threading.Lock``
serializes writes so two threads never commit/rollback the same connection
mid-statement. Per-call-connection modules do not need the Lock (each call
owns its own connection) but still get the WAL pragma via ``open_sqlite``.
"""

from __future__ import annotations

import contextlib
import logging
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)


def open_sqlite(db_path: str | Path, *, readonly: bool = False) -> sqlite3.Connection:
    """Open a SQLite connection with the unified concurrency policy.

    - check_same_thread=False: safe to use from the async server's threadpool.
    - WAL journal mode: readers don't block the writer, reducing lock waits.
    - row_factory=Row: uniform dict-like access across modules.
    The caller owns this connection and must close it (or hand it to SqliteBase).
    """
    path = str(db_path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    uri = f"file:{path}?mode=ro" if readonly else f"file:{path}?mode=rwc"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    if not readonly:
        conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    logger.debug("opened sqlite %s readonly=%s WAL", path, readonly)
    return conn


class SqliteBase:
    """Mixin providing a lazily-created, thread-safe shared SQLite connection.

    Subclass sets ``self.db_path`` then calls ``self._get_conn()``. The
    connection is created once and reused; ``self._db_lock`` serializes writes.
    Close via ``self._close_conn()`` (shutdown) — the connection is NOT closed
    per call.
    """

    db_path: str

    def __init__(self) -> None:
        self._db: sqlite3.Connection | None = None
        self._db_closed = True
        # RLock (reentrant): _get_conn takes this lock to serialize the lazy
        # connection CREATE, and is itself called from methods that already hold
        # the lock (EmbeddingCache locks-then-calls-_get_conn; its methods need
        # the conn mid-block for _evict_if_needed / conditional delete). A plain
        # Lock deadlocks on that re-entrant acquire. RLock permits same-thread
        # re-entry with negligible overhead vs the SQLite I/O it guards.
        self._db_lock = threading.RLock()

    def _get_conn(self) -> sqlite3.Connection:
        if self._db is None or self._db_closed:
            with self._db_lock:
                if self._db is None or self._db_closed:
                    self._db = open_sqlite(self.db_path)
                    self._db_closed = False
                    logger.debug("SqliteBase lazy conn for %s", self.db_path)
        return self._db

    def _close_conn(self) -> None:
        with self._db_lock:
            if self._db is not None and not self._db_closed:
                try:
                    self._db.close()
                except sqlite3.Error as e:
                    logger.warning("SqliteBase close failed for %s: %s", self.db_path, e)
            self._db = None
            self._db_closed = True

    def checkpoint(self) -> None:
        # O-P2-1: PRAGMA wal_checkpoint(TRUNCATE) — fold the -wal sidecar back
        # into the main .db and truncate the WAL file to zero. Run before a
        # stores-dir snapshot (tar/rsync) so the backup captures a consistent
        # .db without a stale -wal that a restore would replay or drop. Safe
        # under concurrent readers (WAL); a writer briefly blocks. Idempotent.
        conn = self._get_conn()
        with self._db_lock:
            try:
                row = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
                logger.debug("SqliteBase checkpoint %s -> %s", self.db_path, tuple(row) if row else None)
            except sqlite3.Error as e:
                logger.warning("SqliteBase checkpoint failed for %s: %s", self.db_path, e)
                raise

    @contextmanager
    def _read_cursor(self):
        # No-lock read path. SQLite WAL lets concurrent readers proceed
        # alongside a writer without blocking, and check_same_thread=False
        # permits cross-thread use. The write lock (_db_lock) is held only by
        # write transactions; a read only fetches — no commit/rollback — so
        # interleaving on the shared connection is safe here. Holding the lock
        # on reads would serialize every read behind every writer.
        conn = self._get_conn()
        try:
            yield conn
        except Exception:
            with contextlib.suppress(sqlite3.Error):
                conn.rollback()
            raise
