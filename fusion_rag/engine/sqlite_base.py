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

import logging
import sqlite3
import threading
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
        self._db_lock = threading.Lock()

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
