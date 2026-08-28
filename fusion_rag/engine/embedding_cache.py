"""EmbeddingCache — SQLite-backed cache for embedding vectors.

callers: EmbeddingClient.embed/embed_batch via integration
API: EmbeddingCache.get(text) -> list[float] | None, .set(text, vector), .clear()
schema: cache table (text_hash TEXT PK, text TEXT, vector BLOB, model TEXT, created_at REAL)
user instruction: "按照你的方案和计划落地所有phase阶段的需求"
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

from .sqlite_base import SqliteBase

logger = logging.getLogger(__name__)


class EmbeddingCache(SqliteBase):
    """SQLite-backed cache for embedding vectors to avoid redundant API calls."""

    def __init__(self, db_path: str = "", ttl: int = 86400 * 7, max_entries: int = 100000):
        if not db_path:
            db_path = str(Path.home() / ".fusion-rag" / "embed_cache.db")
        self.db_path = db_path
        self.ttl = ttl
        self.max_entries = max_entries
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        super().__init__()
        self._init_db()

    def _init_db(self) -> None:
        with self._db_lock:
            conn = self._get_conn()
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS embed_cache (
                    text_hash TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    vector TEXT NOT NULL,
                    model TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_embed_cache_hash ON embed_cache(text_hash);
            """)
            conn.commit()

    def _hash(self, text: str, model: str = "") -> str:
        key = f"{model}:{text}"
        return hashlib.md5(key.encode()).hexdigest()

    def get(self, text: str, model: str = "") -> list[float] | None:
        h = self._hash(text, model)
        with self._db_lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    "SELECT vector, created_at FROM embed_cache WHERE text_hash = ?",
                    (h,),
                ).fetchone()
                if row is None:
                    return None
                if self.ttl and (time.time() - row["created_at"]) > self.ttl:
                    conn.execute("DELETE FROM embed_cache WHERE text_hash = ?", (h,))
                    conn.commit()
                    return None
                return json.loads(row["vector"])
            except Exception as e:
                logger.warning("EmbeddingCache get failed: %s", e)
                return None

    @staticmethod
    def _is_zero_vector(vector: list[float]) -> bool:
        return len(vector) > 0 and all(v == 0.0 for v in vector)

    def set(self, text: str, vector: list[float], model: str = "") -> None:
        if self._is_zero_vector(vector):
            logger.warning("skip caching zero vector for text=%s model=%s", text[:60], model)
            return
        h = self._hash(text, model)
        with self._db_lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """INSERT OR REPLACE INTO embed_cache (text_hash, text, vector, model, created_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (h, text[:1000], json.dumps(vector), model, time.time()),
                )
                conn.commit()
                self._evict_if_needed(conn)
            except Exception as e:
                logger.warning("EmbeddingCache set failed: %s", e)

    def get_batch(self, texts: list[str], model: str = "") -> list[list[float] | None]:
        # P2-5: prior impl called self.get per text -> N connection open/close
        # cycles (each open_sqlite re-ran WAL pragma = a forced checkpoint).
        # One batched SELECT WHERE text_hash IN (...) reuses the shared conn.
        if not texts:
            return []
        hashes = [self._hash(t, model) for t in texts]
        placeholders = ",".join("?" for _ in hashes)
        found: dict[str, list[float]] = {}
        with self._db_lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    f"SELECT text_hash, vector, created_at FROM embed_cache WHERE text_hash IN ({placeholders})",
                    tuple(hashes),
                ).fetchall()
            except Exception as e:
                logger.warning("EmbeddingCache get_batch failed: %s", e)
                return [None for _ in texts]
        now = time.time()
        for row in rows:
            if self.ttl and (now - row["created_at"]) > self.ttl:
                continue
            try:
                found[row["text_hash"]] = json.loads(row["vector"])
            except Exception as e:
                logger.warning("EmbeddingCache get_batch parse failed: %s", e)
        return [found.get(h) for h in hashes]

    def set_batch(self, texts: list[str], vectors: list[list[float]], model: str = "") -> None:
        h = self._hash
        now = time.time()
        rows = []
        for t, v in zip(texts, vectors):
            if self._is_zero_vector(v):
                logger.warning("skip caching zero vector for text=%s model=%s", t[:60], model)
                continue
            rows.append((h(t, model), t[:1000], json.dumps(v), model, now))
        if not rows:
            return
        with self._db_lock:
            conn = self._get_conn()
            try:
                conn.executemany(
                    """INSERT OR REPLACE INTO embed_cache (text_hash, text, vector, model, created_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    rows,
                )
                conn.commit()
                self._evict_if_needed(conn)
            except Exception as e:
                logger.warning("EmbeddingCache set_batch failed: %s", e)

    def clear(self) -> None:
        with self._db_lock:
            conn = self._get_conn()
            conn.execute("DELETE FROM embed_cache")
            conn.commit()

    def count(self) -> int:
        with self._db_lock:
            conn = self._get_conn()
            row = conn.execute("SELECT COUNT(*) as cnt FROM embed_cache").fetchone()
            return row["cnt"] if row else 0

    def _evict_if_needed(self, conn: Any) -> None:
        row = conn.execute("SELECT COUNT(*) as cnt FROM embed_cache").fetchone()
        if row and row["cnt"] > self.max_entries:
            cutoff = conn.execute(
                "SELECT created_at FROM embed_cache ORDER BY created_at ASC LIMIT 1 OFFSET ?",
                (self.max_entries // 2,),
            ).fetchone()
            if cutoff:
                conn.execute(
                    "DELETE FROM embed_cache WHERE created_at < ?",
                    (cutoff["created_at"],),
                )
                conn.commit()
