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
import sqlite3
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """SQLite-backed cache for embedding vectors to avoid redundant API calls."""

    def __init__(self, db_path: str = "", ttl: int = 86400 * 7,
                 max_entries: int = 100000):
        if not db_path:
            db_path = str(Path.home() / ".fusion-rag" / "embed_cache.db")
        self.db_path = db_path
        self.ttl = ttl
        self.max_entries = max_entries
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
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
        conn.close()

    def _hash(self, text: str, model: str = "") -> str:
        key = f"{model}:{text}"
        return hashlib.md5(key.encode()).hexdigest()

    def get(self, text: str, model: str = "") -> list[float] | None:
        h = self._hash(text, model)
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
        finally:
            conn.close()

    def set(self, text: str, vector: list[float], model: str = "") -> None:
        h = self._hash(text, model)
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
        finally:
            conn.close()

    def get_batch(self, texts: list[str], model: str = "") -> list[list[float] | None]:
        return [self.get(t, model) for t in texts]

    def set_batch(self, texts: list[str], vectors: list[list[float]],
                  model: str = "") -> None:
        h = self._hash
        conn = self._get_conn()
        try:
            now = time.time()
            rows = [
                (h(t, model), t[:1000], json.dumps(v), model, now)
                for t, v in zip(texts, vectors)
            ]
            conn.executemany(
                """INSERT OR REPLACE INTO embed_cache (text_hash, text, vector, model, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                rows,
            )
            conn.commit()
            self._evict_if_needed(conn)
        except Exception as e:
            logger.warning("EmbeddingCache set_batch failed: %s", e)
        finally:
            conn.close()

    def clear(self) -> None:
        conn = self._get_conn()
        conn.execute("DELETE FROM embed_cache")
        conn.commit()
        conn.close()

    def count(self) -> int:
        conn = self._get_conn()
        row = conn.execute("SELECT COUNT(*) as cnt FROM embed_cache").fetchone()
        conn.close()
        return row["cnt"] if row else 0

    def _evict_if_needed(self, conn: sqlite3.Connection) -> None:
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
