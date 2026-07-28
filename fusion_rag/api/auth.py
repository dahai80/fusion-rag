"""API authentication — API key-based auth for Fusion-RAG endpoints.

callers: routes.py via FastAPI dependency injection
API: verify_api_key() dependency, AuthConfig for key management
schema: api_keys table (key_hash TEXT PK, name TEXT, created_at REAL)
user instruction: "按照你的方案和计划落地所有phase阶段的需求"
"""
from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


class AuthConfig:
    """Manages API keys with SQLite persistence."""

    def __init__(self, db_path: str = ""):
        if not db_path:
            db_path = str(Path.home() / ".fusion-rag" / "auth.db")
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS api_keys (
                key_hash TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_at REAL NOT NULL
            );
        """)
        conn.commit()
        conn.close()

    @staticmethod
    def _hash_key(key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()

    def add_key(self, key: str, name: str = "default") -> bool:
        h = self._hash_key(key)
        conn = self._get_conn()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO api_keys (key_hash, name, created_at) VALUES (?, ?, ?)",
                (h, name, time.time()),
            )
            conn.commit()
            return True
        except Exception as e:
            logger.warning("Failed to add API key: %s", e)
            return False
        finally:
            conn.close()

    def validate_key(self, key: str) -> bool:
        if not key:
            return False
        h = self._hash_key(key)
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT name FROM api_keys WHERE key_hash = ?", (h,)
            ).fetchone()
            return row is not None
        finally:
            conn.close()

    def remove_key(self, key: str) -> bool:
        h = self._hash_key(key)
        conn = self._get_conn()
        try:
            cursor = conn.execute("DELETE FROM api_keys WHERE key_hash = ?", (h,))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def list_keys(self) -> list[dict[str, Any]]:
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT key_hash, name, created_at FROM api_keys ORDER BY created_at DESC"
            ).fetchall()
            return [
                {"key_hash": r["key_hash"][:12] + "...", "name": r["name"],
                 "created_at": r["created_at"]}
                for r in rows
            ]
        finally:
            conn.close()


def verify_api_key(
    api_key: str | None = Security(API_KEY_HEADER),
) -> str | None:
    """FastAPI dependency for API key verification.

    Returns None if auth is disabled (no FUSION_RAG_API_KEY env var).
    Raises HTTPException 401 if auth is enabled and key is invalid.
    """
    admin_key = os.environ.get("FUSION_RAG_API_KEY", "")
    if not admin_key:
        return None  # Auth disabled

    if not api_key:
        raise HTTPException(401, "API key required. Set X-API-Key header.")

    auth = AuthConfig()
    if api_key == admin_key or auth.validate_key(api_key):
        return api_key

    raise HTTPException(401, "Invalid API key")
