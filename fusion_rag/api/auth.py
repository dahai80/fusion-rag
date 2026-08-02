"""API authentication — pluggable auth backend for Fusion-RAG endpoints.

callers: routes.py via FastAPI dependency injection
API: verify_api_key() dependency, AuthBackend ABC, AuthConfig for key management
schema: api_keys table (key_hash TEXT PK, name TEXT, created_at REAL)
user instruction: "修复所有的issue和pr"
"""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


class AuthBackend(ABC):
    """Abstract auth backend — pluggable authentication strategy."""

    @abstractmethod
    def verify(self, api_key: str | None) -> str | None:
        """Verify API key. Returns identity string or None if auth disabled.
        Raises HTTPException on invalid key."""


class NoAuthBackend(AuthBackend):
    """No-op backend — all requests pass through."""

    def verify(self, api_key: str | None) -> str | None:
        return None


class ApiKeyBackend(AuthBackend):
    """API key validation against env var + SQLite key store."""

    def __init__(self, admin_key: str = "", db_path: str = ""):
        self.admin_key = admin_key
        self.auth_config = AuthConfig(db_path) if admin_key else AuthConfig()

    def verify(self, api_key: str | None) -> str | None:
        if not self.admin_key:
            return None
        if not api_key:
            raise HTTPException(401, "API key required. Set X-API-Key header.")
        if api_key == self.admin_key or self.auth_config.validate_key(api_key):
            return api_key
        raise HTTPException(401, "Invalid API key")


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
            row = conn.execute("SELECT name FROM api_keys WHERE key_hash = ?", (h,)).fetchone()
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
            rows = conn.execute("SELECT key_hash, name, created_at FROM api_keys ORDER BY created_at DESC").fetchall()
            return [
                {"key_hash": r["key_hash"][:12] + "...", "name": r["name"], "created_at": r["created_at"]} for r in rows
            ]
        finally:
            conn.close()


_auth_backend: AuthBackend | None = None


def get_auth_backend() -> AuthBackend:
    global _auth_backend
    if _auth_backend is not None:
        return _auth_backend
    backend_name = os.environ.get("FUSION_RAG_AUTH_BACKEND", "apikey")
    if backend_name == "none":
        _auth_backend = NoAuthBackend()
    else:
        admin_key = os.environ.get("FUSION_RAG_API_KEY", "")
        _auth_backend = ApiKeyBackend(admin_key=admin_key)
    logger.info("auth backend: %s", type(_auth_backend).__name__)
    return _auth_backend


def verify_api_key(
    api_key: str | None = Security(API_KEY_HEADER),
) -> str | None:
    """FastAPI dependency for API key verification — delegates to pluggable backend."""
    return get_auth_backend().verify(api_key)
