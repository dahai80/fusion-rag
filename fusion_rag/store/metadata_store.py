"""Metadata store — SQLite-based metadata for knowledge bases, documents, and chunks."""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class MetadataStore:
    """SQLite-backed metadata for tracking documents and chunks."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
        return self._conn

    @contextmanager
    def _cursor(self):
        conn = self._get_conn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def _init_db(self) -> None:
        with self._cursor() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    doc_type TEXT NOT NULL,
                    chunk_count INTEGER DEFAULT 0,
                    char_count INTEGER DEFAULT 0,
                    file_size INTEGER DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    doc_path TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    tokens INTEGER DEFAULT 0,
                    created_at REAL NOT NULL,
                    FOREIGN KEY (doc_id) REFERENCES documents(id)
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
                CREATE INDEX IF NOT EXISTS idx_docs_path ON documents(file_path);
            """)

    def add_document(self, doc_id: str, file_path: str, file_name: str,
                     doc_type: str, file_size: int = 0) -> None:
        now = time.time()
        with self._cursor() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO documents
                   (id, file_path, file_name, doc_type, file_size, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (doc_id, file_path, file_name, doc_type, file_size, now, now),
            )

    def delete_document(self, doc_id: str) -> None:
        with self._cursor() as conn:
            conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        with self._cursor() as conn:
            row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        return dict(row) if row else None

    def get_document_by_path(self, file_path: str) -> dict[str, Any] | None:
        with self._cursor() as conn:
            row = conn.execute("SELECT * FROM documents WHERE file_path = ?", (file_path,)).fetchone()
        return dict(row) if row else None

    def list_documents(self) -> list[dict[str, Any]]:
        with self._cursor() as conn:
            rows = conn.execute("SELECT * FROM documents ORDER BY updated_at DESC").fetchall()
        return [dict(r) for r in rows]

    def update_chunk_count(self, doc_id: str, count: int, chars: int) -> None:
        now = time.time()
        with self._cursor() as conn:
            conn.execute(
                "UPDATE documents SET chunk_count = ?, char_count = ?, updated_at = ? WHERE id = ?",
                (count, chars, now, doc_id),
            )

    def add_chunk(self, chunk_id: str, doc_id: str, doc_path: str,
                  chunk_index: int, text: str, tokens: int = 0) -> None:
        with self._cursor() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO chunks
                   (id, doc_id, doc_path, chunk_index, text, tokens, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (chunk_id, doc_id, doc_path, chunk_index, text, tokens, time.time()),
            )

    def delete_chunks_by_doc(self, doc_id: str) -> None:
        with self._cursor() as conn:
            conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))

    def get_chunks_by_doc(self, doc_id: str) -> list[dict[str, Any]]:
        with self._cursor() as conn:
            rows = conn.execute(
                "SELECT * FROM chunks WHERE doc_id = ? ORDER BY chunk_index", (doc_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def doc_count(self) -> int:
        with self._cursor() as conn:
            row = conn.execute("SELECT COUNT(*) as cnt FROM documents").fetchone()
        return row["cnt"] if row else 0

    def chunk_count(self) -> int:
        with self._cursor() as conn:
            row = conn.execute("SELECT COUNT(*) as cnt FROM chunks").fetchone()
        return row["cnt"] if row else 0

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None