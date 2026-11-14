"""Connectors — data source connectors for databases and web content."""

from __future__ import annotations

import logging
from typing import Any

from .._validators import validate_sql_identifier, validate_url

logger = logging.getLogger(__name__)


class DatabaseConnector:
    """Connect to SQLite/PostgreSQL databases and extract table data as documents."""

    def __init__(self, db_type: str = "sqlite", connection_string: str = ""):
        self.db_type = db_type
        self.connection_string = connection_string

    async def list_tables(self, schema: str = "public") -> list[dict[str, Any]]:
        """List all tables and their columns."""
        if self.db_type == "sqlite":
            return self._list_sqlite_tables()
        elif self.db_type == "postgresql":
            return await self._list_postgres_tables(schema)
        return []

    async def fetch_table(self, table_name: str, limit: int = 100) -> list[dict[str, Any]]:
        """Fetch rows from a table as documents."""
        if self.db_type == "sqlite":
            return self._fetch_sqlite(table_name, limit)
        elif self.db_type == "postgresql":
            return await self._fetch_postgres(table_name, limit)
        return []

    def _list_sqlite_tables(self) -> list[dict[str, Any]]:
        import sqlite3

        try:
            conn = sqlite3.connect(self.connection_string)
            # F5 fix: tname comes from sqlite_master (trusted), but validate +
            # quote-identifier before f-string to harden against a poisoned table name.
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = []
            for row in cursor.fetchall():
                tname = row[0]
                try:
                    tname = validate_sql_identifier(tname, field="table_name")
                except ValueError:
                    logger.warning("skip table with invalid identifier: %r", tname)
                    continue
                quoted = f'"{tname}"'
                cols = conn.execute(f"PRAGMA table_info({quoted})").fetchall()
                tables.append(
                    {
                        "name": tname,
                        "columns": [{"name": c[1], "type": c[2]} for c in cols],
                        "row_count": conn.execute(f"SELECT COUNT(*) FROM {quoted}").fetchone()[0],
                    }
                )
            conn.close()
            return tables
        except Exception as e:
            logger.error("SQLite connection failed: %s", e)
            return []

    def _fetch_sqlite(self, table_name: str, limit: int) -> list[dict[str, Any]]:
        import sqlite3

        # F5 fix: validate identifier before interpolation; quote with double quotes.
        validate_sql_identifier(table_name, field="table_name")
        try:
            conn = sqlite3.connect(self.connection_string)
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(f'SELECT * FROM "{table_name}" LIMIT ?', (limit,))
            rows = [dict(r) for r in cursor.fetchall()]
            conn.close()
            return rows
        except Exception as e:
            logger.error("SQLite fetch failed: %s", e)
            return []

    async def _list_postgres_tables(self, schema: str) -> list[dict[str, Any]]:
        try:
            import asyncpg

            conn = await asyncpg.connect(self.connection_string)
            rows = await conn.fetch(
                """
                SELECT table_name, column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = $1 ORDER BY table_name, ordinal_position
            """,
                schema,
            )
            await conn.close()
            tables = {}
            for row in rows:
                tname = row["table_name"]
                if tname not in tables:
                    tables[tname] = {"name": tname, "columns": [], "row_count": 0}
                tables[tname]["columns"].append({"name": row["column_name"], "type": row["data_type"]})
            return list(tables.values())
        except ImportError:
            logger.warning("asyncpg not installed, skipping PostgreSQL")
            return []
        except Exception as e:
            logger.error("PostgreSQL connection failed: %s", e)
            return []

    async def _fetch_postgres(self, table_name: str, limit: int) -> list[dict[str, Any]]:
        # F5 fix: validate identifier; confirm table exists in information_schema
        # before SELECT (defense in depth).
        validate_sql_identifier(table_name, field="table_name")
        try:
            import asyncpg

            conn = await asyncpg.connect(self.connection_string)
            exists = await conn.fetchval(
                "SELECT 1 FROM information_schema.tables WHERE table_name = $1", table_name
            )
            if not exists:
                logger.warning("PostgreSQL table not found: %s", table_name)
                await conn.close()
                return []
            # asyncpg has no quote_ident on conn; use $1 for the existence check above
            # and trust the whitelist validation for the SELECT identifier.
            rows = await conn.fetch(f'SELECT * FROM "{table_name}" LIMIT $1', limit)
            await conn.close()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.error("PostgreSQL fetch failed: %s", e)
            return []


class WebLoader:
    """Fetch and extract text content from web pages."""

    async def load(self, url: str, max_chars: int = 10000) -> dict[str, Any]:
        """Fetch a URL and extract its text content."""
        import httpx

        # F6 fix: scheme whitelist + SSRF guard (reject private/loopback).
        validate_url(url, field="url")
        # Byte budget = max_chars*4 (UTF-8 worst case); stream-read to bound memory.
        byte_budget = max(1024, max_chars * 4)
        try:
            async with httpx.AsyncClient(
                timeout=15.0,
                follow_redirects=True,
                max_redirects=3,
            ) as client, client.stream("GET", url) as resp:
                resp.raise_for_status()
                # Stream so a 10GB body never fully enters RAM.
                collected = bytearray()
                async for chunk in resp.aiter_bytes(chunk_size=8192):
                    collected.extend(chunk)
                    if len(collected) >= byte_budget:
                        logger.info(
                            "WebLoader stream truncated at %d bytes (budget=%d)",
                            len(collected),
                            byte_budget,
                        )
                        break
                text = collected[:byte_budget].decode("utf-8", errors="replace")
                text = text[:max_chars]
                return {
                    "url": url,
                    "content": self._extract_text(text),
                    "chars": len(text),
                    "status": resp.status_code,
                }
        except Exception as e:
            logger.warning("WebLoader load failed for %s: %s", url, e)
            return {"url": url, "content": "", "chars": 0, "error": str(e)}

    @staticmethod
    def _extract_text(html: str) -> str:
        """Simple HTML to text extraction."""
        import re

        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:5000]
