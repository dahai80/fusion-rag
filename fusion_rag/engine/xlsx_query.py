"""Issue #161: structured XLSX query — text-to-SQL over sheet schema.

Ingest: each sheet becomes a real SQLite table in the per-KB sheets.db
(column names, inferred types, all rows) alongside a sheet_tables metadata
row. The existing markdown chunks stay untouched — structured retrieval is an
additional tier, never a replacement.

Query: an LLM planner translates the question into ONE SQLite SELECT over the
schema DDL; the SQL is validated by code (deterministic guards, never trusted)
and executed on a READ-ONLY connection with an enforced row cap. Aggregation
happens in SQLite, not LLM arithmetic over chunks — results carry sheet
citations (doc_id, doc_name, sheet_name, row_count).
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any

from .llm_errors import LLMUnavailable
from .sqlite_base import SqliteBase

logger = logging.getLogger(__name__)

MAX_COLUMNS = 256
MAX_ROWS_STORED = 100_000
DEFAULT_MAX_QUERY_ROWS = 100
HARD_MAX_QUERY_ROWS = 1000

_TABLE_RE = re.compile(r"^[0-9a-z_]+$")
_FORBIDDEN_RE = re.compile(
    r"\b(pragma|attach|detach|insert|update|delete|drop|alter|create|replace|"
    r"vacuum|reindex|analyze|grant|revoke|savepoint|release|begin|commit|rollback)\b",
    re.IGNORECASE,
)
_LIMIT_RE = re.compile(r"\blimit\s+\d+", re.IGNORECASE)


def get_xlsx_index(db_path: str) -> XlsxSheetIndex:
    # Pooled like get_sparse_index: one shared connection per KB file.
    global _pool, _pool_lock
    with _pool_lock:
        idx = _pool.get(db_path)
        if idx is None:
            idx = XlsxSheetIndex(db_path)
            _pool[db_path] = idx
        return idx


_pool: dict[str, XlsxSheetIndex] = {}
_pool_lock = threading.Lock()


def table_name(doc_id: str, sheet_idx: int) -> str:
    return f"t_{re.sub(r'[^0-9a-z_]', '', doc_id.lower())}_{sheet_idx}"


def _quote_ident(name: str) -> str:
    if not _TABLE_RE.match(name):
        raise ValueError(f"invalid sql identifier: {name!r}")
    return f'"{name}"'


class XlsxSheetIndex(SqliteBase):
    """Per-KB sheet store: metadata rows + one real SQLite table per sheet."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        super().__init__()
        self._init_db()

    def _init_db(self) -> None:
        conn = self._get_conn()
        with self._db_lock:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS sheet_tables (
                       doc_id TEXT NOT NULL,
                       doc_name TEXT NOT NULL,
                       sheet_name TEXT NOT NULL,
                       table_name TEXT NOT NULL,
                       columns_json TEXT NOT NULL,
                       row_count INTEGER NOT NULL,
                       PRIMARY KEY (doc_id, sheet_name)
                   )"""
            )
            conn.commit()

    def replace_doc(self, doc_id: str, doc_name: str, sheets: list[dict[str, Any]]) -> int:
        # sheets: [{sheet_name, table_name, columns: [{name, header, dtype}], rows}]
        conn = self._get_conn()
        with self._db_lock:
            old = conn.execute(
                "SELECT table_name FROM sheet_tables WHERE doc_id = ?", (doc_id,)
            ).fetchall()
            for row in old:
                conn.execute(f"DROP TABLE IF EXISTS {_quote_ident(row['table_name'])}")
            conn.execute("DELETE FROM sheet_tables WHERE doc_id = ?", (doc_id,))
            for s in sheets:
                conn.execute(
                    "INSERT INTO sheet_tables (doc_id, doc_name, sheet_name, table_name, columns_json, row_count)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        doc_id,
                        doc_name,
                        s["sheet_name"],
                        s["table_name"],
                        json.dumps(s["columns"], ensure_ascii=False),
                        s["row_count"],
                    ),
                )
                col_defs = ", ".join(
                    f"{_quote_ident(c['name'])} {c['dtype']}" for c in s["columns"]
                )
                conn.execute(
                    f"CREATE TABLE {_quote_ident(s['table_name'])} ({col_defs})"
                )
                if s["rows"]:
                    col_names = ", ".join(_quote_ident(c["name"]) for c in s["columns"])
                    placeholders = ", ".join("?" for _ in s["columns"])
                    conn.executemany(
                        f"INSERT INTO {_quote_ident(s['table_name'])} ({col_names})"
                        f" VALUES ({placeholders})",
                        s["rows"],
                    )
            conn.commit()
        logger.info(
            "sheets stored for doc %s (%s): %d tables", doc_id, doc_name, len(sheets)
        )
        return len(sheets)

    def delete_doc(self, doc_id: str) -> int:
        conn = self._get_conn()
        with self._db_lock:
            rows = conn.execute(
                "SELECT table_name FROM sheet_tables WHERE doc_id = ?", (doc_id,)
            ).fetchall()
            for row in rows:
                conn.execute(f"DROP TABLE IF EXISTS {_quote_ident(row['table_name'])}")
            conn.execute("DELETE FROM sheet_tables WHERE doc_id = ?", (doc_id,))
            conn.commit()
            return len(rows)

    def list_sheets(self, doc_id: str | None = None) -> list[dict[str, Any]]:
        conn = self._get_conn()
        if doc_id:
            rows = conn.execute(
                "SELECT doc_id, doc_name, sheet_name, table_name, columns_json, row_count"
                " FROM sheet_tables WHERE doc_id = ? ORDER BY doc_name, sheet_name",
                (doc_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT doc_id, doc_name, sheet_name, table_name, columns_json, row_count"
                " FROM sheet_tables ORDER BY doc_name, sheet_name"
            ).fetchall()
        out = []
        for r in rows:
            out.append(
                {
                    "doc_id": r["doc_id"],
                    "doc_name": r["doc_name"],
                    "sheet_name": r["sheet_name"],
                    "table_name": r["table_name"],
                    "columns": json.loads(r["columns_json"]),
                    "row_count": r["row_count"],
                }
            )
        return out

    def run_sql(self, sql: str, max_rows: int = DEFAULT_MAX_QUERY_ROWS) -> dict[str, Any]:
        """Execute validated SQL on a READ-ONLY connection with an enforced cap."""
        max_rows = max(1, min(int(max_rows), HARD_MAX_QUERY_ROWS))
        # Truncation is only observable if the executed SQL can return
        # max_rows+1 rows: when the planner omitted a LIMIT, wrap with one
        # extra row; a planner-supplied LIMIT is its own intent (a deeper cap
        # than max_rows still trips the fetchmany(max_rows+1) probe).
        if not _LIMIT_RE.search(sql):
            exec_sql = f"SELECT * FROM ({sql}) AS q LIMIT {max_rows + 1}"
        else:
            exec_sql = sql
        # Read-only URI connection: even a guard bypass cannot mutate the KB's
        # sheet store. Own connection per call (not the pooled writer).
        uri = f"file:{self.db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        try:
            cur = conn.execute(exec_sql)
            columns = [d[0] for d in cur.description]
            rows = cur.fetchmany(max_rows + 1)
            truncated = len(rows) > max_rows
            rows = rows[:max_rows]
            return {
                "columns": columns,
                "rows": [list(r) for r in rows],
                "row_count": len(rows),
                "truncated": truncated,
                "sql": exec_sql,
            }
        finally:
            conn.close()


def build_sheets_from_xlsx(path: str | Path) -> list[dict[str, Any]]:
    """Extract every sheet as a typed SQLite-ready table spec.

    First non-empty row is the header. Column types are inferred from a sample
    of the first 100 non-null values (INTEGER / REAL / TEXT). Rows are capped
    at MAX_ROWS_STORED — row_count always reports the TRUE total so a query
    over a truncated sheet is visibly wrong rather than silently wrong."""
    from openpyxl import load_workbook

    wb = load_workbook(str(path), read_only=True, data_only=True)
    sheets: list[dict[str, Any]] = []
    try:
        for idx, ws in enumerate(wb.worksheets):
            rows_raw: list[tuple] = []
            total = 0
            header: list[str] = []
            for row in ws.iter_rows(values_only=True):
                cells = ["" if v is None else v for v in row]
                if not any(str(c).strip() for c in cells):
                    continue
                if not header:
                    header = [str(c).strip() or f"col_{i + 1}" for i, c in enumerate(cells)]
                    continue
                total += 1
                if len(rows_raw) < MAX_ROWS_STORED:
                    rows_raw.append(tuple(cells))
            if not header:
                continue
            n_cols = len(header)
            dtypes = _infer_dtypes(rows_raw, n_cols)
            columns = [
                {
                    "name": f"_c{i + 1}",
                    "header": header[i][:200],
                    "dtype": dtypes[i],
                }
                for i in range(min(n_cols, MAX_COLUMNS))
            ]
            # Trim each stored row to the kept columns and stringify non-primitives
            # (openpyxl can hand back datetime / decimal objects SQLite rejects).
            stored = []
            for r in rows_raw:
                stored.append(
                    tuple(
                        _coerce(v, columns[i]["dtype"])
                        for i, v in enumerate(r[: len(columns)])
                    )
                )
            sheets.append(
                {
                    "sheet_name": ws.title,
                    "table_name": f"t_pending_{idx}",
                    "columns": columns,
                    "row_count": total,
                    "rows": stored,
                }
            )
            if total > MAX_ROWS_STORED:
                logger.warning(
                    "sheet %s in %s truncated: %d of %d rows stored",
                    ws.title,
                    path,
                    MAX_ROWS_STORED,
                    total,
                )
    finally:
        wb.close()
    return sheets


def _coerce(value: Any, dtype: str) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value
    if dtype == "INTEGER":
        try:
            return int(str(value).strip())
        except ValueError:
            return str(value).strip()
    if dtype == "REAL":
        try:
            return float(str(value).strip())
        except ValueError:
            return str(value).strip()
    return str(value).strip()


def _infer_dtypes(rows: list[tuple], n_cols: int) -> list[str]:
    sample = rows[:100]
    dtypes = []
    for i in range(n_cols):
        values = [r[i] for r in sample if i < len(r) and r[i] is not None and str(r[i]).strip()]
        if not values:
            dtypes.append("TEXT")
            continue
        if all(isinstance(v, (bool, int)) for v in values):
            dtypes.append("INTEGER")
        elif all(isinstance(v, (int, float)) for v in values):
            dtypes.append("REAL")
        else:
            dtypes.append("TEXT")
    return dtypes


def validate_sql(sql: str) -> str:
    """Deterministic guards on LLM-generated SQL. Never trust the model output.

    Returns the cleaned single-statement SELECT. Raises ValueError with an
    actionable message on anything that is not a plain read."""
    cleaned = sql.strip()
    # Strip markdown fences if the model added them despite instructions.
    fence = re.match(r"^```[a-zA-Z]*\n(.*)\n```$", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    if not cleaned:
        raise ValueError("empty SQL")
    if ";" in cleaned:
        raise ValueError("multiple SQL statements are not allowed")
    first = re.match(r"^\s*(\w+)", cleaned)
    if not first or first.group(1).upper() not in ("SELECT", "WITH"):
        raise ValueError("only SELECT queries are allowed")
    m = _FORBIDDEN_RE.search(cleaned)
    if m:
        raise ValueError(f"forbidden keyword in query: {m.group(1).upper()}")
    return cleaned


def schema_ddl(sheets: list[dict[str, Any]]) -> str:
    """Render the sheet schema as CREATE TABLE statements for the planner prompt."""
    blocks = []
    for s in sheets:
        lines = [f'-- sheet "{s["sheet_name"]}" from document "{s["doc_name"]}" ({s["row_count"]} rows)']
        lines.append(f"CREATE TABLE {_quote_ident(s['table_name'])} (")
        cols = s["columns"]
        col_lines = [
            f"  {_quote_ident(c['name'])} {c['dtype']}"
            + ("," if i < len(cols) - 1 else "")
            + f" -- header: {c['header']}"
            for i, c in enumerate(cols)
        ]
        lines.append("\n".join(col_lines))
        lines.append(");")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


SQL_PROMPT = (
    "You are a SQLite expert. Given the table schema below, write ONE SQLite "
    "SELECT query that answers the question.\n"
    "Rules:\n"
    "- Use ONLY the tables and columns defined in the schema.\n"
    "- Use column names (the quoted _cN identifiers), NOT the header comments.\n"
    "- Output ONLY the SQL statement, no explanation, no markdown fences, no "
    "semicolon.\n"
    "- Compute aggregations (SUM/COUNT/AVG/GROUP BY) in SQL, not in the answer.\n\n"
    "Schema:\n{ddl}\n\nQuestion: {question}"
)

ANSWER_PROMPT = (
    "Answer the question using ONLY the computed rows below. The aggregation "
    "was already performed — do NOT recompute or estimate numbers, quote the "
    "exact values from the rows. Mention which sheet the data came from.\n\n"
    "Question: {question}\n\n"
    "Computed rows (columns: {columns}):\n{rows}"
)


class XlsxQueryPlanner:
    """LLM engine: question -> one SQLite SELECT; rows -> cited answer text.

    Same wiring discipline as every LLM engine (#72): mlx_url + api_key come
    from the shared EmbeddingClient at construction, never hardcoded."""

    def __init__(
        self,
        mlx_url: str = "http://127.0.0.1:11432/v1",
        model: str = "qwen3.5-9b",
        api_key: str = "",
    ):
        self.mlx_url = mlx_url.rstrip("/")
        self.model = model
        self.api_key = api_key

    def _auth_headers(self) -> dict[str, str]:
        if self.api_key:
            return {"Authorization": f"Bearer {self.api_key}"}
        return {}

    async def _chat(self, user_prompt: str, max_tokens: int = 1024) -> str:
        from fusion_core.http_client import get_async_client, with_retry

        client = get_async_client(self.mlx_url, timeout=60.0)
        try:
            resp = await with_retry(
                lambda: client.post(
                    f"{self.mlx_url}/chat/completions",
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": user_prompt}],
                        "max_tokens": max_tokens,
                        "temperature": 0.0,
                    },
                    headers=self._auth_headers(),
                )
            )
            resp.raise_for_status()
            body = resp.json()
            return (body["choices"][0]["message"]["content"] or "").strip()
        except LLMUnavailable:
            raise
        except Exception as e:
            logger.warning("xlsx planner chat failed: %s", e)
            raise LLMUnavailable(f"LLM unavailable for xlsx query: {e}") from e

    async def generate_sql(self, ddl: str, question: str) -> str:
        raw = await self._chat(SQL_PROMPT.format(ddl=ddl, question=question))
        return raw

    async def compose_answer(self, question: str, columns: list[str], rows: list[list], sheet_names: list[str]) -> str:
        rendered = "\n".join(
            " | ".join("" if v is None else str(v) for v in r) for r in rows
        )
        prompt = ANSWER_PROMPT.format(
            question=question,
            columns=", ".join(columns),
            rows=rendered or "(no rows)",
        )
        if sheet_names:
            prompt += f"\n\nData source sheet(s): {', '.join(sheet_names)}"
        return await self._chat(prompt, max_tokens=2048)
