"""Issue #161: structured XLSX query surface (/kb/bases/{kb_id}/xlsx/*).

Sheets are indexed at ingest (engine/xlsx_query.py). Here: schema
introspection (no LLM) + the text-to-SQL query path. The planner's SQL is
validated by deterministic guards and run on a read-only connection — a
malformed or hostile generation degrades to a 400/502 so the caller can fall
back to the regular /search markdown path, never fabricates a result."""

from __future__ import annotations

import logging
import re
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool

from .access import require_kb_action

logger = logging.getLogger(__name__)

router = APIRouter(tags=["xlsx"])

DEFAULT_MODEL = "qwen3.5-9b"


def _xlsx_index(kb_id: str):
    from ..engine.xlsx_query import get_xlsx_index
    from .routes import _get_kb_storage_path

    return get_xlsx_index(f"{_get_kb_storage_path(kb_id)}/sheets.db")


def _planner(model: str):
    from ..engine.xlsx_query import XlsxQueryPlanner
    from .routes import _get_embed_client

    embed = _get_embed_client()
    return XlsxQueryPlanner(mlx_url=embed.base_url, model=model, api_key=embed.api_key)


@router.get("/bases/{kb_id}/xlsx/sheets", dependencies=[Depends(require_kb_action("read"))])
async def list_sheets(kb_id: str, doc_id: str | None = None) -> dict[str, Any]:
    from .routes import _get_base

    _get_base(kb_id)
    sheets = await run_in_threadpool(_xlsx_index(kb_id).list_sheets, doc_id)
    if not sheets:
        raise HTTPException(404, f"no structured sheets indexed for KB '{kb_id}'")
    return {"kb_id": kb_id, "count": len(sheets), "sheets": sheets}


@router.post("/bases/{kb_id}/xlsx/query", dependencies=[Depends(require_kb_action("search"))])
async def xlsx_query(kb_id: str, data: dict[str, Any]) -> dict[str, Any]:
    from ..engine.xlsx_query import (
        DEFAULT_MAX_QUERY_ROWS,
        HARD_MAX_QUERY_ROWS,
        schema_ddl,
        validate_sql,
    )
    from .routes import _get_base

    _get_base(kb_id)
    question = str(data.get("question", "")).strip()
    if not question:
        raise HTTPException(400, "question is required")
    doc_id = data.get("doc_id") or None
    max_rows = int(data.get("max_rows", DEFAULT_MAX_QUERY_ROWS))
    max_rows = max(1, min(max_rows, HARD_MAX_QUERY_ROWS))
    want_answer = bool(data.get("answer", False))
    model = str(data.get("model", DEFAULT_MODEL))

    index = _xlsx_index(kb_id)
    sheets = await run_in_threadpool(index.list_sheets, doc_id)
    if not sheets:
        raise HTTPException(
            404,
            "no structured sheets indexed for this KB"
            + (f" (doc_id={doc_id})" if doc_id else "")
            + " — ingest an XLSX document first",
        )

    planner = _planner(model)
    ddl = schema_ddl(sheets)
    try:
        raw_sql = await planner.generate_sql(ddl, question)
        sql = validate_sql(raw_sql)
    except ValueError as e:
        # Bad generation — caller falls back to /search (markdown chunks).
        logger.warning("xlsx query SQL rejected for kb %s: %s", kb_id, e)
        raise HTTPException(400, f"generated SQL rejected: {e}")
    except Exception as e:
        logger.error("xlsx query planner failed for kb %s: %s", kb_id, e)
        raise HTTPException(502, f"LLM unavailable for xlsx query: {e}")

    try:
        result = await run_in_threadpool(index.run_sql, sql, max_rows)
    except Exception as e:
        logger.warning("xlsx query execution failed for kb %s: %s", kb_id, e)
        raise HTTPException(400, f"SQL execution failed: {e}")

    # Citations: every sheet whose table the SQL actually touched; unknown
    # table names fall back to the full scoped sheet set.
    used = set(re.findall(r'(?i)\bt_[0-9a-z_]+', sql))
    cited = [s for s in sheets if s["table_name"] in used] or sheets
    citations = [
        {
            "doc_id": s["doc_id"],
            "doc_name": s["doc_name"],
            "sheet_name": s["sheet_name"],
            "table_name": s["table_name"],
            "row_count": s["row_count"],
        }
        for s in cited
    ]

    answer = None
    if want_answer:
        try:
            answer = await planner.compose_answer(
                question,
                result["columns"],
                result["rows"],
                [s["sheet_name"] for s in cited],
            )
        except Exception as e:
            # Answer text is an enhancement — computed rows are the contract.
            logger.warning("xlsx answer composition failed for kb %s (non-fatal): %s", kb_id, e)

    logger.info(
        "xlsx query kb=%s sql_rows=%d truncated=%s citations=%d",
        kb_id,
        result["row_count"],
        result["truncated"],
        len(citations),
    )
    return {
        "question": question,
        "sql": result["sql"],
        "columns": result["columns"],
        "rows": result["rows"],
        "row_count": result["row_count"],
        "truncated": result["truncated"],
        "citations": citations,
        "answer": answer,
    }
