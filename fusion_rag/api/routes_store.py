from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool

from .._validators import validate_identifier
from .auth import verify_api_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/bases/{kb_id}/store", tags=["store"], dependencies=[Depends(verify_api_key)])


def _get_base(kb_id: str):
    try:
        validate_identifier(kb_id, field="kb_id")
    except ValueError:
        raise HTTPException(400, f"Invalid kb_id: {kb_id}")
    # Tenant scoping: the store surface is M2M (another fusion-rag node acting
    # as RemoteBackend). That caller authenticates via X-API-Key and does NOT
    # carry the gateway-origin headers (it is not a gateway-routed user
    # request). Exempt the /store path from the gateway-origin gate in
    # tenant_middleware so a correctly-authenticated node call is not 403'd;
    # tenant scoping here is read-only-safe (a node calls its own kb_id).
    from .routes import _get_base as _routes_get_base

    return _routes_get_base(kb_id)


def _vec_store(kb_id: str):
    from .routes import _get_vector_store

    return _get_vector_store(kb_id)


@router.post("/add_batch")
async def add_batch(kb_id: str, data: dict[str, Any]) -> dict[str, Any]:
    _get_base(kb_id)
    records = data.get("records")
    if not isinstance(records, list) or not records:
        raise HTTPException(400, "records (non-empty list) required")
    try:
        await run_in_threadpool(_vec_store(kb_id).add_batch, records)
    except Exception as e:
        logger.error("store add_batch failed for kb=%s: %s", kb_id, e)
        raise HTTPException(500, f"add_batch failed: {e}")
    return {"ok": True}


@router.post("/search")
async def search(kb_id: str, data: dict[str, Any]) -> dict[str, Any]:
    _get_base(kb_id)
    query_vector = data.get("query_vector")
    if not isinstance(query_vector, list):
        raise HTTPException(400, "query_vector (list) required")
    top_k = int(data.get("top_k", 10))
    threshold = float(data.get("threshold", 0.0))
    try:
        results = await run_in_threadpool(_vec_store(kb_id).search, query_vector, top_k, threshold)
    except Exception as e:
        logger.error("store search failed for kb=%s: %s", kb_id, e)
        raise HTTPException(500, f"search failed: {e}")
    return {"results": results}


@router.post("/keyword_search")
async def keyword_search(kb_id: str, data: dict[str, Any]) -> dict[str, Any]:
    _get_base(kb_id)
    query = data.get("query", "")
    if not query:
        raise HTTPException(400, "query required")
    top_k = int(data.get("top_k", 10))
    try:
        results = await run_in_threadpool(_vec_store(kb_id).keyword_search, query, top_k)
    except Exception as e:
        logger.error("store keyword_search failed for kb=%s: %s", kb_id, e)
        raise HTTPException(500, f"keyword_search failed: {e}")
    return {"results": results}


@router.post("/delete_by_doc")
async def delete_by_doc(kb_id: str, data: dict[str, Any]) -> dict[str, Any]:
    _get_base(kb_id)
    doc_path = data.get("doc_path", "")
    if not doc_path:
        raise HTTPException(400, "doc_path required")
    try:
        deleted = await run_in_threadpool(_vec_store(kb_id).delete_by_doc, doc_path)
    except Exception as e:
        logger.error("store delete_by_doc failed for kb=%s: %s", kb_id, e)
        raise HTTPException(500, f"delete_by_doc failed: {e}")
    return {"deleted": int(deleted)}


@router.get("/count")
async def count(kb_id: str) -> dict[str, Any]:
    _get_base(kb_id)
    try:
        n = await run_in_threadpool(_vec_store(kb_id).count)
    except Exception as e:
        logger.error("store count failed for kb=%s: %s", kb_id, e)
        raise HTTPException(500, f"count failed: {e}")
    return {"count": int(n)}


@router.post("/clear")
async def clear(kb_id: str) -> dict[str, Any]:
    _get_base(kb_id)
    try:
        await run_in_threadpool(_vec_store(kb_id).clear)
    except Exception as e:
        logger.error("store clear failed for kb=%s: %s", kb_id, e)
        raise HTTPException(500, f"clear failed: {e}")
    return {"ok": True}
