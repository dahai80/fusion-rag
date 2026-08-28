from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from .._validators import validate_identifier
from .access import require_kb_action
from .app_state import get_kb_manager
from .auth import verify_api_key

logger = logging.getLogger(__name__)

router = APIRouter(tags=["kb-crud"])

# 硬伤1: read kb_manager from app.state via contextvar, not a module global.
_get_kb_manager = get_kb_manager


def _get_base(kb_id: str):
    # F12: validate kb_id before any path construction (path-traversal guard).
    try:
        validate_identifier(kb_id, field="kb_id")
    except ValueError:
        raise HTTPException(400, f"Invalid kb_id: {kb_id}")
    try:
        return _get_kb_manager().get(kb_id)
    except KeyError:
        raise HTTPException(404, f"Knowledge base '{kb_id}' not found")


@router.get("/bases", dependencies=[Depends(verify_api_key)])
async def list_knowledge_bases() -> list[dict[str, Any]]:
    return _get_kb_manager().list()


@router.post("/bases", dependencies=[Depends(verify_api_key)])
async def create_knowledge_base(data: dict[str, Any]) -> dict[str, Any]:
    # No {kb_id} path param yet -> cannot use require_kb_action (no KB to check
    # against). Plain auth gate; the new KB is open until an admin writes rules.
    name = data.get("name", "")
    kb_id = data.get("kb_id", "")
    if not name and not kb_id:
        raise HTTPException(400, "name or kb_id is required")
    # F12: a caller-supplied kb_id must match the identifier charset or it would
    # be unreadable later (and could carry path separators). Reject at the gate.
    if kb_id:
        try:
            validate_identifier(kb_id, field="kb_id")
        except ValueError:
            raise HTTPException(400, f"Invalid kb_id: {kb_id}")
    if not name:
        name = kb_id
    description = data.get("description", "")
    chunk_strategy = data.get("chunk_strategy", "semantic")
    embedding_model = data.get("embedding_model", "BGE-M3")
    existing = kb_id and kb_id in _get_kb_manager()._bases
    kb = _get_kb_manager().create(
        name=name,
        description=description,
        chunk_strategy=chunk_strategy,
        embedding_model=embedding_model,
        kb_id=kb_id,
    )
    status = "exists" if existing else "created"
    return {"id": kb.id, "name": kb.config.name, "status": status}


@router.get("/bases/{kb_id}", dependencies=[Depends(require_kb_action("read"))])
async def get_knowledge_base(kb_id: str) -> dict[str, Any]:
    return _get_base(kb_id).to_dict()


@router.delete("/bases/{kb_id}", dependencies=[Depends(require_kb_action("delete"))])
async def delete_knowledge_base(kb_id: str) -> dict[str, str]:
    # F12: validate kb_id before delete (path-traversal guard, same charset rule).
    try:
        validate_identifier(kb_id, field="kb_id")
    except ValueError:
        raise HTTPException(400, f"Invalid kb_id: {kb_id}")
    if _get_kb_manager().delete(kb_id):
        return {"id": kb_id, "status": "deleted"}
    raise HTTPException(404, f"Knowledge base '{kb_id}' not found")


@router.get("/bases/{kb_id}/stats", dependencies=[Depends(require_kb_action("read"))])
async def kb_stats(kb_id: str) -> dict[str, Any]:
    from .routes import _get_meta_store, _get_vector_store

    kb = _get_base(kb_id)
    meta_store = _get_meta_store(kb_id)
    vec_store = _get_vector_store(kb_id)
    return {
        "id": kb.id,
        "name": kb.config.name,
        "documents": meta_store.doc_count(),
        "chunks": meta_store.chunk_count(),
        "vectors": vec_store.count(),
        "file_count": kb.file_count,
        "chunk_count": kb.chunk_count,
    }
