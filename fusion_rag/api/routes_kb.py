from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from ..engine.knowledge_base import KnowledgeBaseManager
from .auth import verify_api_key

logger = logging.getLogger(__name__)

router = APIRouter(tags=["kb-crud"])

_kb_manager: KnowledgeBaseManager | None = None


def set_kb_manager(kb_manager: KnowledgeBaseManager, embed_client: Any = None) -> None:
    global _kb_manager
    _kb_manager = kb_manager


set_kb_context = set_kb_manager


def _get_kb_manager() -> KnowledgeBaseManager:
    if _kb_manager is None:
        raise HTTPException(503, "Knowledge base manager not initialized")
    return _kb_manager


def _get_base(kb_id: str):
    try:
        return _get_kb_manager().get(kb_id)
    except KeyError:
        raise HTTPException(404, f"Knowledge base '{kb_id}' not found")


@router.get("/bases")
async def list_knowledge_bases() -> list[dict[str, Any]]:
    return _get_kb_manager().list()


@router.post("/bases", dependencies=[Depends(verify_api_key)])
async def create_knowledge_base(data: dict[str, Any]) -> dict[str, Any]:
    name = data.get("name", "")
    kb_id = data.get("kb_id", "")
    if not name and not kb_id:
        raise HTTPException(400, "name or kb_id is required")
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


@router.get("/bases/{kb_id}")
async def get_knowledge_base(kb_id: str) -> dict[str, Any]:
    try:
        return _get_kb_manager().get(kb_id).to_dict()
    except KeyError:
        raise HTTPException(404, f"Knowledge base '{kb_id}' not found")


@router.delete("/bases/{kb_id}", dependencies=[Depends(verify_api_key)])
async def delete_knowledge_base(kb_id: str) -> dict[str, str]:
    if _get_kb_manager().delete(kb_id):
        return {"id": kb_id, "status": "deleted"}
    raise HTTPException(404, f"Knowledge base '{kb_id}' not found")


@router.get("/bases/{kb_id}/stats")
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
