from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool

from .access import require_kb_action

logger = logging.getLogger(__name__)

router = APIRouter(tags=["graph"])

# Issue #79: GraphRAG was dead code advertised as a feature — now wired.
# Entity extraction runs one LLM call per chunk (semaphore-bounded inside
# build_graph), so a build is capped to keep a single request bounded.
DEFAULT_MAX_CHUNKS = 200
HARD_MAX_CHUNKS = 1000
DEFAULT_MODEL = "qwen3.5-9b"


def _graph_rag(kb_id: str, model: str = DEFAULT_MODEL):
    from .routes import _get_embed_client, _get_kb_storage_path

    embed = _get_embed_client()
    db_path = os.path.join(_get_kb_storage_path(kb_id), "graph.db")
    from ..engine.graph_rag import GraphRAG

    return GraphRAG(db_path=db_path, mlx_url=embed.base_url, model=model)


@router.post("/bases/{kb_id}/graph/build", dependencies=[Depends(require_kb_action("write"))])
async def build_graph(kb_id: str, data: dict[str, Any]) -> dict[str, Any]:
    from .routes import _get_base, _get_meta_store

    _get_base(kb_id)
    max_chunks = min(int(data.get("max_chunks", DEFAULT_MAX_CHUNKS)), HARD_MAX_CHUNKS)
    if max_chunks < 1:
        raise HTTPException(400, "max_chunks must be >= 1")
    model = str(data.get("model", DEFAULT_MODEL))

    meta_store = _get_meta_store(kb_id)
    docs = await run_in_threadpool(meta_store.list_documents, 2000, 0)
    chunks: list[dict[str, Any]] = []
    for doc in docs:
        doc_chunks = await run_in_threadpool(meta_store.get_chunks_by_doc, doc.get("id", ""))
        for c in doc_chunks:
            # chunks table PK column is `id` (the chunk id, {doc_id}_{index})
            chunks.append({"id": c.get("id", ""), "text": c.get("text", "")})
        if len(chunks) >= max_chunks:
            break
    chunks = chunks[:max_chunks]
    if not chunks:
        raise HTTPException(400, f"KB '{kb_id}' has no chunks to extract entities from — ingest documents first")

    graph = _graph_rag(kb_id, model=model)
    try:
        counts = await graph.build_graph(chunks, kb_id=kb_id)
    except Exception as e:
        logger.error("graph build failed for kb %s: %s", kb_id, e)
        raise HTTPException(502, f"graph build failed: {e}")
    return {
        "status": "ok",
        "kb_id": kb_id,
        "chunks_processed": len(chunks),
        "entities": counts["entities"],
        "relations": counts["relations"],
    }


@router.get("/bases/{kb_id}/graph/stats")
async def graph_stats(kb_id: str) -> dict[str, Any]:
    from .routes import _get_base

    _get_base(kb_id)
    graph = _graph_rag(kb_id)
    entities = await run_in_threadpool(graph.get_entity_count, kb_id)
    relations = await run_in_threadpool(graph.get_relation_count, kb_id)
    return {"kb_id": kb_id, "entities": entities, "relations": relations}


@router.post("/bases/{kb_id}/graph/search", dependencies=[Depends(require_kb_action("search"))])
async def graph_search(kb_id: str, data: dict[str, Any]) -> list[dict[str, Any]]:
    from .routes import _get_base, _get_meta_store, _normalize_evidence

    _get_base(kb_id)
    query = str(data.get("query", "")).strip()
    if not query:
        raise HTTPException(400, "query is required")
    max_hops = min(int(data.get("max_hops", 2)), 4)

    graph = _graph_rag(kb_id)
    hits = await run_in_threadpool(graph.search, query, kb_id, max_hops)
    if not hits:
        return []

    # Resolve hit chunk ids back to chunk text via the metadata store
    # (chunk_id format is {doc_id}_{index}).
    meta_store = _get_meta_store(kb_id)
    results: list[dict[str, Any]] = []
    for hit in hits:
        chunk_id = hit.get("chunk_id", "")
        doc_id = chunk_id.rsplit("_", 1)[0] if "_" in chunk_id else chunk_id
        text = ""
        metadata: dict[str, Any] = {}
        try:
            doc_chunks = await run_in_threadpool(meta_store.get_chunks_by_doc, doc_id)
            for c in doc_chunks:
                if c.get("id") == chunk_id:
                    text = c.get("text", "")
                    metadata = c.get("metadata", {}) or {}
                    break
        except Exception as e:
            logger.debug("graph search chunk resolve failed %s: %s", chunk_id, e)
        results.append(
            {
                "id": chunk_id,
                "doc_id": doc_id,
                "text": text,
                "score": 0.0,
                "metadata": metadata,
                "entity_count": hit.get("entity_count", 0),
            }
        )
    return _normalize_evidence(results)
