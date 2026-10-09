from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool

from .._validators import validate_identifier
from ..bnup.corpus import EDITION, BnupCorpus
from .app_state import get_embed_client, get_kb_manager
from .auth import verify_api_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/bnup", tags=["bnup"])

DEFAULT_KB_ID = "bnup-default"
KG_SIDECAR = "bnup_corpus.json"
UPSTREAM_MISCONCEPTIONS_NOTE = "upstream fusion-k12-teacher #21 common_misconceptions field"


def _bnup_kb(kb_id: str = DEFAULT_KB_ID):
    try:
        validate_identifier(kb_id, field="kb_id")
    except ValueError:
        raise HTTPException(400, f"Invalid kb_id: {kb_id}")
    mgr = get_kb_manager()
    try:
        return mgr.get(kb_id)
    except KeyError:
        kb = mgr.create(
            name="BNUP Math Corpus",
            description="北师大版小学数学教材语料 (issue #74)",
            chunk_strategy="semantic",
            kb_id=kb_id,
        )
        logger.info("created BNUP KB id=%s", kb_id)
        return kb


def _kb_storage_path(kb_id: str) -> str:
    kb = _bnup_kb(kb_id)
    return kb.vector_path.rsplit("/vectors", 1)[0] if "/vectors" in kb.vector_path else kb.vector_path


def _sidecar_path(kb_id: str) -> Path:
    return Path(_kb_storage_path(kb_id)) / KG_SIDECAR


def _write_sidecar(kb_id: str, corpus: BnupCorpus) -> dict[str, Any]:
    kg = corpus.knowledge_graph()
    stats = corpus.stats()
    lessons = corpus.lessons()
    misconceptions_by_kp: dict[str, list[Any]] = {}
    for d in lessons:
        m = d["metadata"]
        for kp in m["knowledge_point_ids"]:
            misconceptions_by_kp.setdefault(kp, [])
            for item in m.get("common_misconceptions", []):
                if item not in misconceptions_by_kp[kp]:
                    misconceptions_by_kp[kp].append(item)
    payload = {
        "edition": EDITION,
        "knowledge_graph": kg,
        "stats": stats,
        "misconceptions_by_kp": misconceptions_by_kp,
        "lesson_count": len(lessons),
    }
    p = _sidecar_path(kb_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    logger.info("wrote BNUP sidecar %s (%d lessons)", p, len(lessons))
    return payload


def _read_sidecar(kb_id: str) -> dict[str, Any] | None:
    p = _sidecar_path(kb_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("BNUP sidecar read failed %s: %s", p, e)
        return None


def _ensure_ingested(kb_id: str) -> dict[str, Any]:
    payload = _read_sidecar(kb_id)
    if payload is None:
        raise HTTPException(404, f"BNUP corpus not ingested on KB '{kb_id}'. POST /api/v1/bnup/ingest first.")
    return payload


@router.post("/ingest")
async def ingest(
    data: dict[str, Any],
    kb_id: str = DEFAULT_KB_ID,
    _auth: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    if not isinstance(data, dict) or "grades" not in data:
        raise HTTPException(400, "corpus JSON with 'grades' object is required")
    try:
        corpus = BnupCorpus(data)
    except ValueError as e:
        raise HTTPException(400, f"invalid corpus: {e}")
    lessons = corpus.lessons()
    if not lessons:
        raise HTTPException(400, "corpus parsed to zero lessons — check 'grades' structure")

    kb = _bnup_kb(kb_id)
    from .app_state import get_meta_store, get_or_create_vec_store

    backend_type = os.environ.get("FUSION_RAG_STORE_BACKEND", "local")
    vec_store = get_or_create_vec_store(kb.vector_path, backend_type)
    meta_store = get_meta_store(kb.metadata_path)
    embed = get_embed_client()

    texts = [d["content"] for d in lessons]
    vectors = await embed.embed_batch(texts)
    if not vectors or len(vectors) != len(lessons):
        raise HTTPException(502, "embedding batch failed (count mismatch)")

    records = []
    for d, vec in zip(lessons, vectors):
        chunk_id = f"{d['doc_id']}_0"
        records.append(
            {
                "id": chunk_id,
                "vector": vec,
                "text": d["content"],
                "doc_path": d["doc_path"],
                "doc_name": d["doc_name"],
                "doc_type": d["doc_type"],
                "chunk_index": 0,
                "metadata": d["metadata"],
                "context": "",
            }
        )
    for d in lessons:
        try:
            await run_in_threadpool(vec_store.delete_by_doc, d["doc_path"])
        except Exception as e:
            logger.debug("pre-ingest delete %s: %s", d["doc_path"], e)
    try:
        await run_in_threadpool(vec_store.add_batch, records)
    except Exception as e:
        logger.error("BNUP ingest add_batch failed: %s", e)
        raise HTTPException(500, f"index write failed: {e}")
    for d in lessons:
        doc_id = d["doc_id"]
        try:
            await run_in_threadpool(
                meta_store.add_document,
                doc_id,
                d["doc_path"],
                d["doc_name"],
                d["doc_type"],
                len(d["content"]),
                metadata=d["metadata"],
            )
            await run_in_threadpool(
                meta_store.add_chunks,
                [
                    {
                        "chunk_id": f"{doc_id}_0",
                        "doc_id": doc_id,
                        "doc_path": d["doc_path"],
                        "chunk_index": 0,
                        "text": d["content"],
                        "tokens": 0,
                        "metadata": d["metadata"],
                    }
                ],
            )
        except Exception as e:
            logger.error("BNUP metadata write failed for %s: %s", doc_id, e)

    payload = _write_sidecar(kb_id, corpus)
    return {
        "status": "ok",
        "kb_id": kb_id,
        "edition": EDITION,
        "lessons_ingested": len(lessons),
        "knowledge_points": payload["knowledge_graph"]["knowledge_point_count"],
    }


@router.get("/retrieve")
async def retrieve(
    knowledge_point: str = "",
    grade: int | None = None,
    limit: int = 10,
    kb_id: str = DEFAULT_KB_ID,
) -> list[dict[str, Any]]:
    from .routes import _apply_search_filters, _get_embed_client, _get_vector_store

    if limit < 1 or limit > 100:
        raise HTTPException(400, "limit must be 1..100")
    payload = _ensure_ingested(kb_id)
    embed = _get_embed_client()
    vec_store = _get_vector_store(kb_id)

    kp_index = payload["knowledge_graph"].get("knowledge_points", {})
    if knowledge_point and knowledge_point in kp_index:
        info = kp_index[knowledge_point]
        query = f"{info.get('title','')} {info.get('topic','')}".strip() or knowledge_point
    elif grade is not None:
        query = f"{grade}年级 数学"
    else:
        query = "数学"
    if knowledge_point and knowledge_point not in kp_index:
        query = knowledge_point

    query_vector = await embed.embed(query)
    if not query_vector or all(v == 0.0 for v in query_vector):
        raise HTTPException(500, "embedding failed")

    meta_filter: dict[str, Any] = {"edition": EDITION}
    if grade is not None:
        meta_filter["grade"] = grade

    fetch_k = max(limit * 4, limit)
    results = await run_in_threadpool(vec_store.search, query_vector, top_k=fetch_k, threshold=0.0)
    results = await run_in_threadpool(_apply_search_filters, results, None, meta_filter)
    if knowledge_point:
        results = [
            r for r in results if knowledge_point in (r.get("metadata", {}) or {}).get("knowledge_point_ids", [])
        ]

    if not results and knowledge_point:
        from .app_state import get_meta_store

        meta_store = get_meta_store(_bnup_kb(kb_id).metadata_path)
        docs = await run_in_threadpool(meta_store.list_documents, 2000, 0)
        for doc in docs:
            dm = doc.get("metadata", {}) or {}
            if dm.get("edition") == EDITION and knowledge_point in (dm.get("knowledge_point_ids") or []):
                results.append(
                    {
                        "id": doc.get("id", ""),
                        "doc_path": doc.get("file_path", ""),
                        "doc_name": doc.get("file_name", ""),
                        "text": "",
                        "score": 0.0,
                        "metadata": dm,
                    }
                )
                if len(results) >= limit:
                    break
    return results[:limit]


@router.get("/knowledge_graph")
async def knowledge_graph(kb_id: str = DEFAULT_KB_ID) -> dict[str, Any]:
    payload = _ensure_ingested(kb_id)
    return payload["knowledge_graph"]


@router.get("/misconceptions")
async def misconceptions(knowledge_point: str = "", kb_id: str = DEFAULT_KB_ID) -> dict[str, Any]:
    payload = _ensure_ingested(kb_id)
    by_kp: dict[str, list[Any]] = payload.get("misconceptions_by_kp", {})
    if knowledge_point:
        items = by_kp.get(knowledge_point, [])
        return {
            "knowledge_point": knowledge_point,
            "misconceptions": items,
            "count": len(items),
            "dependency": UPSTREAM_MISCONCEPTIONS_NOTE,
        }
    total = sum(len(v) for v in by_kp.values())
    return {
        "knowledge_point": "",
        "misconceptions_by_kp": by_kp,
        "total": total,
        "dependency": UPSTREAM_MISCONCEPTIONS_NOTE,
    }


@router.get("/stats")
async def stats(kb_id: str = DEFAULT_KB_ID) -> dict[str, Any]:
    payload = _ensure_ingested(kb_id)
    return payload["stats"]
