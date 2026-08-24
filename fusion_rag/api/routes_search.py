from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from ..engine.llm_errors import LLMUnavailable
from ..engine.query_rewriter import QueryRewriter
from ..engine.reranker import HybridSearch

logger = logging.getLogger(__name__)

router = APIRouter(tags=["search"])


def _audit_and_trajectory(
    kb_id: str,
    query: str,
    caller: str,
    results: list[dict],
    latency_ms: float,
    metadata: dict | None = None,
) -> None:
    from ..engine.audit_logger import AuditLogger
    from ..engine.trajectory_writer import TrajectoryWriter
    from .routes import _get_kb_storage_path

    top_sources = [
        {"doc_name": r.get("doc_name", ""), "score": r.get("score", 0.0), "id": r.get("id", "")} for r in results[:5]
    ]
    try:
        al = AuditLogger(f"{_get_kb_storage_path(kb_id)}/audit.db")
        al.log_search(kb_id, query, caller, len(results), top_sources, latency_ms, metadata)
    except Exception as e:
        logger.warning("audit log_search failed: %s", e)
    try:
        tw = TrajectoryWriter()
        tw.write(kb_id, query, caller, len(results), top_sources, latency_ms, metadata)
    except Exception as e:
        logger.warning("trajectory write failed: %s", e)


async def _audit_and_trajectory_async(
    kb_id: str,
    query: str,
    caller: str,
    results: list[dict],
    latency_ms: float,
    metadata: dict | None = None,
) -> None:
    # P8/硬伤8: audit/trajectory do sync sqlite writes; running them inline in
    # the async handler blocks the event loop for the write duration. Push the
    # fire-and-forget logging off the loop thread.
    await run_in_threadpool(
        _audit_and_trajectory, kb_id, query, caller, results, latency_ms, metadata
    )


@router.post("/bases/{kb_id}/search")
async def search(kb_id: str, data: dict[str, Any]) -> list[dict[str, Any]]:
    from .routes import _apply_search_filters, _do_rerank, _get_base, _get_embed_client, _get_vector_store

    kb = _get_base(kb_id)
    query = data.get("query", "")
    if not query:
        raise HTTPException(400, "query is required")

    top_k = data.get("top_k", kb.config.max_results)
    threshold = data.get("threshold", kb.config.similarity_threshold)
    folder_prefix = data.get("folder_prefix")
    use_hybrid = data.get("hybrid", False)
    use_rerank = data.get("rerank", False)
    hybrid_alpha = data.get("hybrid_alpha", 0.7)
    hybrid_method = data.get("hybrid_method", "rrf")
    metadata_filter = data.get("filter")
    template = data.get("template")

    if template:
        from ..engine.search_template import SearchTemplateManager
        from .routes import _get_kb_storage_path

        # P8/硬伤8: SearchTemplateManager opens + queries sqlite synchronously.
        tpl_mgr = SearchTemplateManager(f"{_get_kb_storage_path(kb_id)}/templates.db")
        tpl = await run_in_threadpool(tpl_mgr.get_template, kb_id, template)
        if tpl:
            top_k = tpl.get("top_k", top_k)
            threshold = tpl.get("threshold", threshold)
            hybrid_alpha = tpl.get("alpha", hybrid_alpha)
            use_rerank = tpl.get("rerank", use_rerank)
            rewrite_mode = tpl.get("rewrite_mode", data.get("rewrite_mode"))
            doc_type_filter = tpl.get("doc_type_filter", [])
        else:
            rewrite_mode = data.get("rewrite_mode")
            doc_type_filter = []
    else:
        rewrite_mode = data.get("rewrite_mode")
        doc_type_filter = []

    embed = _get_embed_client()
    vec_store = _get_vector_store(kb_id)
    _start = time.time()

    if rewrite_mode:
        rewriter = QueryRewriter(enabled=True)
        try:
            rewritten = await rewriter.rewrite(query, mode=rewrite_mode)
        except LLMUnavailable as e:
            # L1/L15: rewrite is an enhancement — fall back to the original
            # query (logged), do NOT 503. The user still gets retrieval; they
            # just lose the query-expansion lift.
            logger.warning("search rewrite LLM unavailable, using original query: %s", e)
            rewritten = query
        if isinstance(rewritten, list):
            # L15: the rewrite-list path previously ran plain vec_store.search
            # per sub-query, silently dropping hybrid when the client asked for
            # hybrid=True + rewrite_mode=... (behavior fork vs the non-rewrite
            # path). Honor use_hybrid per sub-query, and over-fetch so the
            # post-fusion _apply_search_filters (L10) doesn't truncate below
            # top_k when many rows are filtered out.
            fetch_k = max(top_k * 4, top_k)
            all_results = []
            for q in rewritten:
                qv = await embed.embed(q)
                if not qv or all(v == 0.0 for v in qv):
                    continue
                if use_hybrid:
                    hs = HybridSearch(vec_store, alpha=hybrid_alpha, method=hybrid_method)
                    sub_filters = {}
                    if folder_prefix:
                        sub_filters["folder_prefix"] = folder_prefix
                    sub = await hs.search(
                        qv,
                        q,
                        top_k=fetch_k,
                        threshold=threshold,
                        filters=sub_filters if sub_filters else None,
                    )
                else:
                    # P8/硬伤8: vec_store.search is sync LanceDB+BM25; push off-loop.
                    sub = await run_in_threadpool(vec_store.search, qv, top_k=fetch_k, threshold=threshold)
                    sub = await run_in_threadpool(_apply_search_filters, sub, folder_prefix, metadata_filter)
                all_results.extend(sub)
            seen = {}
            for r in all_results:
                rid = r.get("id", "")
                if rid not in seen or r.get("score", 0) > seen[rid].get("score", 0):
                    seen[rid] = r
            results = sorted(seen.values(), key=lambda x: x.get("score", 0), reverse=True)[:top_k]
            if doc_type_filter:
                results = [r for r in results if r.get("doc_type", "") in doc_type_filter]
            if use_rerank:
                results = await _do_rerank(query, results, top_k)
            await _audit_and_trajectory_async(kb_id, query, "search", results, (time.time() - _start) * 1000)
            return results
        query = rewritten

    query_vector = await embed.embed(query)
    if not query_vector or all(v == 0.0 for v in query_vector):
        raise HTTPException(500, "Embedding failed")

    if use_hybrid:
        hs = HybridSearch(vec_store, alpha=hybrid_alpha, method=hybrid_method)
        filters = {}
        if folder_prefix:
            filters["folder_prefix"] = folder_prefix
        results = await hs.search(
            query_vector,
            query,
            top_k=top_k,
            threshold=threshold,
            filters=filters if filters else None,
        )
    else:
        # L10: fetch wider than top_k so _apply_search_filters (folder_prefix /
        # metadata_filter, applied client-side AFTER the top_k truncation) does
        # not silently shrink the result set below top_k when many rows are
        # filtered out. Matching rows may sit just past top_k.
        fetch_k = top_k * 4 if (folder_prefix or metadata_filter) else top_k
        # P8/硬伤8: sync LanceDB search + per-row json.loads filter block the
        # event loop if run inline in an async handler.
        results = await run_in_threadpool(vec_store.search, query_vector, top_k=fetch_k, threshold=threshold)
        results = await run_in_threadpool(_apply_search_filters, results, folder_prefix, metadata_filter)
        results = results[:top_k]

    if doc_type_filter:
        results = [r for r in results if r.get("doc_type", "") in doc_type_filter]

    if use_rerank:
        results = await _do_rerank(query, results, top_k)

    await _audit_and_trajectory_async(kb_id, query, "search", results, (time.time() - _start) * 1000)
    return results


@router.post("/bases/{kb_id}/ask")
async def ask(kb_id: str, data: dict[str, Any]) -> dict[str, Any]:
    from .routes import (
        _apply_search_filters,
        _do_rerank,
        _generate_answer,
        _get_base,
        _get_embed_client,
        _get_vector_store,
    )

    kb = _get_base(kb_id)
    question = data.get("question", "")
    if not question:
        raise HTTPException(400, "question is required")

    top_k = data.get("top_k", kb.config.max_results)
    model = data.get("model", "qwen3.5-9b")
    max_tokens = data.get("max_tokens", 4096)
    temperature = data.get("temperature", 0.3)
    system_prompt = data.get("system_prompt", "")
    use_hybrid = data.get("hybrid", False)
    use_rerank = data.get("rerank", False)
    folder_prefix = data.get("folder_prefix")
    metadata_filter = data.get("filter")

    embed = _get_embed_client()
    vec_store = _get_vector_store(kb_id)
    _start = time.time()

    rewrite_mode = data.get("rewrite_mode")
    history = data.get("history")
    if rewrite_mode or history:
        rewriter = QueryRewriter(enabled=True)
        mode = rewrite_mode or ("condense" if history else "hyde")
        try:
            rewritten = await rewriter.rewrite(question, history=history, mode=mode)
            if isinstance(rewritten, str) and rewritten:
                question = rewritten
        except LLMUnavailable as e:
            # L1: rewrite is an enhancement; condense failure on multi-turn
            # means we lose history context — keep original question, log it.
            logger.warning("ask rewrite LLM unavailable, using original question: %s", e)

    query_vector = await embed.embed(question)
    if not query_vector or all(v == 0.0 for v in query_vector):
        raise HTTPException(500, "Embedding failed")

    if use_hybrid:
        hs = HybridSearch(vec_store)
        filters = {}
        if folder_prefix:
            filters["folder_prefix"] = folder_prefix
        chunks = await hs.search(
            query_vector,
            question,
            top_k=top_k,
            threshold=kb.config.similarity_threshold,
            filters=filters if filters else None,
        )
    else:
        fetch_k = top_k * 4 if (folder_prefix or metadata_filter) else top_k
        # P8/硬伤8: sync LanceDB search + filter block the event loop.
        chunks = await run_in_threadpool(
            vec_store.search, query_vector, top_k=fetch_k, threshold=kb.config.similarity_threshold
        )
        chunks = await run_in_threadpool(_apply_search_filters, chunks, folder_prefix, metadata_filter)
        chunks = chunks[:top_k]

    if not chunks:
        await _audit_and_trajectory_async(kb_id, question, "ask", [], (time.time() - _start) * 1000)
        return {"answer": "No relevant documents found.", "sources": []}

    if use_rerank:
        chunks = await _do_rerank(question, chunks, top_k)

    context = "\n\n".join(f"[{c['doc_name']}] {c['text'][:2000]}" for c in chunks)

    try:
        answer = await _generate_answer(
            question,
            context,
            chunks,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
        )
    except LLMUnavailable as e:
        # L9: fusion-mlx down → no fabricated 200 answer. Surface as 502 so
        # the caller knows generation failed, not "answered with no info".
        raise HTTPException(502, "Answer generation failed (upstream LLM unavailable)") from e
    await _audit_and_trajectory_async(
        kb_id,
        question,
        "ask",
        chunks,
        (time.time() - _start) * 1000,
        {"model": model, "has_answer": bool(answer.get("answer"))},
    )
    return answer
