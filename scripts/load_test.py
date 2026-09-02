"""Fusion-RAG load test — target-environment throughput + latency baseline.

No new deps (httpx only). Seeds a KB with N synthetic docs, then fires
concurrent /search (hybrid, needs embedding) + /search (keyword) requests and
collects RPS / p50 / p90 / p99 / error count. /ask (LLM) is optional and off
by default (slow, model-bound). Writes a JSON report to scripts/load_test_report.json.

Usage:
    python scripts/load_test.py [--concurrency 8] [--duration 30] [--docs 60]
                                [--ask] [--base-url http://127.0.0.1:11436]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx

logger = logging.getLogger("load_test")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ── helpers ──


def percentile(sorted_latencies: list[float], pct: float) -> float:
    if not sorted_latencies:
        return 0.0
    k = (len(sorted_latencies) - 1) * pct / 100.0
    f = int(k)
    c = min(f + 1, len(sorted_latencies) - 1)
    if f == c:
        return sorted_latencies[f]
    return sorted_latencies[f] + (sorted_latencies[c] - sorted_latencies[f]) * (k - f)


def make_doc(i: int) -> dict:
    topics = ["machine learning", "vector database", "retrieval augmented generation",
              "knowledge graph", "embedding model", "semantic search", "chunking strategy",
              "reranking", "hybrid search", "BM25"]
    topic = topics[i % len(topics)]
    title = f"doc_{i:03d}_{topic.replace(' ', '_')}"
    body = (
        f"This document discusses {topic} in the context of the fusion ecosystem. "
        f"Section {i} covers core concepts of {topic}, including practical examples "
        f"and architectural tradeoffs for {topic} on Apple Silicon. The {topic} "
        f"pipeline integrates with the embedding service and the BM25 keyword index "
        f"to deliver hybrid retrieval. Document {i} is one of a synthetic set generated "
        f"for load testing the search endpoint under concurrent access."
    )
    return {"doc_name": f"{title}.md", "doc_type": "md", "content": body}


async def seed_kb(base_url: str, kb_id: str, n_docs: int, client: httpx.AsyncClient) -> None:
    import tempfile

    seed_dir = Path(tempfile.mkdtemp(prefix="rag-loadtest-"))
    r = await client.post(f"{base_url}/kb/bases", json={"name": "load-test-kb", "kb_id": kb_id})
    if r.status_code not in (200, 409):
        r.raise_for_status()
    logger.info("Seeding KB %s with %d docs into %s ...", kb_id, n_docs, seed_dir)
    t0 = time.perf_counter()
    ok = 0
    for i in range(n_docs):
        doc = make_doc(i)
        fp = seed_dir / doc["doc_name"]
        fp.write_text(doc["content"])
        rr = await client.post(f"{base_url}/kb/bases/{kb_id}/documents",
                               json={"file_path": str(fp), "doc_type": "md"})
        if rr.status_code == 200:
            ok += 1
        else:
            logger.warning("ingest doc %d failed: %s %s", i, rr.status_code, rr.text[:200])
    logger.info("Seed done in %.1fs — %d/%d ingested", time.perf_counter() - t0, ok, n_docs)
    # cleanup seed files (KB vectors persist in the store)
    import shutil

    shutil.rmtree(seed_dir, ignore_errors=True)


async def worker(name: str, base_url: str, kb_id: str, queries: list[str],
                 stop_at: float, client: httpx.AsyncClient,
                 results: list, mode: str, errors: list, ask_model: str | None):
    path = f"{base_url}/kb/bases/{kb_id}/search"
    ask_path = f"{base_url}/kb/bases/{kb_id}/ask"
    i = 0
    while time.perf_counter() < stop_at:
        q = queries[i % len(queries)]
        i += 1
        t0 = time.perf_counter()
        try:
            if mode == "ask":
                rr = await client.post(ask_path, json={"query": q, "top_k": 5},
                                       timeout=120.0)
            else:
                payload = {"query": q, "top_k": 10}
                if mode == "keyword":
                    payload["method"] = "keyword"
                rr = await client.post(path, json=payload, timeout=60.0)
            dt = (time.perf_counter() - t0) * 1000.0
            if rr.status_code != 200:
                errors.append(f"{mode}:{rr.status_code}:{rr.text[:120]}")
            results.append(dt)
        except Exception as e:
            dt = (time.perf_counter() - t0) * 1000.0
            errors.append(f"{mode}:exc:{type(e).__name__}:{str(e)[:120]}")
            results.append(dt)


async def run_phase(name: str, base_url: str, kb_id: str, concurrency: int,
                    duration: float, mode: str, client: httpx.AsyncClient,
                    ask_model: str | None = None) -> dict:
    queries = [
        "machine learning on Apple Silicon",
        "vector database retrieval",
        "hybrid search BM25 embedding",
        "reranking strategy",
        "knowledge graph integration",
        "embedding model BGE-M3",
        "semantic search chunking",
        "retrieval augmented generation pipeline",
    ]
    results: list[float] = []
    errors: list[str] = []
    stop_at = time.perf_counter() + duration
    workers = [
        asyncio.create_task(worker(f"{name}-{w}", base_url, kb_id, queries, stop_at,
                                   client, results, mode, errors, ask_model))
        for w in range(concurrency)
    ]
    await asyncio.gather(*workers)
    results.sort()
    n = len(results)
    elapsed = duration
    err_count = len(errors)
    report = {
        "phase": name,
        "mode": mode,
        "concurrency": concurrency,
        "duration_s": round(elapsed, 2),
        "requests": n,
        "rps": round(n / elapsed, 1) if elapsed else 0.0,
        "errors": err_count,
        "latency_ms": {
            "min": round(results[0], 2) if results else 0,
            "p50": round(percentile(results, 50), 2),
            "p90": round(percentile(results, 90), 2),
            "p99": round(percentile(results, 99), 2),
            "max": round(results[-1], 2) if results else 0,
        },
    }
    if err_count:
        report["error_samples"] = errors[:5]
    logger.info("[%s] rps=%.1f p50=%.1f p90=%.1f p99=%.1f errors=%d",
                name, report["rps"], report["latency_ms"]["p50"],
                report["latency_ms"]["p90"], report["latency_ms"]["p99"], err_count)
    return report


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=os.environ.get("LOADTEST_URL", "http://127.0.0.1:11436"))
    ap.add_argument("--kb-id", default="loadtest-kb")
    ap.add_argument("--docs", type=int, default=60)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--ask", action="store_true", help="also run /ask (LLM) phase")
    ap.add_argument("--skip-seed", action="store_true", help="reuse existing KB")
    ap.add_argument("--out", default=str(Path(__file__).parent / "load_test_report.json"))
    args = ap.parse_args()

    logger.info("Load test → %s  KB=%s  docs=%d  concurrency=%d  duration=%.0fs",
                args.base_url, args.kb_id, args.docs, args.concurrency, args.duration)

    async with httpx.AsyncClient(base_url=args.base_url, timeout=120.0) as client:
        # health gate
        h = await client.get("/ready")
        if h.status_code != 200:
            logger.error("/ready not 200: %s %s", h.status_code, h.text[:200])
            sys.exit(1)
        logger.info("/ready ok: %s", h.json().get("checks"))

        if not args.skip_seed:
            await seed_kb(args.base_url, args.kb_id, args.docs, client)

        phases = []
        # warmup (prime embedding cache for the query set)
        logger.info("Warmup: 8 hybrid searches to prime embedding cache...")
        for q in ["machine learning", "vector database", "reranking", "embedding model"]:
            await client.post(f"/kb/bases/{args.kb_id}/search",
                              json={"query": q, "top_k": 5}, timeout=60.0)

        phases.append(await run_phase("hybrid_search", args.base_url, args.kb_id,
                                      args.concurrency, args.duration, "hybrid", client))
        phases.append(await run_phase("keyword_search", args.base_url, args.kb_id,
                                      args.concurrency, args.duration, "keyword", client))
        if args.ask:
            phases.append(await run_phase("ask_llm", args.base_url, args.kb_id,
                                          max(2, args.concurrency // 4), args.duration,
                                          "ask", client, ask_model="qwen"))

    report = {
        "base_url": args.base_url,
        "kb_id": args.kb_id,
        "docs": args.docs,
        "concurrency": args.concurrency,
        "duration_s": args.duration,
        "timestamp": int(time.time()),
        "phases": phases,
    }
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    logger.info("Report → %s", args.out)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
