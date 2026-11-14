#!/usr/bin/env python3
# R6 (issue #128): routing A/B — always-full-pipeline (hybrid RRF) vs routed
# (rule-based router: simple -> vector-only, standard/complex -> hybrid).
# Requires live BGE-M3 embeddings. Metrics: recall@5/@10, mrr@10 per arm +
# latency p50 per route level. Output JSON for docs/benchmarks/.

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fusion_rag.engine.query_router import QueryRouter
from fusion_rag.engine.recall_bench import (
    DEFAULT_TOPICS,
    SEED,
    _map_at_k,
    _mrr_at_k,
    _ndcg_at_k,
    _recall_at_k,
    build_chunk_records,
    generate_corpus,
)
from fusion_rag.engine.reranker import HybridSearch


def _metrics(ids: list[str], truth: set[str]) -> dict[str, float]:
    return {
        "recall@5": _recall_at_k(ids, truth, 5),
        "recall@10": _recall_at_k(ids, truth, 10),
        "mrr@10": _mrr_at_k(ids, truth, 10),
        "ndcg@10": _ndcg_at_k(ids, truth, 10),
        "map@10": _map_at_k(ids, truth, 10),
    }


async def _main(out_path: str) -> int:
    from fusion_rag.embed.client import EmbeddingClient
    from fusion_rag.store.vector_store import VectorStore

    docs, queries = generate_corpus(seed=SEED, topics=DEFAULT_TOPICS)
    records, doc_to_chunks = await build_chunk_records(docs)
    embed = EmbeddingClient()
    vectors = await embed.embed_batch([r["text"] for r in records])
    for r, v in zip(records, vectors):
        r["vector"] = v
    router = QueryRouter()
    arms = {"full": {}, "routed": {}}
    lat = {"full": {}, "routed": {}}
    with tempfile.TemporaryDirectory(prefix="fusion-rag-router-bench-") as workdir:
        vs = VectorStore(vector_path=os.path.join(workdir, "vectors"), dimension=len(vectors[0]), backend_type="local")
        vs.add_batch(records)
        hs = HybridSearch(vs)

        for query, doc_id in queries:
            truth = set(doc_to_chunks.get(doc_id, []))
            if not truth:
                continue
            decision = router.route(query)
            qv = await embed.embed(query)
            for arm in ("full", "routed"):
                t0 = time.time()
                if arm == "full" or decision.level != "simple":
                    chunks = await hs.search(qv, query, top_k=10, threshold=0.0)
                else:
                    chunks = vs.search(qv, top_k=10, threshold=0.0)
                ms = (time.time() - t0) * 1000
                m = _metrics([c["id"] for c in chunks], truth)
                lvl = "full" if arm == "full" else decision.level
                arms[arm].setdefault(lvl, []).append(m)
                lat[arm].setdefault(lvl, []).append(ms)
        vs.close()

    def _agg(store: dict, k: str) -> float:
        rows = [r for ms in store.values() for r in ms]
        return round(sum(r[k] for r in rows) / len(rows), 4) if rows else 0.0

    def _p50(vals: list[float]) -> float:
        s = sorted(vals)
        return round(s[len(s) // 2], 1) if s else 0.0

    keys = ["recall@5", "recall@10", "mrr@10", "ndcg@10", "map@10"]
    summary = {
        "generated_at": os.environ.get("BENCH_DATE", "2026-11-14"),
        "queries": sum(len(v) for v in arms["full"].values()),
        "route_mix": {lvl: len(ms) for lvl, ms in arms["routed"].items()},
        "metrics_full": {k: _agg(arms["full"], k) for k in keys},
        "metrics_routed": {k: _agg(arms["routed"], k) for k in keys},
        "latency_ms_p50_by_level": {
            arm: {lvl: _p50(v) for lvl, v in lat[arm].items()} for arm in ("full", "routed")
        },
    }
    print(json.dumps(summary, indent=2))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(f"\nwritten to {out_path}", file=sys.stderr)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="routing A/B (issue #128)")
    parser.add_argument(
        "--out",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "docs",
            "benchmarks",
            "router-ablation.json",
        ),
    )
    args = parser.parse_args()
    return asyncio.run(_main(args.out))


if __name__ == "__main__":
    sys.exit(main())
