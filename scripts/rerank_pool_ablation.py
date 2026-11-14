#!/usr/bin/env python3
# R3 (issue #125): rerank candidate-pool A/B — 20 vs 50/100/150.
# Requires a live fusion-mlx with a cross-encoder rerank model
# (bge-reranker-v2-m3) and live BGE-M3 embeddings. Output: ablation table
# with recall@5/@10, mrr@10, ndcg@10, map@10 + rerank latency p50/p95 per
# pool size, written as JSON for docs/benchmarks/.

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fusion_rag.engine.recall_bench import (
    DEFAULT_TOPICS,
    SEED,
    generate_corpus,
    run_rerank_pool_ablation,
)


async def _main(pools: tuple[int, ...], out_path: str) -> int:
    from fusion_rag.embed.client import EmbeddingClient

    docs, queries = generate_corpus(seed=SEED, topics=DEFAULT_TOPICS)
    embed = EmbeddingClient()
    with tempfile.TemporaryDirectory(prefix="fusion-rag-pool-ablation-") as workdir:
        metrics = await run_rerank_pool_ablation(embed, docs, queries, workdir=workdir, pools=pools)
    payload = {"generated_at": os.environ.get("ABLATION_DATE", "2026-11-14"), **metrics}
    print(json.dumps(payload, indent=2))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"\nwritten to {out_path}", file=sys.stderr)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="rerank candidate-pool A/B (issue #125)")
    parser.add_argument("--pools", default="20,50,100,150", help="comma-separated pool sizes")
    parser.add_argument(
        "--out",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "docs",
            "benchmarks",
            "rerank-pool-ablation.json",
        ),
    )
    args = parser.parse_args()
    pools = tuple(int(p) for p in args.pools.split(",") if p.strip())
    return asyncio.run(_main(pools, args.out))


if __name__ == "__main__":
    sys.exit(main())
