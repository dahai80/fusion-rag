#!/usr/bin/env python3
# R4 (issue #126): contextual-BM25 A/B — no context vs doc-level context prefix
# prepended to the BM25 text. Requires live BGE-M3 embeddings (dense channel is
# identical in both arms; only the BM25 side changes). Output: per-arm
# recall@5/@10, mrr@10, ndcg@10, map@10, written as JSON for docs/benchmarks/.

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
    run_contextual_bm25_ablation,
)


async def _main(out_path: str) -> int:
    from fusion_rag.embed.client import EmbeddingClient

    docs, queries = generate_corpus(seed=SEED, topics=DEFAULT_TOPICS)
    embed = EmbeddingClient()
    with tempfile.TemporaryDirectory(prefix="fusion-rag-ctx-bm25-") as workdir:
        metrics = await run_contextual_bm25_ablation(embed, docs, queries, workdir=workdir)
    payload = {"generated_at": os.environ.get("ABLATION_DATE", "2026-11-14"), **metrics}
    print(json.dumps(payload, indent=2))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"\nwritten to {out_path}", file=sys.stderr)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="contextual BM25 A/B (issue #126)")
    parser.add_argument(
        "--out",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "docs",
            "benchmarks",
            "contextual-bm25-ablation.json",
        ),
    )
    args = parser.parse_args()
    return asyncio.run(_main(args.out))


if __name__ == "__main__":
    sys.exit(main())
