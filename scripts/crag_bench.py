#!/usr/bin/env python3
# R5 (issue #127): CRAG retry-loop A/B — single-shot vs bounded grade→rewrite→
# re-retrieve loop on the deterministic labeled corpus. Requires live BGE-M3
# embeddings + a live fusion-mlx chat model (grader + rewriter). Metrics:
# recall@5/@10, mrr@10 per arm + per-round latency. Output JSON for
# docs/benchmarks/.

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


async def _main(max_rounds: int, min_relevant: int, out_path: str) -> int:
    from fusion_rag.embed.client import EmbeddingClient
    from fusion_rag.engine.crag import CorrectiveRAG, RetrievalGrader
    from fusion_rag.engine.query_rewriter import QueryRewriter
    from fusion_rag.store.vector_store import VectorStore

    docs, queries = generate_corpus(seed=SEED, topics=DEFAULT_TOPICS)
    records, doc_to_chunks = await build_chunk_records(docs)
    embed = EmbeddingClient()
    vectors = await embed.embed_batch([r["text"] for r in records])
    for r, v in zip(records, vectors):
        r["vector"] = v
    with tempfile.TemporaryDirectory(prefix="fusion-rag-crag-bench-") as workdir:
        vs = VectorStore(vector_path=os.path.join(workdir, "vectors"), dimension=len(vectors[0]), backend_type="local")
        vs.add_batch(records)
        hs = HybridSearch(vs)

        async def retrieve_fn(q: str) -> list[dict]:
            qv = await embed.embed(q)
            return await hs.search(qv, q, top_k=10, threshold=0.0)

        grader = RetrievalGrader(mlx_url=embed.base_url, api_key=embed.api_key)
        rewriter = QueryRewriter(mlx_url=embed.base_url, enabled=True, api_key=embed.api_key)
        crag = CorrectiveRAG(grader=grader, rewriter=rewriter, max_rounds=max_rounds, min_relevant=min_relevant)

        arms = {"single_shot": [], "crag": []}
        lat = {"single_shot": [], "crag": []}
        retried = 0
        for qi, (query, doc_id) in enumerate(queries):
            truth = set(doc_to_chunks.get(doc_id, []))
            if not truth:
                continue
            for arm in ("single_shot", "crag"):
                t0 = time.time()
                if arm == "single_shot":
                    chunks = await retrieve_fn(query)
                    ids = [c["id"] for c in chunks]
                else:
                    chunks, trace = await crag.run(query, retrieve_fn)
                    ids = [c["id"] for c in chunks]
                    if any(t.get("action") == "retried" for t in trace):
                        retried += 1
                lat[arm].append((time.time() - t0) * 1000)
                arms[arm].append(_metrics(ids, truth))
            if (qi + 1) % 10 == 0:
                print(f"{qi + 1}/{len(queries)} queries done", file=sys.stderr)

        vs.close()

    def _avg(rows: list[dict], k: str) -> float:
        return round(sum(r[k] for r in rows) / len(rows), 4) if rows else 0.0

    def _p50(vals: list[float]) -> float:
        s = sorted(vals)
        return round(s[len(s) // 2], 1) if s else 0.0

    summary = {
        "generated_at": os.environ.get("BENCH_DATE", "2026-11-14"),
        "queries": len(arms["single_shot"]),
        "crag_rounds": max_rounds,
        "min_relevant": min_relevant,
        "crag_retry_rate": round(retried / max(len(arms["single_shot"]), 1), 4),
        "single_shot": {k: _avg(arms["single_shot"], k) for k in arms["single_shot"][0]},
        "crag": {k: _avg(arms["crag"], k) for k in arms["crag"][0]},
        "latency_ms_p50": {k: _p50(v) for k, v in lat.items()},
    }
    print(json.dumps(summary, indent=2))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(f"\nwritten to {out_path}", file=sys.stderr)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="CRAG retry-loop A/B (issue #127)")
    parser.add_argument("--rounds", type=int, default=2, help="max CRAG retry rounds")
    parser.add_argument("--min-relevant", type=int, default=1, help="relevant chunks required to skip a retry")
    parser.add_argument(
        "--out",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "docs",
            "benchmarks",
            "crag-ablation.json",
        ),
    )
    args = parser.parse_args()
    return asyncio.run(_main(args.rounds, args.min_relevant, args.out))


if __name__ == "__main__":
    sys.exit(main())
