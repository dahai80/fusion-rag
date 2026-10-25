# Issue #79: recall benchmark CLI — ≥500-chunk mixed corpus through the REAL
# chunker, ≥50 ground-truth queries, recall@k + MRR, ablation BM25-only vs
# vector-only vs RRF fusion.
#
#   mock mode: deterministic hashed-bag embeddings — wiring/no-regression
#     evidence ONLY, never quality evidence. Output is labeled mode=mock.
#   real mode: live fusion-mlx BGE-M3 via EmbeddingClient (env FUSION_MLX_URL).
#     Quality evidence — write results to docs/benchmarks/ with --output.
#
# Cleans up its temp store dir after the run. Exit 0 iff the ablation contract
# (rrf >= max(single-mode) on recall@10) holds in mock mode; real mode reports
# the numbers and always exits 0 unless the run itself fails.
#
# Callers: manual + CI (tests/test_recall_benchmark.py drives the module).

import argparse
import asyncio
import json
import logging
import os
import shutil
import sys
import tempfile
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("recall_bench")


def _print_table(mode_label: str, metrics: dict) -> None:
    logger.info("=== Fusion-RAG recall benchmark (%s) ===", mode_label)
    logger.info(
        "corpus: %d chunks / %d docs, %d queries",
        metrics["corpus"]["chunks"],
        metrics["corpus"]["docs"],
        metrics["corpus"]["queries"],
    )
    logger.info("%-14s %10s %10s %10s", "mode", "recall@5", "recall@10", "mrr@10")
    for m in ("bm25_only", "vector_only", "rrf_fusion"):
        row = metrics[m]
        logger.info("%-14s %10.4f %10.4f %10.4f", m, row["recall@5"], row["recall@10"], row["mrr@10"])
    ab = metrics["ablation"]
    logger.info(
        "ablation: best single mode recall@10=%.4f, rrf delta=%+.4f -> rrf >= max(single): %s",
        ab["best_single_mode_recall@10"],
        ab["rrf_vs_best_single"],
        "PASS" if ab["rrf_geq_max_single"] else "FAIL",
    )


async def run(mode: str, seed: int, output: str) -> int:
    from fusion_rag.engine.recall_bench import MockEmbed, generate_corpus, run_benchmark

    docs, queries = generate_corpus(seed=seed)
    if mode == "real":
        from fusion_rag.embed.client import EmbeddingClient

        # FUSION_MLX_URL must reach reach the engine (or gateway) WITH auth —
        # EmbeddingClient's default :11432 gateway 401s without a key and the
        # client silently degrades to the local sentence-transformers fallback,
        # which would make a "real" run not run against the endpoint at all.
        base_url = os.environ.get("FUSION_MLX_URL", "http://127.0.0.1:11434/v1")
        # cache_enabled=False: the SQLite embedding cache is keyed by text, so
        # a prior run (or a local-fallback run) would silently serve stale
        # vectors and the "real" numbers would be neither real nor fresh.
        embed = EmbeddingClient(base_url=base_url, cache_enabled=False)
        ok = await embed.health()
        if not ok:
            logger.error(
                "real mode: fusion-mlx embeddings not reachable at %s — start it or run --mode mock", embed.base_url
            )
            return 2
        mode_label = "mode=real (BGE-M3 via fusion-mlx) — QUALITY evidence"
    else:
        embed = MockEmbed()
        mode_label = "mode=mock (deterministic hashed embeddings) — WIRING evidence only, NOT quality evidence"

    tmpdir = tempfile.mkdtemp(prefix="recall_bench_")
    try:
        metrics = await run_benchmark(embed, docs, queries, workdir=tmpdir)
        metrics["embed_mode"] = mode
        metrics["seed"] = seed
        _print_table(mode_label, metrics)
        if output:
            os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
            with open(output, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            logger.info("wrote results to %s", output)
        if mode == "mock":
            return 0 if metrics["ablation"]["rrf_geq_max_single"] else 1
        return 0
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        logger.info("cleaned up temp store dir")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fusion-RAG recall benchmark (issue #79)")
    parser.add_argument("--mode", choices=["mock", "real"], default="mock")
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument(
        "--output",
        default="",
        help=f"write JSON results here (e.g. docs/benchmarks/recall-{date.today().isoformat()}.json)",
    )
    args = parser.parse_args()
    return asyncio.run(run(args.mode, args.seed, args.output))


if __name__ == "__main__":
    sys.exit(main())
