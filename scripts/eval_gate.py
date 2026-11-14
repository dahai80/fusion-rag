#!/usr/bin/env python3
# Issue #117: CI eval gate. Runs the offline BM25 eval over the deterministic
# seed corpus and compares against scripts/eval_baseline.json.
#
# Gate rule: a metric FAILS when it drops below baseline - tolerance
# (default 0.02). Deltas are always printed so CI reports improvements and
# regressions, not just pass/fail.
#
# Usage:
#   python scripts/eval_gate.py                        # gate vs baseline
#   python scripts/eval_gate.py --update-baseline      # re-baseline current metrics
#   python scripts/eval_gate.py --tolerance 0.05       # custom tolerance

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fusion_rag.engine.eval_set import EvalStore, run_offline_eval, seed_default_cases

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("eval_gate")

BASELINE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_baseline.json")
DEFAULT_TOLERANCE = 0.02

# R12 (issue #134): retune protocol — Qdrant fusion-tuning discipline:
# "hand-tuned weights without measurement are unlikely to beat the default
# reliably." Any change to the embedding model, chunking, or fusion
# parameters MUST re-run against the fixed labeled sample before merging,
# and an intentional improvement re-baselines in the SAME PR. Full doc:
# docs/retrieval-retune-checklist.md
RETUNE_CHECKLIST = """
Retune checklist (docs/retrieval-retune-checklist.md):
  1. Identify the lever you changed: embedding model / chunking / fusion
     params / rerank — each one invalidates the previous baseline.
  2. Re-run the benchmark against the FIXED labeled sample:
     python scripts/eval_gate.py            (offline, BM25-only)
     python scripts/recall_benchmark.py     (ablation; --real for live BGE-M3)
  3. Compare per-metric: recall@5/@10, mrr@10, ndcg@10, map@10. A change
     that lifts one metric while dropping another is a trade-off — decide
     explicitly, do not cherry-pick.
  4. Improvement: commit the re-baselined scripts/eval_baseline.json in the
     SAME PR. Regression: do not merge, or raise tolerance with a written
     justification in the PR body.
"""


async def _main(tolerance: float, update_baseline: bool) -> int:
    with tempfile.TemporaryDirectory(prefix="fusion-rag-eval-") as workdir:
        metrics, per_case = await run_offline_eval(workdir)
        store = EvalStore(os.path.join(workdir, "eval_set.db"))
        await seed_default_cases(store)
        store.record_run(metrics, len(per_case))
        latest = store.latest_run()

    print(
        json.dumps(
            {"metrics": metrics, "case_count": len(per_case), "run_id": latest["run_id"] if latest else None},
            indent=2,
        )
    )

    if update_baseline:
        with open(BASELINE_PATH, "w", encoding="utf-8") as f:
            json.dump({"metrics": metrics, "case_count": len(per_case)}, f, indent=2)
            f.write("\n")
        logger.info("baseline updated at %s", BASELINE_PATH)
        return 0

    if not os.path.exists(BASELINE_PATH):
        logger.error("baseline missing at %s — run with --update-baseline once and commit it", BASELINE_PATH)
        return 2

    with open(BASELINE_PATH, encoding="utf-8") as f:
        baseline = json.load(f)
    base_metrics = baseline.get("metrics", {})

    failures = []
    print("\nmetric      baseline   current   delta   status")
    for metric in sorted(base_metrics):
        base_v = float(base_metrics[metric])
        cur_v = float(metrics.get(metric, 0.0))
        delta = round(cur_v - base_v, 4)
        status = "ok" if cur_v >= base_v - tolerance else "FAIL"
        print(f"{metric:<10}  {base_v:>8.4f}  {cur_v:>7.4f}  {delta:>+6.4f}  {status}")
        if status == "FAIL":
            failures.append(metric)
        if cur_v > base_v:
            logger.info("%s improved by %+.4f — consider --update-baseline", metric, delta)

    if failures:
        logger.error("eval gate FAILED for %s (tolerance %.4f) — retrieval regression", failures, tolerance)
        print(RETUNE_CHECKLIST)
        return 1
    logger.info("eval gate passed (tolerance %.4f)", tolerance)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="fusion-rag offline eval gate")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--update-baseline", action="store_true")
    args = parser.parse_args()
    return asyncio.run(_main(args.tolerance, args.update_baseline))


if __name__ == "__main__":
    sys.exit(main())
