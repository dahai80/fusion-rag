---
doc_id: fusion-rag/recall-benchmarks
product: fusion-rag
class: platform
entry: service
doc_type: guide
title: {zh: "召回基准结果", en: "Recall benchmark results"}
audience: engineer
status: draft
version: 0.8.21
updated: 2026-11-07
owner: "@dahai"
source_of_truth:
  - docs/benchmarks/README.md
---

# Recall Benchmark Results

Retrieval-quality evidence for issue #79. Corpus: 505 chunks / 90 docs
(markdown + Python mix, chunked by the production semantic/code chunkers,
seed 20261025), 55 ground-truth queries. Run with
`scripts/recall_benchmark.py --mode real` against live fusion-mlx BGE-M3.

## Rerank pool A/B — 2026-11-14 (`rerank-pool-ablation.json`, mode=real, R3/#125)

Live bge-reranker-v2-m3 (via fusion-mlx `/v1/rerank`, direct) over the same 55-query corpus, RRF candidates reranked per pool then truncated to top-10:

| pool | recall@5 | recall@10 | mrr@10 | ndcg@10 | map@10 | p50 ms | p95 ms |
|-----:|---------:|----------:|-------:|--------:|-------:|-------:|-------:|
| 20   | 0.7939 | 0.9212 | 1.0000 | 0.9469 | 0.9212 | 114   | 162   |
| 50   | 0.7939 | 0.9212 | 1.0000 | 0.9469 | 0.9212 | 260   | 394   |
| 100  | 0.7939 | 0.9212 | 1.0000 | 0.9469 | 0.9212 | 600   | 1017  |
| 150  | 0.7939 | 0.9212 | 1.0000 | 0.9469 | 0.9212 | 1298  | 1838  |

**Verdict: `rerank_top_n=20` already saturates on this corpus** — identical metrics at every pool size, latency grows ~linearly (5.6x p50 from 20→100, 11x to 150). Default stays 20. Caveat: the synthetic corpus has clean ground truth and few hard negatives; re-run `scripts/rerank_pool_ablation.py` on a real labeled set before concluding for production traffic.

## Latest run — 2026-11-14 (`recall-2026-11-14.json`, mode=real)

| mode | recall@5 | recall@10 | mrr@10 |
|------|----------|-----------|--------|
| BM25-only (jieba) | 0.7939 | 0.9212 | 1.0000 |
| Vector-only (BGE-M3) | 0.7909 | 0.9182 | 1.0000 |
| RRF fusion | 0.7939 | **0.9212** | 1.0000 |

Re-run with the R1 fusion consolidation (issue #123): RRF remains the single
supported fusion path; the weighted-score `FusionRetriever` is deprecated
(experimental only). Numbers confirm the 2026-10-25 baseline — rrf >= max(single)
on recall@10 holds.

## Previous run — 2026-10-25 (`recall-2026-10-25.json`, mode=real)

| mode | recall@5 | recall@10 | mrr@10 |
|------|----------|-----------|--------|
| BM25-only (jieba) | 0.7939 | 0.9212 | 1.0000 |
| Vector-only (BGE-M3) | 0.7848 | 0.9182 | 1.0000 |
| RRF fusion | 0.7909 | **0.9212** | 1.0000 |

## Ablation decision

**Keep RRF fusion as the hybrid default.** On this corpus RRF matches the
best single mode at recall@10 (0.9212, equal to BM25-only, +0.003 over
vector-only) and never drops below either channel. The corpus is
keyword-heavy by construction (distinctive terms), which favors BM25; the
fusion's value is insurance on corpora where either channel degrades
(paraphrased queries hurt BM25, rare-token collisions hurt dense). The CI
gate (`tests/test_recall_benchmark.py`) pins the no-regression contract:
rrf recall@10 >= max(bm25, vector) on the deterministic mock set.

Mock-mode numbers (`--mode mock`, hashed-bag embeddings) are
wiring-regression evidence only and are never recorded here as quality
evidence.
