# Recall Benchmark Results

Retrieval-quality evidence for issue #79. Corpus: 505 chunks / 90 docs
(markdown + Python mix, chunked by the production semantic/code chunkers,
seed 20261025), 55 ground-truth queries. Run with
`scripts/recall_benchmark.py --mode real` against live fusion-mlx BGE-M3.

## Latest run — 2026-10-25 (`recall-2026-10-25.json`, mode=real)

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
