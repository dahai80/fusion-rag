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

## Contextual BM25 A/B — 2026-11-14 (`contextual-bm25-ablation.json`, mode=real, issue #126)

Same deterministic corpus (55 queries, ~505 chunks). Dense channel (BGE-M3
vectors) is IDENTICAL in both arms; only the text fed to the BM25 index
changes. `prefix` arm prepends a deterministic template proxy for the
contextual-retrieval context sentence (`Context: this chunk is from document
{doc_id}. Document opening: {first 240 chars}`) — the production
`Contextualizer` writes richer LLM-generated context into the same
`chunk.context` field, and `BM25Index.add_documents` already indexes
`context + text` (verified at `bm25_index.py`).

| arm | recall@5 | recall@10 | mrr@10 | ndcg@10 | map@10 |
|-----|----------|-----------|--------|---------|--------|
| none (baseline) | 0.7939 | 0.9212 | 1.0000 | 0.9469 | 0.9212 |
| context prefix | **0.8333** | **1.0000** | 1.0000 | **1.0000** | **1.0000** |

Verdict: even a crude template context lifts BM25 recall@10 by +0.079 and
pushes recall@10/ndcg/map to 1.0 on this corpus. The production LLM-generated
context (Contextualizer) is strictly richer, so enabling it is strictly
non-regressive for the BM25 channel. Caveat: synthetic keyword-heavy corpus;
re-run on real corpora before quoting absolute numbers.

## CRAG retry-loop A/B — 2026-11-14 (`crag-ablation.json`, `crag-ablation-min3.json`, mode=real, issue #127)

Bounded grade → rewrite → re-retrieve loop (`engine/crag.py`, opt-in via
`crag_rounds` / `FUSION_RAG_CRAG_ROUNDS`) A/B'd against single-shot retrieval
on the deterministic corpus (55 queries, real BGE-M3 + real qwen grader via
fusion-mlx). Grader prompt is strict (off-topic queries grade 0 relevant —
verified live); loop fires only when the graded-relevant count falls below
`min_relevant`.

| config | retry rate | recall@10 delta | crag latency p50 |
|--------|-----------|-----------------|------------------|
| min_relevant=1 | 0.00 | 0.000 | 45.9 ms |
| min_relevant=3 | 0.00 | 0.000 | 11.0 ms |
| min_relevant=6 | 0.00 | 0.000 | 12.0 ms |

Verdict: on this corpus the loop NEVER fires — the fused top-10 always
contains enough grader-relevant chunks, so metrics are bit-identical to
single-shot and the only cost is one grading call (sub-100 ms cached; ~0.6 s
uncached). The retry path itself is verified working (unit tests cover
rewrite → re-retrieve → dedupe, degrade-on-grader-failure, round bounds; live
trace shows `sufficient` / `rewrite_noop` actions). The benchmark corpus is
single-hop and keyword-heavy by construction — the loop's target case (weak
first retrieval on multi-hop/paraphrased queries) does not occur here. Re-run
on a multi-hop labeled set before quoting any recall lift. Default
`FUSION_RAG_CRAG_ROUNDS=0` (off) unchanged.

## Query routing A/B — 2026-11-14 (`router-ablation.json`, mode=real, issue #128)

Rule-based complexity router (`engine/query_router.py`, `FUSION_RAG_ROUTER=auto`
or per-request `route`) A/B'd against always-full-pipeline on the
deterministic corpus (55 queries, real BGE-M3; full = hybrid RRF, simple =
vector-only).

| arm | recall@5 | recall@10 | mrr@10 | ndcg@10 | map@10 |
|-----|----------|-----------|--------|---------|--------|
| full (hybrid) | 0.7909 | 0.9212 | 1.0000 | 0.9460 | 0.9190 |
| routed | 0.7848 | 0.9182 | 1.0000 | 0.9430 | 0.9147 |

Route mix: 49 simple / 6 standard. Simple-bucket latency p50 3.4 ms vs 5.0 ms
full (−32%); quality delta −0.003 recall@10 — inside the eval-gate tolerance
(0.02). Routing decisions logged per query (`query routed: level=...`), the
decision rides on the `/ask` response as `route`. Default `FUSION_RAG_ROUTER=
off` unchanged.

## PPR graph-layer ablation — 2026-11-14 (`ppr-ablation.json`, mode=real, issue #129)

`GraphRAG.search_ppr` (personalized PageRank over the extracted entity graph,
seeds = query-token entity matches, chunks inherit summed entity PPR mass)
fused with the vector path via RRF — never replacing it. Ablation arms:
vector-only / vector+BFS-expand (current graph search) / vector+PPR. Graph
built deterministically from the corpus term vocabulary (entities = topic
terms, relations = within-doc co-occurrence) so the run isolates the ranking
layer, not extraction quality.

| subset | vector-only | vector+BFS | vector+PPR |
|--------|-------------|------------|------------|
| simple-fact (55 q) | 0.9182 | 0.9212 | **1.0000** |
| multi-hop two-topic (14 q) | 0.4702 | 0.4702 | **0.5119** |

HippoRAG-2 regression check: vector+PPR >= vector-only on the simple-fact
subset — PASS (1.0 >= 0.9182). PPR also strictly beats BFS on both subsets.
Caveat (recorded in the artifact): the synthetic corpus has no cross-doc
entity chains, so PPR's walk-across-documents advantage cannot express
itself — treat these numbers as a regression gate, not a quality measurement;
a real multi-hop corpus is required before enabling the layer in the service.
`search_ppr` is NOT wired into any route (feature off by default until that
validation exists).

## Chunk-size ablation — 2026-11-14 (`chunk-size-ablation.json`, mode=real, issue #130)

Semantic chunker at sizes 256/384/512/768/1024 chars (overlap = size/8) over
the labeled corpus, real BGE-M3 + BM25:

| size | chunks | vector recall@10 | bm25 recall@10 | vector map@10 | bm25 map@10 |
|------|--------|------------------|----------------|---------------|-------------|
| 256 | 582 | 0.9121 | 0.9212 | 0.8498 | 0.8818 |
| 384 | 578 | 0.9121 | 0.9212 | 0.8498 | 0.8818 |
| **512** | 505 | **0.9182** | **0.9212** | **0.9147** | **0.9212** |
| 768 | 432 | 0.9061 | 0.9091 | 0.9026 | 0.9091 |
| 1024 | 309 | 0.5394 | 0.5394 | 0.5384 | 0.5394 |

**512 @ 12.5% overlap wins recall AND map** — the existing default is
bench-validated, not magic. 768 worse on both axes; 1024 collapses (topic
docs merge into few chunks, evidence diluted across retrieved chunks).
Consequence: `Chunker.CHUNK_PROFILES` = markdown/prose 512/64 (bench winner),
code 384/48, table 256/32 (structural profiles; smaller units for boundary-
rich content). KB create accepts `chunk_profile` (optional; explicit
`chunk_size`/`chunk_overlap` override), and KB `chunk_overlap` is now
actually honored at ingest (previously the parameter existed on
KnowledgeBaseConfig but every Chunker call site dropped it — fixed).

## Late-chunking spike — 2026-11-14 (BLOCKED upstream, issue #130)

Late Chunking (arXiv 2409.04701) needs per-token embeddings from the
long-context pass, pooled over chunk spans. Probe: 7000-char input to
fusion-mlx `POST /v1/embeddings` (BGE-M3) returns exactly one pooled 1024-d
vector — no token-level field. **Conclusion: BLOCKED upstream** — filed
dahai80/fusion-mlx#1624 (opt-in `return_token_embeddings` + token offsets).
Re-evaluate the profile when that lands; no fusion-rag-side workaround
(pooled-vector arithmetic cannot reconstruct token states).
