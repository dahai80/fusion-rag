"""CI gate for the recall benchmark (issue #79): deterministic seed, mock
embeddings, full ≥500-chunk / ≥50-query corpus. Asserts the pipeline runs
end-to-end through the real chunker + VectorStore + BM25Index + HybridSearch
code paths and that RRF fusion never drops recall@10 below the best single
mode on the mock set. Wiring-regression evidence only — quality evidence
comes from `python scripts/recall_benchmark.py --mode real`."""

from __future__ import annotations

import tempfile

from fusion_rag.engine.recall_bench import (
    SEED,
    MockEmbed,
    build_chunk_records,
    generate_corpus,
    run_benchmark,
)


def test_corpus_size_contract():
    docs, queries = generate_corpus(seed=SEED)
    assert len(queries) >= 50
    assert len(docs) >= 80


def test_corpus_deterministic():
    d1, q1 = generate_corpus(seed=SEED)
    d2, q2 = generate_corpus(seed=SEED)
    assert d1 == d2
    assert q1 == q2
    d3, _ = generate_corpus(seed=SEED + 1)
    assert d3[0]["text"] != d1[0]["text"]


def test_chunk_records_meet_500_chunk_floor():
    import asyncio

    docs, _ = generate_corpus(seed=SEED)
    records, doc_to_chunks = asyncio.run(build_chunk_records(docs))
    assert len(records) >= 500
    assert len(doc_to_chunks) == len(docs)
    # every chunk id maps back to its doc and carries evidence metadata
    for r in records[:20]:
        assert r["metadata"]["doc_id"]
        assert r["id"].startswith(r["metadata"]["doc_id"] + "_")


def test_mock_benchmark_pipeline_and_ablation():
    import asyncio

    docs, queries = generate_corpus(seed=SEED)
    metrics = asyncio.run(run_benchmark(MockEmbed(), docs, queries, workdir=tempfile.mkdtemp()))
    assert metrics["corpus"]["chunks"] >= 500
    assert metrics["corpus"]["queries"] >= 50
    for mode in ("bm25_only", "vector_only", "rrf_fusion"):
        assert 0.0 <= metrics[mode]["recall@5"] <= 1.0
        assert 0.0 <= metrics[mode]["recall@10"] <= 1.0
        assert 0.0 <= metrics[mode]["mrr@10"] <= 1.0
        # mock channels carry real signal (distinctive terms) — not a vacuous pass
        assert metrics[mode]["recall@10"] > 0.5
    # CI contract: fusion never drops below the best single mode
    assert metrics["ablation"]["rrf_geq_max_single"], (
        f"rrf recall@10 {metrics['rrf_fusion']['recall@10']} < best single "
        f"{metrics['ablation']['best_single_mode_recall@10']}"
    )
