# R3 (issue #125): rerank pool ablation wiring test (fake cross-encoder —
# real-model quality evidence comes from scripts/rerank_pool_ablation.py).

from __future__ import annotations

import tempfile

from fusion_rag.engine.recall_bench import (
    SEED,
    MockEmbed,
    generate_corpus,
    run_rerank_pool_ablation,
)


class _FakeReranker:
    # Deterministic relevance proxy: keeps the RRF order but stamps scores,
    # so the ablation exercises the full path without a live model.
    def __init__(self, **_kwargs):
        pass

    async def rerank(self, query, documents, top_k=5):
        return [dict(d, score=1.0 - i * 0.01) for i, d in enumerate(documents[:top_k])]


class TestPoolAblation:
    async def test_pools_reported_with_latency(self, monkeypatch):
        monkeypatch.setattr(
            "fusion_rag.engine.cross_encoder_reranker.CrossEncoderReranker", _FakeReranker
        )
        docs, queries = generate_corpus(seed=SEED, topics=5, distractors=3)
        with tempfile.TemporaryDirectory() as workdir:
            out = await run_rerank_pool_ablation(
                MockEmbed(), docs, queries, workdir=workdir, pools=(2, 4)
            )
        assert out["mode"] == "real"
        assert set(out["pools"]) == {2, 4}
        for pool, m in out["pools"].items():
            assert 0.0 <= m["recall@10"] <= 1.0
            assert 0.0 <= m["ndcg@10"] <= 1.0
            assert m["rerank_calls"] > 0
            assert m["rerank_latency_p50_ms"] >= 0.0
            assert m["rerank_latency_p95_ms"] >= m["rerank_latency_p50_ms"]
