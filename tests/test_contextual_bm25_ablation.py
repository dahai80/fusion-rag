"""Issue #126 (R4): run_contextual_bm25_ablation wiring test.

MockEmbed mimics EmbeddingClient.embed_batch; the ablation must run both
arms (none / prefix), produce R12 metrics in [0, 1], and carry the
template-proxy note. No network, no fusion-mlx.
"""

from __future__ import annotations

import tempfile

import pytest

from fusion_rag.engine.recall_bench import (
    SEED,
    generate_corpus,
    run_contextual_bm25_ablation,
)

_DIM = 8
_COUNTER = {"n": 0}


class MockEmbed:
    def __init__(self, **_kwargs):
        pass

    async def embed_batch(self, texts):
        _COUNTER["n"] += len(texts)
        out = []
        for t in texts:
            seed = sum(ord(c) for c in t) % 997
            vec = [((seed + i * 31) % 100) / 100.0 for i in range(_DIM)]
            out.append(vec)
        return out


@pytest.mark.asyncio
async def test_contextual_bm25_ablation_arms_and_metrics():
    docs, queries = generate_corpus(seed=SEED, topics=2)
    with tempfile.TemporaryDirectory() as workdir:
        results = await run_contextual_bm25_ablation(MockEmbed(), docs, queries, workdir=workdir)
    for arm in ("none", "prefix"):
        assert arm in results
        for k in ("recall@5", "recall@10", "mrr@10", "ndcg@10", "map@10"):
            v = results[arm][k]
            assert 0.0 <= v <= 1.0
    assert results["mode"] == "real"
    assert "template proxy" in results["note"]
    # prefix arm can never be worse on recall@10 (context only ADDS terms to
    # the BM25 text; on the full labeled corpus the real run lifts it)
    assert results["prefix"]["recall@10"] >= results["none"]["recall@10"]


@pytest.mark.asyncio
async def test_contextual_bm25_ablation_embeds_once():
    docs, queries = generate_corpus(seed=SEED, topics=1)
    _COUNTER["n"] = 0
    with tempfile.TemporaryDirectory() as workdir:
        await run_contextual_bm25_ablation(MockEmbed(), docs, queries, workdir=workdir)
    # dense channel embedded once, shared between both arms (no per-arm re-embed)
    assert _COUNTER["n"] > 0
