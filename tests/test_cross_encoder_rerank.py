"""Issue #70 — cross-encoder rerank (fusion-mlx POST /v1/rerank) tests.

Locks the contract:
- CrossEncoderReranker.rerank reorders by relevance_score, stamps score, returns top_k.
- Empty docs => []. Network/5xx/404/400 => LLMUnavailable (route fallback contract).
- _do_rerank cross_encoder backend reorders the pool; on failure falls back to
  llm rerank then original order.
- /search: rerank=true + rerank_top_n fetches a wider pool, returns top_k reordered.
- /search default (no FUSION_RAG_RERANK_MODEL): use_rerank stays False (backward compat).
- /search with FUSION_RAG_RERANK_MODEL set: use_rerank defaults True.

The /v1/rerank HTTP call is mocked by monkeypatching CrossEncoderReranker._call_rerank
— no live fusion-mlx needed.
"""
from __future__ import annotations

import contextlib
import tempfile
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from fusion_rag.api.server import create_app
from fusion_rag.engine.cross_encoder_reranker import CrossEncoderReranker
from fusion_rag.engine.llm_errors import LLMUnavailable


async def _fake_embed_batch(self, texts):
    return [[0.01] * 1024 for _ in texts]


async def _fake_health(self):
    return True


def _teardown(tc):
    with contextlib.suppress(Exception):
        tc.__exit__(None, None, None)
    tc._patcher.stop()
    with contextlib.suppress(Exception):
        tc._health_patcher.stop()
    from fusion_rag.api import auth as auth_mod

    auth_mod._auth_backend = None
    from fusion_rag.api.tenant import reset_request_tenant

    reset_request_tenant()
    from fusion_rag.engine.runtime_config import reset_runtime_config

    reset_runtime_config()


def _make_client(tmp_path, monkeypatch, *, rerank_model="", rerank_backend="llm", rerank_top_n=None):
    from fusion_rag.api import auth as auth_mod
    from fusion_rag.embed.client import EmbeddingClient

    monkeypatch.delenv("FUSION_RAG_REQUIRE_GATEWAY", raising=False)
    monkeypatch.delenv("FUSION_RAG_REQUIRE_IDENTITY", raising=False)
    monkeypatch.setenv("FUSION_RAG_RERANK_MODEL", rerank_model)
    monkeypatch.setenv("FUSION_RAG_RERANK_BACKEND", rerank_backend)
    if rerank_top_n is not None:
        monkeypatch.setenv("FUSION_RAG_RERANK_TOP_N", str(rerank_top_n))
    else:
        monkeypatch.delenv("FUSION_RAG_RERANK_TOP_N", raising=False)
    from fusion_rag.engine.runtime_config import reset_runtime_config

    reset_runtime_config()
    auth_mod._auth_backend = auth_mod.ApiKeyBackend(admin_key="admin-key", db_path=str(tmp_path / "auth.db"))
    storage_dir = tempfile.mkdtemp()
    app = create_app(kb_storage_dir=storage_dir)
    patcher = patch.object(EmbeddingClient, "embed_batch", _fake_embed_batch)
    patcher.start()
    health_patcher = patch.object(EmbeddingClient, "health", _fake_health)
    health_patcher.start()
    tc_cm = TestClient(app)
    tc_cm.__enter__()
    tc_cm.kb_storage_dir = storage_dir
    tc_cm._patcher = patcher
    tc_cm._health_patcher = health_patcher
    return tc_cm


ADMIN = {"X-API-Key": "admin-key"}


def _docs(n):
    return [{"id": str(i), "text": f"doc number {i}", "score": float(n - i)} for i in range(n)]


class TestCrossEncoderRerankerUnit:
    @pytest.mark.asyncio
    async def test_rerank_reorders_and_stamps_score(self, tmp_path, monkeypatch):
        r = CrossEncoderReranker(mlx_base_url="http://x/v1", model="m")
        # fake /v1/rerank: doc 2 most relevant, then 0, then 1.
        async def fake_call(payload, headers):
            return [
                {"index": 2, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.5},
                {"index": 1, "relevance_score": 0.1},
            ]

        r._call_rerank = fake_call
        out = await r.rerank("q", _docs(3), top_k=3)
        assert [d["id"] for d in out] == ["2", "0", "1"]
        assert out[0]["score"] == 0.9

    @pytest.mark.asyncio
    async def test_rerank_empty(self, tmp_path, monkeypatch):
        r = CrossEncoderReranker(mlx_base_url="http://x/v1", model="m")
        assert await r.rerank("q", [], top_k=5) == []

    @pytest.mark.asyncio
    async def test_rerank_truncates_to_top_k(self, tmp_path, monkeypatch):
        r = CrossEncoderReranker(mlx_base_url="http://x/v1", model="m")
        async def fake_call(payload, headers):
            return [{"index": i, "relevance_score": 1.0 - i * 0.1} for i in range(5)]

        r._call_rerank = fake_call
        out = await r.rerank("q", _docs(5), top_k=2)
        assert len(out) == 2
        assert out[0]["id"] == "0"

    @pytest.mark.asyncio
    async def test_rerank_network_error_raises(self, tmp_path, monkeypatch):
        r = CrossEncoderReranker(mlx_base_url="http://x/v1", model="m")
        async def boom(payload, headers):
            raise LLMUnavailable()

        r._call_rerank = boom
        with pytest.raises(LLMUnavailable):
            await r.rerank("q", _docs(3), top_k=3)

    @pytest.mark.asyncio
    async def test_call_rerank_404_raises(self, tmp_path, monkeypatch):
        r = CrossEncoderReranker(mlx_base_url="http://x/v1", model="missing-model")

        class FakeResp:
            status_code = 404

            def json(self):
                return {"detail": "Model not found"}

        async def fake_post(*a, **k):
            return FakeResp()

        from fusion_rag.engine import cross_encoder_reranker as cer

        monkeypatch.setattr(cer, "with_retry", lambda fn, **kw: fake_post())
        with pytest.raises(LLMUnavailable):
            await r._call_rerank({}, {})


class TestDoRerankBackend:
    @pytest.mark.asyncio
    async def test_cross_encoder_backend_reorders_pool(self, tmp_path, monkeypatch):
        # _do_rerank with backend=cross_encoder routes to CrossEncoderReranker.
        from fusion_rag.api import routes as routes_mod

        monkeypatch.setenv("FUSION_RAG_RERANK_MODEL", "m")
        monkeypatch.setenv("FUSION_RAG_RERANK_BACKEND", "cross_encoder")
        from fusion_rag.engine.runtime_config import reset_runtime_config

        reset_runtime_config()

        class FakeEmbed:
            base_url = "http://127.0.0.1:11432/v1"

        monkeypatch.setattr(routes_mod, "_get_embed_client", lambda: FakeEmbed())
        called = {}

        async def fake_rerank(self, query, documents, top_k=5):
            called["args"] = (query, len(documents), top_k)
            return sorted(documents, key=lambda d: d["score"], reverse=True)[:top_k]

        monkeypatch.setattr(CrossEncoderReranker, "rerank", fake_rerank)
        out = await routes_mod._do_rerank("q", _docs(10), top_k=5, backend="cross_encoder", model="m")
        assert len(out) == 5
        assert called["args"][1] == 10  # full pool passed, not pre-truncated

    @pytest.mark.asyncio
    async def test_cross_encoder_failure_falls_back(self, tmp_path, monkeypatch):
        # cross-encoder raises LLMUnavailable => _do_rerank falls back to original order
        # (the LLM-prompt Reranker is also mocked to raise, so the final fallback is
        # original order — proving the fallback chain does not crash).
        from fusion_rag.api import routes as routes_mod

        monkeypatch.setenv("FUSION_RAG_RERANK_BACKEND", "cross_encoder")
        from fusion_rag.engine.runtime_config import reset_runtime_config

        reset_runtime_config()

        class FakeEmbed:
            base_url = "http://127.0.0.1:11432/v1"

        monkeypatch.setattr(routes_mod, "_get_embed_client", lambda: FakeEmbed())

        async def boom_rerank(self, query, documents, top_k=5):
            raise LLMUnavailable()

        monkeypatch.setattr(CrossEncoderReranker, "rerank", boom_rerank)
        from fusion_rag.engine.reranker import Reranker

        async def boom_llm(self, query, documents, top_k=5):
            raise LLMUnavailable()

        monkeypatch.setattr(Reranker, "rerank", boom_llm)
        docs = _docs(8)
        out = await routes_mod._do_rerank("q", docs, top_k=4, backend="cross_encoder", model="m")
        assert len(out) == 4
        # original order preserved (fallback), not reordered.
        assert [d["id"] for d in out] == [d["id"] for d in docs[:4]]


class TestSearchRerankIntegration:
    def _seed_kb(self, tc, tmp_path, kb_id="kb1"):
        r = tc.post("/kb/bases", json={"name": kb_id, "kb_id": kb_id}, headers=ADMIN)
        assert r.status_code == 200, r.text
        # Ingest a few real temp files so search has a pool. The docs endpoint
        # requires a real file_path (LFI-guarded ingest); write txt files.
        for i in range(6):
            fp = tmp_path / f"d{i}.txt"
            fp.write_text(f"document content number {i} keyword{i} unique{i}")
            ir = tc.post(
                f"/kb/bases/{kb_id}/documents",
                json={"file_path": str(fp)},
                headers=ADMIN,
            )
            assert ir.status_code == 200, f"ingest {i}: {ir.status_code} {ir.text}"
        return kb_id

    def test_search_default_no_rerank_model(self, tmp_path, monkeypatch):
        # No FUSION_RAG_RERANK_MODEL => use_rerank defaults False, no rerank call.
        rerank_calls = []

        async def fake_rerank(self, query, documents, top_k=5):
            rerank_calls.append(len(documents))
            return documents[:top_k]

        monkeypatch.setattr(CrossEncoderReranker, "rerank", fake_rerank)
        tc = _make_client(tmp_path, monkeypatch, rerank_model="", rerank_backend="cross_encoder")
        try:
            kb = self._seed_kb(tc, tmp_path)
            r = tc.post(f"/kb/bases/{kb}/search", json={"query": "keyword0", "top_k": 3}, headers=ADMIN)
            assert r.status_code == 200, r.text
            assert rerank_calls == []  # rerank not engaged by default
        finally:
            _teardown(tc)

    def test_search_rerank_model_default_on(self, tmp_path, monkeypatch):
        # FUSION_RAG_RERANK_MODEL set => use_rerank defaults True.
        rerank_calls = []

        async def fake_rerank(self, query, documents, top_k=5):
            rerank_calls.append((len(documents), top_k))
            return documents[:top_k]

        monkeypatch.setattr(CrossEncoderReranker, "rerank", fake_rerank)
        tc = _make_client(tmp_path, monkeypatch, rerank_model="bge-reranker-v2-m3",
                          rerank_backend="cross_encoder", rerank_top_n=20)
        try:
            kb = self._seed_kb(tc, tmp_path)
            r = tc.post(f"/kb/bases/{kb}/search", json={"query": "keyword0", "top_k": 3}, headers=ADMIN)
            assert r.status_code == 200, r.text
            assert len(rerank_calls) == 1
            # pool widened to rerank_top_n (20), final top_k 3.
            pool, top_k = rerank_calls[0]
            assert top_k == 3
            assert pool >= 3
        finally:
            _teardown(tc)

    def test_search_explicit_rerank_true(self, tmp_path, monkeypatch):
        # Caller passes rerank=true explicitly even without env model.
        rerank_calls = []

        async def fake_rerank(self, query, documents, top_k=5):
            rerank_calls.append(len(documents))
            return documents[:top_k]

        monkeypatch.setattr(CrossEncoderReranker, "rerank", fake_rerank)
        tc = _make_client(tmp_path, monkeypatch, rerank_model="", rerank_backend="cross_encoder")
        try:
            kb = self._seed_kb(tc, tmp_path)
            r = tc.post(
                f"/kb/bases/{kb}/search",
                json={"query": "keyword0", "top_k": 2, "rerank": True, "rerank_top_n": 5},
                headers=ADMIN,
            )
            assert r.status_code == 200, r.text
            assert len(rerank_calls) == 1
        finally:
            _teardown(tc)
