"""Issue #106 — local embedding fallback: one shared model load + fail-fast mode.

Two defects in the fallback path: (1) get_local_model had a module singleton
but NO lock, so concurrent fallback requests each cold-loaded BGE-M3 (~2-4 min
per load on MPS) instead of one load + reuse; (2) there was no way to opt out
of the local fallback, so with fusion-mlx down every request hung for minutes.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import types

import pytest

from fusion_rag.embed import local as local_mod
from fusion_rag.embed.client import EmbeddingClient, EmbeddingError


class _FakeSentenceTransformer:
    loads = 0
    load_lock = threading.Lock()

    def __init__(self, name: str, cache_folder: str = ""):
        with _FakeSentenceTransformer.load_lock:
            _FakeSentenceTransformer.loads += 1

    def get_embedding_dimension(self) -> int:
        return 1024

    def encode(self, texts, normalize_embeddings: bool = False):
        return [[0.01] * 1024 for _ in texts]


@pytest.fixture()
def fake_st(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", types.ModuleType("sentence_transformers"))
    sys.modules["sentence_transformers"].SentenceTransformer = _FakeSentenceTransformer
    _FakeSentenceTransformer.loads = 0
    local_mod.reset_local_model()
    yield _FakeSentenceTransformer
    local_mod.reset_local_model()


def test_concurrent_requests_load_model_once(fake_st):
    # Issue #106 core defect: N concurrent fallback callers must trigger ONE
    # SentenceTransformer load, not N. Threads start on a barrier to maximize
    # the race window that used to produce 4+ loads.
    n = 8
    barrier = threading.Barrier(n)
    results: list = [None] * n
    errors: list = []

    def worker(i: int):
        try:
            barrier.wait(timeout=5)
            results[i] = local_mod.get_local_model("BGE-M3")
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors
    assert fake_st.loads == 1
    assert all(r is results[0] for r in results)


def test_sequential_calls_reuse_loaded_model(fake_st):
    first = local_mod.get_local_model("BGE-M3")
    second = local_mod.get_local_model("BGE-M3")
    assert first is second
    assert fake_st.loads == 1


def test_failed_load_is_retryable(fake_st, monkeypatch):
    # A failed load must NOT leave a poisoned singleton; the next caller
    # retries (existing no-cache-on-failure behavior preserved under the lock).
    calls = []

    def _flaky(name: str, cache_folder: str = ""):
        calls.append(name)
        if len(calls) == 1:
            raise RuntimeError("disk hiccup")
        return _FakeSentenceTransformer(name)

    monkeypatch.setattr(sys.modules["sentence_transformers"], "SentenceTransformer", _flaky)
    assert local_mod.get_local_model("BGE-M3") is None
    assert local_mod.get_local_model("BGE-M3") is not None


def _dead_mlx_client() -> EmbeddingClient:
    return EmbeddingClient(
        base_url="http://127.0.0.1:1/v1",
        model="BGE-M3",
        api_key="test",
        timeout=0.2,
        max_retries=1,
        cache_enabled=False,
    )


def test_fallback_disabled_fails_fast(monkeypatch):
    # FUSION_RAG_EMBED_LOCAL_FALLBACK=0: embed fails FAST with a retryable
    # EmbeddingError and never touches the local model (no multi-minute hang).
    from fusion_rag.engine.runtime_config import reset_runtime_config

    monkeypatch.setenv("FUSION_RAG_EMBED_LOCAL_FALLBACK", "0")
    reset_runtime_config()
    called = []

    def _spy_embed_local(texts, model_name):
        called.append(texts)
        return [[0.01] * 1024 for _ in texts]

    monkeypatch.setattr(local_mod, "embed_local", _spy_embed_local)
    try:
        client = _dead_mlx_client()
        with pytest.raises(EmbeddingError, match="local fallback disabled"):
            asyncio.run(client.embed("hello"))
        assert called == []
    finally:
        reset_runtime_config()


def test_fallback_default_on_uses_local(monkeypatch):
    # Default unchanged: MLX down falls back to local sentence-transformers.
    from fusion_rag.engine.runtime_config import reset_runtime_config

    monkeypatch.delenv("FUSION_RAG_EMBED_LOCAL_FALLBACK", raising=False)
    reset_runtime_config()

    def _fake_embed_local(texts, model_name):
        return [[0.02] * 1024 for _ in texts]

    monkeypatch.setattr(local_mod, "embed_local", _fake_embed_local)
    try:
        client = _dead_mlx_client()
        vec = asyncio.run(client.embed("hello"))
        assert len(vec) == 1024
    finally:
        reset_runtime_config()


def test_health_skips_local_when_disabled(monkeypatch):
    # Fail-fast mode: local model is not part of the serving path, so health()
    # must NOT report healthy via a local model no request may use.
    from fusion_rag.engine.runtime_config import reset_runtime_config

    monkeypatch.setenv("FUSION_RAG_EMBED_LOCAL_FALLBACK", "0")
    reset_runtime_config()
    try:
        client = _dead_mlx_client()
        assert asyncio.run(client.health()) is False
    finally:
        reset_runtime_config()


def test_embedding_error_maps_to_503(tmp_path, monkeypatch):
    # Issue #106: EmbeddingError (all providers down / fail-fast) must surface
    # as a clean retryable 503 with a readable detail, not a bare 500
    # "Internal Server Error".
    import tempfile
    from unittest.mock import patch

    from fastapi.testclient import TestClient

    from fusion_rag.api.server import create_app
    from fusion_rag.embed.client import EmbeddingClient

    async def _raise_embed_error(self, texts):
        raise EmbeddingError("all embedding providers failed (test)")

    monkeypatch.setenv("FUSION_RAG_DATA_DIR", str(tmp_path))
    app = create_app(kb_storage_dir=tempfile.mkdtemp())
    with (
        patch.object(EmbeddingClient, "embed_batch", _raise_embed_error),
        TestClient(app) as tc,
    ):
        tc.post("/kb/bases", json={"kb_id": "fb503", "name": "t"})
        resp = tc.post("/kb/bases/fb503/search", json={"query": "q", "top_k": 2})
        assert resp.status_code == 503
        assert "all embedding providers failed" in resp.json()["detail"]


def test_ready_respects_health_false(tmp_path, monkeypatch):
    # Issue #106: /ready used to ignore health()'s return value — health()
    # returning False (no exception) still reported "ok". Must be 503.
    import tempfile
    from unittest.mock import patch

    from fastapi.testclient import TestClient

    from fusion_rag.api.server import create_app
    from fusion_rag.embed.client import EmbeddingClient

    async def _false_health(self):
        return False

    monkeypatch.setenv("FUSION_RAG_DATA_DIR", str(tmp_path))
    app = create_app(kb_storage_dir=tempfile.mkdtemp())
    with (
        patch.object(EmbeddingClient, "health", _false_health),
        TestClient(app) as tc,
    ):
        resp = tc.get("/ready")
        assert resp.status_code == 503
        assert resp.json()["checks"]["embedding"] == "unavailable: embedding backend down"
