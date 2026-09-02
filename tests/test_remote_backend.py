"""RemoteBackend (#remote-store) — full-stack client/server round-trip tests.

Spins an in-process fusion-rag app (routes_store server on a LocalBackend KB)
and a RemoteBackend httpx client pointed at it via httpx ASGITransport. Verifies
the full-stack path: client add_batch → server → LocalBackend → client search
recalls, keyword_search, delete_by_doc, count, clear. No open port, no
fusion-mlx (embeddings mocked). Exercises the contract both sides share.
"""
from __future__ import annotations

import contextlib
import tempfile
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


async def _fake_embed_batch(self, texts):
    return [[0.01] * 1024 for _ in texts]


def _make_server(tmp_path, monkeypatch):
    from fusion_rag.api import auth as auth_mod
    from fusion_rag.api.server import create_app
    from fusion_rag.embed.client import EmbeddingClient

    monkeypatch.delenv("FUSION_RAG_REQUIRE_GATEWAY", raising=False)
    monkeypatch.delenv("FUSION_RAG_STORE_BACKEND", raising=False)
    auth_mod._auth_backend = auth_mod.ApiKeyBackend(admin_key="admin-key", db_path=str(tmp_path / "auth.db"))
    storage_dir = tempfile.mkdtemp()
    app = create_app(kb_storage_dir=storage_dir)
    patcher = patch.object(EmbeddingClient, "embed_batch", _fake_embed_batch)
    patcher.start()
    tc = TestClient(app)
    tc.__enter__()
    tc._storage_dir = storage_dir
    tc._patcher = patcher
    return tc


def _teardown(tc):
    with contextlib.suppress(Exception):
        tc.__exit__(None, None, None)
    tc._patcher.stop()
    from fusion_rag.api import auth as auth_mod

    auth_mod._auth_backend = None
    from fusion_rag.api.tenant import reset_request_tenant

    reset_request_tenant()


def _remote_client(server_tc, kb_id, monkeypatch):
    from fusion_rag.store.remote_backend import RemoteBackend

    # Starlette TestClient subclasses httpx.Client and mounts an ASGI transport
    # internally, so it IS a synchronous httpx.Client against the in-process
    # app. Reuse it directly as RemoteBackend's client — no open port, real
    # client→server HTTP round-trip through the ASGI stack.
    server_tc.headers.update({"X-API-Key": "admin-key"})
    backend = RemoteBackend(
        vector_path=f"{tempfile.gettempdir()}/test-stores/{kb_id}/vectors",
        dimension=1024,
        endpoint="http://testserver",
        api_key="admin-key",
        kb_id=kb_id,
    )
    backend._client = server_tc
    return backend


def _seed_kb(server_tc, kb_id):
    r = server_tc.post("/kb/bases", json={"name": "remote-kb", "kb_id": kb_id}, headers={"X-API-Key": "admin-key"})
    assert r.status_code == 200, r.text
    return kb_id


def _record(chunk_id, vector):
    return {
        "id": chunk_id,
        "vector": vector,
        "text": f"chunk text {chunk_id}",
        "doc_path": f"/docs/{chunk_id.split('_')[0]}.md",
        "doc_name": f"{chunk_id.split('_')[0]}.md",
        "doc_type": "md",
        "chunk_index": 0,
        "metadata": {"src": "test"},
        "context": "",
    }


class TestRemoteBackendRoundTrip:
    def test_add_and_search_recall(self, tmp_path, monkeypatch):
        server = _make_server(tmp_path, monkeypatch)
        try:
            kb_id = _seed_kb(server, "rt-kb")
            rb = _remote_client(server, kb_id, monkeypatch)
            # two distinct docs, orthogonal-ish vectors
            v1 = [1.0] + [0.0] * 1023
            v2 = [0.0] * 512 + [1.0] + [0.0] * 511
            rb.add_batch([_record("d1_0", v1), _record("d2_0", v2)])
            assert rb.count() == 2
            res = rb.search(v1, top_k=2)
            assert len(res) >= 1
            top = res[0]
            assert top["id"] == "d1_0", f"expected d1_0 top, got {top['id']}"
            assert "score" in top and "text" in top and "doc_path" in top
        finally:
            _teardown(server)

    def test_keyword_search(self, tmp_path, monkeypatch):
        server = _make_server(tmp_path, monkeypatch)
        try:
            kb_id = _seed_kb(server, "kw-kb")
            rb = _remote_client(server, kb_id, monkeypatch)
            rb.add_batch([_record("d1_0", [0.1] * 1024), _record("d2_0", [0.2] * 1024)])
            res = rb.keyword_search("chunk", top_k=10)
            assert len(res) == 2, f"keyword_search returned {len(res)}"
        finally:
            _teardown(server)

    def test_delete_by_doc(self, tmp_path, monkeypatch):
        server = _make_server(tmp_path, monkeypatch)
        try:
            kb_id = _seed_kb(server, "del-kb")
            rb = _remote_client(server, kb_id, monkeypatch)
            rb.add_batch([_record("d1_0", [0.1] * 1024), _record("d1_1", [0.2] * 1024), _record("d2_0", [0.3] * 1024)])
            assert rb.count() == 3
            deleted = rb.delete_by_doc("/docs/d1.md")
            assert deleted == 2, f"deleted {deleted}, expected 2"
            assert rb.count() == 1
        finally:
            _teardown(server)

    def test_clear(self, tmp_path, monkeypatch):
        server = _make_server(tmp_path, monkeypatch)
        try:
            kb_id = _seed_kb(server, "clr-kb")
            rb = _remote_client(server, kb_id, monkeypatch)
            rb.add_batch([_record("d1_0", [0.1] * 1024)])
            assert rb.count() == 1
            rb.clear()
            assert rb.count() == 0
        finally:
            _teardown(server)


class TestRemoteBackendConfig:
    def test_missing_endpoint_raises(self):
        from fusion_rag.store.remote_backend import RemoteBackend

        with pytest.raises(ValueError, match="FUSION_RAG_REMOTE_ENDPOINT"):
            RemoteBackend(vector_path="/x/vectors", endpoint="", kb_id="k")

    def test_missing_kb_id_raises(self):
        from fusion_rag.store.remote_backend import RemoteBackend

        with pytest.raises(ValueError, match="remote kb_id"):
            RemoteBackend(vector_path="/vectors", endpoint="http://node-b:11436", kb_id="")

    def test_derives_kb_id_from_vector_path(self):
        from fusion_rag.store.remote_backend import RemoteBackend

        rb = RemoteBackend(vector_path="/home/.fusion-rag/stores/derived-kb/vectors", endpoint="http://x")
        assert rb.kb_id == "derived-kb"

    def test_env_endpoint_used(self, monkeypatch):
        from fusion_rag.store.remote_backend import RemoteBackend

        monkeypatch.setenv("FUSION_RAG_REMOTE_ENDPOINT", "http://env-node:11436")
        monkeypatch.setenv("FUSION_RAG_REMOTE_API_KEY", "env-key")
        rb = RemoteBackend(vector_path="/x/k/vectors")
        assert rb.endpoint == "http://env-node:11436"
        assert rb.api_key == "env-key"


class TestStoreServerAuth:
    def test_unauth_rejected(self, tmp_path, monkeypatch):
        server = _make_server(tmp_path, monkeypatch)
        try:
            kb_id = _seed_kb(server, "auth-kb")
            r = server.get(f"/kb/bases/{kb_id}/store/count")
            assert r.status_code == 401, f"unauth must 401: {r.status_code}"
        finally:
            _teardown(server)

    def test_auth_allowed(self, tmp_path, monkeypatch):
        server = _make_server(tmp_path, monkeypatch)
        try:
            kb_id = _seed_kb(server, "auth2-kb")
            r = server.get(f"/kb/bases/{kb_id}/store/count", headers={"X-API-Key": "admin-key"})
            assert r.status_code == 200, f"auth must pass: {r.status_code} {r.text}"
            assert r.json() == {"count": 0}
        finally:
            _teardown(server)
