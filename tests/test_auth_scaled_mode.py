"""Issue #102 — local auth.db key creation disabled in scaled mode.

Option 1 (config-plane keys only): when FUSION_RAG_REQUIRE_GATEWAY or
FUSION_RAG_REQUIRE_IDENTITY is on, the deployment is horizontally scaled and
the per-instance SQLite auth.db cannot serve as a shared key store — a key
provisioned on node A 401s on node B. POST /kb/auth/keys must refuse loudly
(409) instead of minting a single-node key. Single-tenant default is unchanged.
"""

from __future__ import annotations

import tempfile
from unittest.mock import patch

from fastapi.testclient import TestClient

from fusion_rag.api.server import create_app


async def _fake_embed_batch(self, texts):
    return [[0.01] * 1024 for _ in texts]


async def _fake_health(self):
    return True


def _make_client(tmp_path, monkeypatch, gateway=False, identity=False):
    from fusion_rag.api import auth as auth_mod
    from fusion_rag.embed.client import EmbeddingClient

    # routes_auth's management endpoints construct AuthConfig() with the
    # DEFAULT db_path — point the data dir at tmp so tests never touch the
    # real ~/.fusion-rag/auth.db.
    monkeypatch.setenv("FUSION_RAG_DATA_DIR", str(tmp_path))
    if gateway:
        monkeypatch.setenv("FUSION_RAG_REQUIRE_GATEWAY", "1")
    else:
        monkeypatch.delenv("FUSION_RAG_REQUIRE_GATEWAY", raising=False)
    if identity:
        monkeypatch.setenv("FUSION_RAG_REQUIRE_IDENTITY", "1")
    else:
        monkeypatch.delenv("FUSION_RAG_REQUIRE_IDENTITY", raising=False)
    auth_mod._auth_backend = auth_mod.ApiKeyBackend(admin_key="admin-key", db_path=str(tmp_path / "auth.db"))
    app = create_app(kb_storage_dir=tempfile.mkdtemp())
    patcher = patch.object(EmbeddingClient, "embed_batch", _fake_embed_batch)
    patcher.start()
    health_patcher = patch.object(EmbeddingClient, "health", _fake_health)
    health_patcher.start()
    tc_cm = TestClient(app)
    tc_cm.__enter__()
    tc_cm._patcher = patcher
    tc_cm._health_patcher = health_patcher
    return tc_cm


def _teardown(tc):
    import contextlib

    with contextlib.suppress(Exception):
        tc.__exit__(None, None, None)
    tc._patcher.stop()
    with contextlib.suppress(Exception):
        tc._health_patcher.stop()
    from fusion_rag.api import auth as auth_mod

    auth_mod._auth_backend = None
    from fusion_rag.api.tenant import reset_request_tenant

    reset_request_tenant()


GW = {"X-Fusion-Route": "gateway-decision", "X-API-Key": "admin-key"}
ADMIN = {"X-API-Key": "admin-key"}


def test_create_refused_in_gateway_mode(tmp_path, monkeypatch):
    tc = _make_client(tmp_path, monkeypatch, gateway=True)
    try:
        resp = tc.post("/kb/auth/keys", json={"name": "scaled-key"}, headers=GW)
        assert resp.status_code == 409
        assert "disabled in scaled mode" in resp.json()["detail"]
    finally:
        _teardown(tc)


def test_create_refused_in_identity_mode(tmp_path, monkeypatch):
    tc = _make_client(tmp_path, monkeypatch, identity=True)
    try:
        # Identity mode gates /kb/* on a JWT via fusion-identity; the create
        # refusal fires after the route-header gate, so send the gateway header
        # to reach the endpoint without a live identity service.
        resp = tc.post("/kb/auth/keys", json={"name": "scaled-key"}, headers=GW)
        assert resp.status_code in (401, 409)
        if resp.status_code == 409:
            assert "disabled in scaled mode" in resp.json()["detail"]
    finally:
        _teardown(tc)


def test_create_works_in_single_tenant_mode(tmp_path, monkeypatch):
    tc = _make_client(tmp_path, monkeypatch)
    try:
        resp = tc.post("/kb/auth/keys", json={"name": "local-key"}, headers=ADMIN)
        assert resp.status_code == 200
        assert resp.json()["key"].startswith("frg_")
    finally:
        _teardown(tc)


def test_list_still_available_in_scaled_mode(tmp_path, monkeypatch):
    # Read-only management endpoints stay up in scaled mode — an operator can
    # still inspect the local instance's keys; only creation is refused.
    tc = _make_client(tmp_path, monkeypatch, gateway=True)
    try:
        resp = tc.get("/kb/auth/keys", headers=GW)
        assert resp.status_code == 200
        assert resp.json() == {"keys": []}
    finally:
        _teardown(tc)
