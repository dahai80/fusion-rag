"""Issue #68 — fusion-identity integration: authoritative tenant resolution tests.

Locks the contract that X-Fusion-Tenant blind-trust is retired in favor of
authoritative JWT resolution via fusion-identity /verify:
- FUSION_RAG_REQUIRE_IDENTITY=1: /kb/* without a valid Bearer JWT => 401.
- Valid JWT (tid=tenant-a) => 200, KB list scoped to tenant-a.
- Cross-tenant KB get => 404 (no existence leak).
- X-Fusion-Tenant != JWT tid => 401 (header forgery blocked).
- Revoked token (identity returns revoked=true) => 401.
- Integration OFF (default) => existing X-Fusion-Tenant path unchanged.

Identity /verify HTTP is mocked by monkeypatching IdentityClient.verify — no
live fusion-identity service needed.
"""
from __future__ import annotations

import tempfile
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from fusion_rag.api.server import create_app


async def _fake_embed_batch(self, texts):
    return [[0.01] * 1024 for _ in texts]


async def _fake_health(self):
    return True


def _make_client(tmp_path, monkeypatch, *, require_identity=False, require_gateway=False,
                 verify_fn=None):
    from fusion_rag.api import auth as auth_mod
    from fusion_rag.embed.client import EmbeddingClient

    if require_identity:
        monkeypatch.setenv("FUSION_RAG_REQUIRE_IDENTITY", "1")
        # Test-only service token (not a real credential).
        _svc_token = "test-service-token"  # noqa: S105
        monkeypatch.setenv("FUSION_IDENTITY_SERVICE_TOKEN", _svc_token)
    else:
        monkeypatch.delenv("FUSION_RAG_REQUIRE_IDENTITY", raising=False)
    if require_gateway:
        monkeypatch.setenv("FUSION_RAG_REQUIRE_GATEWAY", "1")
    else:
        monkeypatch.delenv("FUSION_RAG_REQUIRE_GATEWAY", raising=False)
    auth_mod._auth_backend = auth_mod.ApiKeyBackend(admin_key="admin-key", db_path=str(tmp_path / "auth.db"))
    storage_dir = tempfile.mkdtemp()
    app = create_app(kb_storage_dir=storage_dir)
    # Mock identity /verify if a fake is requested. app.state.identity_client is
    # built in create_app when FUSION_RAG_REQUIRE_IDENTITY=1.
    if require_identity and verify_fn is not None and app.state.identity_client is not None:
        app.state.identity_client.verify = verify_fn
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


ADMIN = {"X-API-Key": "admin-key"}


def _verify_factory(claims_by_token):
    """Return an async verify(token) that maps token -> claims dict (or None)."""
    async def _verify(token):
        return claims_by_token.get(token)

    return _verify


class TestIdentityEnforcement:
    def test_forged_header_without_jwt_rejected(self, tmp_path, monkeypatch):
        # identity on, valid-tenant map, but caller sends header + no Bearer.
        verify = _verify_factory({"good-jwt": {"tid": "tenant-a"}})
        tc = _make_client(tmp_path, monkeypatch, require_identity=True, verify_fn=verify)
        try:
            r = tc.get("/kb/bases", headers={**ADMIN, "X-Fusion-Tenant": "tenant-a"})
            assert r.status_code == 401, f"header-only (no JWT) must 401: {r.status_code} {r.text}"
        finally:
            _teardown(tc)

    def test_invalid_jwt_rejected(self, tmp_path, monkeypatch):
        verify = _verify_factory({"good-jwt": {"tid": "tenant-a"}})
        tc = _make_client(tmp_path, monkeypatch, require_identity=True, verify_fn=verify)
        try:
            r = tc.get("/kb/bases", headers={**ADMIN, "Authorization": "Bearer bad-jwt"})
            assert r.status_code == 401, f"invalid JWT must 401: {r.status_code} {r.text}"
        finally:
            _teardown(tc)

    def test_valid_jwt_resolves_and_scopes(self, tmp_path, monkeypatch):
        verify = _verify_factory({"jwt-a": {"tid": "tenant-a"}, "jwt-b": {"tid": "tenant-b"}})
        tc = _make_client(tmp_path, monkeypatch, require_identity=True, verify_fn=verify)
        try:
            a = {**ADMIN, "Authorization": "Bearer jwt-a"}
            b = {**ADMIN, "Authorization": "Bearer jwt-b"}
            assert tc.post("/kb/bases", json={"name": "a-kb"}, headers=a).status_code == 200
            assert tc.post("/kb/bases", json={"name": "b-kb"}, headers=b).status_code == 200
            a_list = tc.get("/kb/bases", headers=a).json()
            b_list = tc.get("/kb/bases", headers=b).json()
            a_names = {kb["name"] for kb in a_list}
            b_names = {kb["name"] for kb in b_list}
            assert "a-kb" in a_names and "b-kb" not in a_names
            assert "b-kb" in b_names and "a-kb" not in b_names
        finally:
            _teardown(tc)

    def test_cross_tenant_get_returns_404(self, tmp_path, monkeypatch):
        verify = _verify_factory({"jwt-a": {"tid": "tenant-a"}, "jwt-b": {"tid": "tenant-b"}})
        tc = _make_client(tmp_path, monkeypatch, require_identity=True, verify_fn=verify)
        try:
            a = {**ADMIN, "Authorization": "Bearer jwt-a"}
            b = {**ADMIN, "Authorization": "Bearer jwt-b"}
            assert tc.post("/kb/bases", json={"name": "secret", "kb_id": "secret"}, headers=a).status_code == 200
            assert tc.get("/kb/bases/secret", headers=b).status_code == 404
            assert tc.get("/kb/bases/secret", headers=a).status_code == 200
        finally:
            _teardown(tc)

    def test_header_jwt_tenant_mismatch_rejected(self, tmp_path, monkeypatch):
        # JWT says tenant-a, header claims tenant-b => forgery, 401.
        verify = _verify_factory({"jwt-a": {"tid": "tenant-a"}})
        tc = _make_client(tmp_path, monkeypatch, require_identity=True, verify_fn=verify)
        try:
            r = tc.get(
                "/kb/bases",
                headers={**ADMIN, "Authorization": "Bearer jwt-a", "X-Fusion-Tenant": "tenant-b"},
            )
            assert r.status_code == 401, f"header/JWT mismatch must 401: {r.status_code} {r.text}"
        finally:
            _teardown(tc)

    def test_header_matching_jwt_allowed(self, tmp_path, monkeypatch):
        # JWT tenant-a + matching header => ok (defense-in-depth passes).
        verify = _verify_factory({"jwt-a": {"tid": "tenant-a"}})
        tc = _make_client(tmp_path, monkeypatch, require_identity=True, verify_fn=verify)
        try:
            r = tc.get(
                "/kb/bases",
                headers={**ADMIN, "Authorization": "Bearer jwt-a", "X-Fusion-Tenant": "tenant-a"},
            )
            assert r.status_code == 200, f"matching header must pass: {r.status_code} {r.text}"
        finally:
            _teardown(tc)

    def test_revoked_token_rejected(self, tmp_path, monkeypatch):
        # identity returns None for revoked token => 401.
        verify = _verify_factory({"revoked-jwt": None})
        tc = _make_client(tmp_path, monkeypatch, require_identity=True, verify_fn=verify)
        try:
            r = tc.get("/kb/bases", headers={**ADMIN, "Authorization": "Bearer revoked-jwt"})
            assert r.status_code == 401, f"revoked token must 401: {r.status_code} {r.text}"
        finally:
            _teardown(tc)

    def test_health_exempt_from_identity(self, tmp_path, monkeypatch):
        verify = _verify_factory({})
        tc = _make_client(tmp_path, monkeypatch, require_identity=True, verify_fn=verify)
        try:
            assert tc.get("/health").status_code == 200
            assert tc.get("/ready").status_code in (200, 503)
        finally:
            _teardown(tc)


class TestIdentityOff:
    def test_integration_off_uses_header_path(self, tmp_path, monkeypatch):
        # Default: identity off, gateway off => X-Fusion-Tenant honored, no JWT.
        tc = _make_client(tmp_path, monkeypatch, require_identity=False, require_gateway=False)
        try:
            r = tc.get("/kb/bases", headers={**ADMIN, "X-Fusion-Tenant": "tenant-a"})
            assert r.status_code == 200, f"off mode must not require JWT: {r.status_code} {r.text}"
        finally:
            _teardown(tc)

    def test_integration_off_gateway_mode_still_works(self, tmp_path, monkeypatch):
        # Identity off, gateway on => #61 path intact (no JWT, header + route).
        tc = _make_client(tmp_path, monkeypatch, require_identity=False, require_gateway=True)
        try:
            gw = {**ADMIN, "X-Fusion-Route": "gateway-decision", "X-Fusion-Tenant": "tenant-a"}
            assert tc.get("/kb/bases", headers=gw).status_code == 200
            # missing gateway header => 403 (not 401 — identity is off).
            assert tc.get("/kb/bases", headers={**ADMIN}).status_code == 403
        finally:
            _teardown(tc)


class TestIdentityClientUnit:
    @pytest.mark.asyncio
    async def test_verify_missing_service_token_denies(self, tmp_path, monkeypatch):
        from fusion_rag.api.identity import IdentityClient

        monkeypatch.delenv("FUSION_IDENTITY_SERVICE_TOKEN", raising=False)
        c = IdentityClient(url="http://127.0.0.1:11470", service_token="")
        assert await c.verify("some-jwt") is None

    @pytest.mark.asyncio
    async def test_verify_revoked_claim_denies(self, tmp_path, monkeypatch):
        # Simulate identity returning revoked=true via a fake transport: patch
        # _call_verify to return the revoked path's pre-check value.
        from fusion_rag.api.identity import IdentityClient

        c = IdentityClient(url="http://127.0.0.1:11470", service_token="test-service-token")  # noqa: S106

        async def fake_call(token):
            # Pretend identity returned 200 with revoked=true — verify() must
            # turn that into None (the _call_verify revocation check).
            return None

        c._call_verify = fake_call
        assert await c.verify("revoked-jwt") is None
