"""Issue #61 — gateway-origin + tenant scoping regression tests.

Locks the multi-tenant isolation contract:
- FUSION_RAG_REQUIRE_GATEWAY=on rejects /kb/* without X-Fusion-Route: gateway-decision (403).
- X-Fusion-Tenant is the authoritative tenant: list/get scoped to it.
- A tenant-A caller cannot see/address tenant-B's KBs (404, not 403 — no existence leak).
- Default (isolation off) = zero behavior change: all KBs visible, no header required.
- Health/ready/metrics/MCP/auth routes exempt from the gateway-origin gate.
"""
from __future__ import annotations

import tempfile
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from fusion_rag.api.server import create_app


async def _fake_embed_batch(self, texts):
    return [[0.01] * 1024 for _ in texts]


def _make_client(tmp_path, require_gateway=False, monkeypatch=None):
    from fusion_rag.api import auth as auth_mod
    from fusion_rag.embed.client import EmbeddingClient

    if monkeypatch is not None:
        if require_gateway:
            monkeypatch.setenv("FUSION_RAG_REQUIRE_GATEWAY", "1")
        else:
            monkeypatch.delenv("FUSION_RAG_REQUIRE_GATEWAY", raising=False)
    auth_mod._auth_backend = auth_mod.ApiKeyBackend(admin_key="admin-key", db_path=str(tmp_path / "auth.db"))
    storage_dir = tempfile.mkdtemp()
    app = create_app(kb_storage_dir=storage_dir)
    patcher = patch.object(EmbeddingClient, "embed_batch", _fake_embed_batch)
    patcher.start()
    tc_cm = TestClient(app)
    # Enter as context manager so the lifespan startup runs and app.state is
    # bound (init_app_state populates kb_manager/embed_client on app.state).
    tc_cm.__enter__()
    tc_cm.kb_storage_dir = storage_dir
    tc_cm._patcher = patcher
    return tc_cm


def _teardown(tc):
    import contextlib

    with contextlib.suppress(Exception):
        tc.__exit__(None, None, None)
    tc._patcher.stop()
    from fusion_rag.api import auth as auth_mod

    auth_mod._auth_backend = None
    from fusion_rag.api.tenant import reset_request_tenant

    reset_request_tenant()


GW = {"X-Fusion-Route": "gateway-decision"}
ADMIN = {"X-API-Key": "admin-key"}


class TestGatewayOriginEnforcement:
    def test_kb_request_without_gateway_header_rejected(self, tmp_path, monkeypatch):
        tc = _make_client(tmp_path, require_gateway=True, monkeypatch=monkeypatch)
        try:
            r = tc.get("/kb/bases", headers={**ADMIN})
            assert r.status_code == 403, f"non-gateway /kb must 403: {r.status_code} {r.text}"
        finally:
            _teardown(tc)

    def test_kb_request_with_gateway_header_allowed(self, tmp_path, monkeypatch):
        tc = _make_client(tmp_path, require_gateway=True, monkeypatch=monkeypatch)
        try:
            r = tc.get("/kb/bases", headers={**ADMIN, **GW})
            assert r.status_code == 200, f"gateway /kb must pass: {r.status_code} {r.text}"
        finally:
            _teardown(tc)

    def test_health_exempt_from_gateway_gate(self, tmp_path, monkeypatch):
        tc = _make_client(tmp_path, require_gateway=True, monkeypatch=monkeypatch)
        try:
            r = tc.get("/health")
            assert r.status_code == 200, f"/health must stay open: {r.status_code} {r.text}"
        finally:
            _teardown(tc)

    def test_ready_exempt_from_gateway_gate(self, tmp_path, monkeypatch):
        tc = _make_client(tmp_path, require_gateway=True, monkeypatch=monkeypatch)
        try:
            r = tc.get("/ready")
            assert r.status_code in (200, 503), f"/ready must not be gateway-gated: {r.status_code} {r.text}"
        finally:
            _teardown(tc)

    def test_isolation_off_no_header_required(self, tmp_path, monkeypatch):
        tc = _make_client(tmp_path, require_gateway=False, monkeypatch=monkeypatch)
        try:
            r = tc.get("/kb/bases", headers={**ADMIN})
            assert r.status_code == 200, f"default mode must not require header: {r.status_code} {r.text}"
        finally:
            _teardown(tc)


class TestTenantScoping:
    def test_list_scoped_to_tenant(self, tmp_path, monkeypatch):
        tc = _make_client(tmp_path, require_gateway=True, monkeypatch=monkeypatch)
        try:
            tenant_a = {"X-API-Key": "admin-key", "X-Fusion-Route": "gateway-decision", "X-Fusion-Tenant": "tenant-a"}
            tenant_b = {"X-API-Key": "admin-key", "X-Fusion-Route": "gateway-decision", "X-Fusion-Tenant": "tenant-b"}
            r = tc.post("/kb/bases", json={"name": "a-kb"}, headers=tenant_a)
            assert r.status_code == 200, r.text
            r = tc.post("/kb/bases", json={"name": "b-kb"}, headers=tenant_b)
            assert r.status_code == 200, r.text
            a_list = tc.get("/kb/bases", headers=tenant_a).json()
            b_list = tc.get("/kb/bases", headers=tenant_b).json()
            a_names = {kb["name"] for kb in a_list}
            b_names = {kb["name"] for kb in b_list}
            assert "a-kb" in a_names and "b-kb" not in a_names, f"tenant-a leaked b-kb: {a_names}"
            assert "b-kb" in b_names and "a-kb" not in b_names, f"tenant-b leaked a-kb: {b_names}"
        finally:
            _teardown(tc)

    def test_cross_tenant_get_returns_404(self, tmp_path, monkeypatch):
        tc = _make_client(tmp_path, require_gateway=True, monkeypatch=monkeypatch)
        try:
            tenant_a = {"X-API-Key": "admin-key", "X-Fusion-Route": "gateway-decision", "X-Fusion-Tenant": "tenant-a"}
            tenant_b = {"X-API-Key": "admin-key", "X-Fusion-Route": "gateway-decision", "X-Fusion-Tenant": "tenant-b"}
            r = tc.post("/kb/bases", json={"name": "secret-kb", "kb_id": "secret-kb"}, headers=tenant_a)
            assert r.status_code == 200, r.text
            r = tc.get("/kb/bases/secret-kb", headers=tenant_b)
            assert r.status_code == 404, f"cross-tenant get must 404: {r.status_code} {r.text}"
            r = tc.get("/kb/bases/secret-kb", headers=tenant_a)
            assert r.status_code == 200, f"same-tenant get must pass: {r.status_code} {r.text}"
        finally:
            _teardown(tc)

    def test_tenant_stamped_on_create(self, tmp_path, monkeypatch):
        tc = _make_client(tmp_path, require_gateway=True, monkeypatch=monkeypatch)
        try:
            tenant_a = {"X-API-Key": "admin-key", "X-Fusion-Route": "gateway-decision", "X-Fusion-Tenant": "tenant-a"}
            r = tc.post("/kb/bases", json={"name": "stamped-kb"}, headers=tenant_a)
            kb_id = r.json()["id"]
            kb = tc.get(f"/kb/bases/{kb_id}", headers=tenant_a).json()
            assert kb.get("tenant") == "tenant-a", f"tenant not stamped: {kb.get('tenant')}"
        finally:
            _teardown(tc)

    def test_invalid_tenant_charset_treated_as_none(self, tmp_path, monkeypatch):
        tc = _make_client(tmp_path, require_gateway=True, monkeypatch=monkeypatch)
        try:
            bad = {"X-API-Key": "admin-key", "X-Fusion-Route": "gateway-decision", "X-Fusion-Tenant": "../etc"}
            r = tc.get("/kb/bases", headers=bad)
            assert r.status_code == 200, r.text
            assert r.json() == [], f"invalid tenant must not error: {r.text}"
        finally:
            _teardown(tc)


class TestTenantUnit:
    def test_normalize_tenant_valid(self):
        from fusion_rag.api.tenant import _normalize_tenant

        assert _normalize_tenant("tenant-a") == "tenant-a"
        assert _normalize_tenant("  tenant-b  ") == "tenant-b"
        assert _normalize_tenant("tenant:c0.re") == "tenant:c0.re"

    def test_normalize_tenant_invalid(self):
        from fusion_rag.api.tenant import _normalize_tenant

        assert _normalize_tenant(None) is None
        assert _normalize_tenant("") is None
        assert _normalize_tenant("   ") is None
        assert _normalize_tenant("../etc") is None
        assert _normalize_tenant("a" * 200) is None

    def test_tenant_scope_off_when_not_enabled(self, monkeypatch):
        from fusion_rag.api.tenant import reset_request_tenant, tenant_scope

        monkeypatch.delenv("FUSION_RAG_REQUIRE_GATEWAY", raising=False)
        reset_request_tenant()
        assert tenant_scope() == (None, False)

    def test_kb_manager_tenant_filter(self, tmp_path):
        from fusion_rag.engine.knowledge_base import KnowledgeBaseManager

        mgr = KnowledgeBaseManager(storage_dir=str(tmp_path))
        mgr.create(name="a", kb_id="ka", tenant="tenant-a")
        mgr.create(name="b", kb_id="kb", tenant="tenant-b")
        mgr.create(name="u", kb_id="ku", tenant=None)
        a_list = mgr.list(tenant="tenant-a", require_tenant_match=True)
        assert {kb["id"] for kb in a_list} == {"ka"}, f"tenant-a filter wrong: {a_list}"
        all_list = mgr.list()
        assert {kb["id"] for kb in all_list} == {"ka", "kb", "ku"}, f"unfiltered list wrong: {all_list}"
        assert mgr.get("ka", tenant="tenant-a", require_tenant_match=True).id == "ka"
        with pytest.raises(KeyError):
            mgr.get("ka", tenant="tenant-b", require_tenant_match=True)
