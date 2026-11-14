# Issue #136: MCP kb_search / kb_ask treat kb_id as optional — default to the
# single KB, or the most recently created when several exist.

from __future__ import annotations

import pytest

from fusion_rag.api.mcp_server import _dispatch_tool, _enforce_mcp_acl, _resolve_kb_id


class _FakeKB:
    def __init__(self, id: str, created_at: float):
        self.id = id
        self.created_at = created_at


class _FakeManager:
    def __init__(self, kbs):
        self.kbs = kbs

    def list(self, tenant=None, require_tenant_match=False):
        return self.kbs


@pytest.fixture
def patch_manager(monkeypatch):
    def _install(kbs):
        monkeypatch.setattr(
            "fusion_rag.api.routes._get_kb_manager", lambda: _FakeManager(kbs)
        )
    return _install


class TestResolveKbId:
    def test_explicit_kb_id_wins(self, patch_manager):
        patch_manager([_FakeKB("other", 1.0)])
        assert _resolve_kb_id({"kb_id": "mine"}) == "mine"

    def test_single_kb_default(self, patch_manager):
        patch_manager([_FakeKB("only", 1.0)])
        assert _resolve_kb_id({}) == "only"

    def test_multiple_kbs_most_recent(self, patch_manager):
        patch_manager([_FakeKB("old", 100.0), _FakeKB("new", 200.0)])
        assert _resolve_kb_id({}) == "new"

    def test_no_kbs_raises(self, patch_manager):
        patch_manager([])
        with pytest.raises(Exception, match="no knowledge bases exist"):
            _resolve_kb_id({})


class TestDispatchDefaults:
    async def test_kb_search_without_kb_id(self, patch_manager, monkeypatch):
        patch_manager([_FakeKB("only", 1.0)])

        class _Store:
            def search(self, vec, top_k=5):
                return [{"id": "c1", "text": "hit"}]

        fake_kb = type("KB", (), {"config": type("C", (), {"max_results": 5})})
        monkeypatch.setattr("fusion_rag.api.routes._get_base", lambda kb_id: fake_kb)
        monkeypatch.setattr("fusion_rag.api.routes._get_vector_store", lambda kb_id: _Store())

        async def _fake_embed(query):
            return [0.1] * 8

        class _Embed:
            async def embed(self, query):
                return [0.1] * 8

        monkeypatch.setattr("fusion_rag.api.routes._get_embed_client", lambda: _Embed())
        out = await _dispatch_tool("kb_search", {"query": "q"})
        assert out[0]["id"] == "c1"


class TestAclDefault:
    def test_acl_resolves_default_kb(self, patch_manager, monkeypatch):
        patch_manager([_FakeKB("only", 1.0)])
        seen = {}

        def _fake_check(kb_id, subject, action, path):
            seen["kb_id"] = kb_id

        monkeypatch.setattr("fusion_rag.api.access._check_kb_access", _fake_check)
        _enforce_mcp_acl("kb_search", {}, None)
        assert seen["kb_id"] == "only"
