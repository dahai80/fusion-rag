"""Audit-3 P0/P1 security regression tests: privilege-escalation chain fixed
by S-P0-1/S-P0-2/S-P1-1/S-P1-2/S-P1-3.

Chain (pre-fix): create_api_key accepted name="admin" -> verify() returned
"admin" -> access.py ACL bypass -> full admin reach via a stored key. Three
mutation surfaces (key mgmt, permission rules) had no admin-only gate; MCP
tools/call skipped per-KB ACL entirely. A subtree-restricted subject could
read/upload any KB via /mcp.

Fixes locked here:
- S-P0-1: stored key named "admin" rejected (add_key returns False, route 400).
- S-P1-3: non-admin subject -> 403 on GET/POST/DELETE /kb/auth/keys.
- S-P1-1/S-P1-2: non-admin subject -> 403 on POST/DELETE /kb/.../permissions.
- S-P0-2: MCP tools/call enforces per-KB ACL; restricted subject denied
  kb_search on a KB with path-scoped rules.
"""
from __future__ import annotations

import tempfile
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from fusion_rag.api.server import create_app


async def _fake_embed_batch(self, texts):
    # fusion-mlx has no MLX embedding model -> real embed_batch 502s. These
    # tests assert ACL gates, not embedding. Return deterministic 1024-dim
    # vectors (BGE-M3 dim) so the store write path runs for real.
    return [[0.01] * 1024 for _ in texts]


class TestSecurityPrivEsc:
    @pytest.fixture
    def client(self, tmp_path):
        # create_app has no admin_api_key param; admin key comes from the
        # injected auth backend. Fresh ApiKeyBackend + temp auth.db so the
        # subject keys added here do NOT pollute the real ~/.fusion-rag/auth.db.
        from fusion_rag.api import auth as auth_mod
        from fusion_rag.embed.client import EmbeddingClient

        backend = auth_mod.ApiKeyBackend(admin_key="admin-key", db_path=str(tmp_path / "auth.db"))
        auth_mod._auth_backend = backend
        storage_dir = tempfile.mkdtemp()
        app = create_app(kb_storage_dir=storage_dir)
        with patch.object(EmbeddingClient, "embed_batch", _fake_embed_batch), TestClient(app) as tc:
            tc.kb_storage_dir = storage_dir
            yield tc
        auth_mod._auth_backend = None  # reset singleton for the rest of the suite

    def test_stored_key_cannot_be_named_admin(self, client):
        # S-P0-1: the admin identity is the env key only. A stored key named
        # "admin" would authenticate as admin (verify returns the name) and
        # bypass ACL. The create route must 400 and add_key must reject.
        from fusion_rag.api.auth import AuthConfig

        # Route path: admin tries to mint a stored key named "admin" -> 400.
        r = client.post("/kb/auth/keys", json={"name": "admin"}, headers={"X-API-Key": "admin-key"})
        assert r.status_code == 400, f"reserved name must be rejected: {r.status_code} {r.text}"

        # Store path: add_key directly returns False for a reserved name.
        assert AuthConfig().add_key("frg_reserved", "admin") is False

    def test_non_admin_cannot_list_create_delete_keys(self, client):
        # S-P1-3: key management is admin-only. Register a non-admin subject
        # "alice" via the backend, then assert each key route denies her.
        from fusion_rag.api.auth import get_auth_backend

        get_auth_backend().auth_config.add_key("alice-key", "alice")
        alice = {"X-API-Key": "alice-key"}

        r = client.get("/kb/auth/keys", headers=alice)
        assert r.status_code == 403, f"alice must not list keys: {r.status_code} {r.text}"

        r = client.post("/kb/auth/keys", json={"name": "bob"}, headers=alice)
        assert r.status_code == 403, f"alice must not create keys: {r.status_code} {r.text}"

        r = client.delete("/kb/auth/keys/deadbeef", headers=alice)
        assert r.status_code == 403, f"alice must not delete keys: {r.status_code} {r.text}"

    def test_non_admin_cannot_mutate_permissions(self, client):
        # S-P1-1/S-P1-2: permission-rule mutation is admin-only. A non-admin
        # subject with write/delete rights on an OPEN kb still must not add or
        # delete ACL rules (those define who can reach what).
        from fusion_rag.api.auth import get_auth_backend

        admin = {"X-API-Key": "admin-key"}
        create = client.post("/kb/bases", json={"name": "pekb", "kb_id": "pekb"}, headers=admin).json()
        kb_id = create["id"]

        get_auth_backend().auth_config.add_key("alice-key", "alice")
        alice = {"X-API-Key": "alice-key"}

        rule = {
            "subject": "alice",
            "resource_type": "document",
            "resource_path": "/",
            "actions": ["read", "write", "delete"],
        }
        r = client.post(f"/kb/bases/{kb_id}/permissions", json=rule, headers=alice)
        assert r.status_code == 403, f"alice must not add rules: {r.status_code} {r.text}"

        # Seed a rule as admin, then assert alice cannot delete it.
        seeded = client.post(f"/kb/bases/{kb_id}/permissions", json=rule, headers=admin).json()
        rule_id = seeded.get("id", "")
        r = client.delete(f"/kb/bases/{kb_id}/permissions/{rule_id}", headers=alice)
        assert r.status_code == 403, f"alice must not delete rules: {r.status_code} {r.text}"

    def test_mcp_tools_call_enforces_per_kb_acl(self, client):
        # S-P0-2: MCP tools/call previously called only verify() and skipped
        # the per-KB ACL gate REST enforces. A subtree-restricted subject could
        # read any KB via /mcp kb_search. With the fix, kb_search runs
        # _check_kb_access -> 403-equivalent JSON-RPC error on denial.
        from fusion_rag.api.auth import get_auth_backend

        admin = {"X-API-Key": "admin-key"}
        create = client.post("/kb/bases", json={"name": "mcpkb", "kb_id": "mcpkb"}, headers=admin).json()
        kb_id = create["id"]

        # Ingest an inline doc so the KB is non-empty.
        client.post(
            f"/kb/bases/{kb_id}/documents/ingest",
            json={"content": "mcp acl body", "content_type": "text", "doc_name": "secret", "contextualize": False},
            headers=admin,
        )

        # Scope subject "alice" to a path she did NOT ingest under, so a
        # kb_search (resource_path="/") is out of her subtree -> denied.
        get_auth_backend().auth_config.add_key("alice-key", "alice")
        client.post(
            f"/kb/bases/{kb_id}/permissions",
            json={
                "subject": "alice",
                "resource_type": "document",
                "resource_path": "inline://other",
                "actions": ["read"],
            },
            headers=admin,
        )

        # MCP initialize is open; tools/call must enforce ACL.
        init = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        assert init.status_code == 200

        alice = {"X-API-Key": "alice-key"}
        r = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "kb_search", "arguments": {"kb_id": kb_id, "query": "secret"}},
            },
            headers=alice,
        )
        # ACL denial surfaces as a JSON-RPC error (code -32001) with 403 status.
        assert r.status_code == 403, f"alice must be denied MCP kb_search: {r.status_code} {r.text}"
        body = r.json()
        assert "error" in body, f"expected JSON-RPC error, got: {body}"
