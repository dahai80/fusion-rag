"""Tests for issue #93: a rejected LLM credential must surface distinctly
(startup probe, /health llm_auth, /ask error message) instead of a generic
'upstream LLM unavailable'."""

from __future__ import annotations

import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from fusion_rag.api.server import create_app
from fusion_rag.embed.client import EmbeddingClient
from fusion_rag.engine.llm_auth_state import (
    get_llm_auth_status,
    key_hash_prefix,
    probe_llm_auth_sync,
    set_llm_auth_status,
)
from fusion_rag.engine.llm_errors import LLMAuthRejected


@pytest.fixture(autouse=True)
def _reset_auth_state():
    set_llm_auth_status("unknown")
    yield
    set_llm_auth_status("unknown")


@pytest.fixture
def client():
    app = create_app(kb_storage_dir=tempfile.mkdtemp())
    with (
        TestClient(app) as tc,
        patch.object(EmbeddingClient, "health", new_callable=AsyncMock, return_value=True),
        patch.object(
            EmbeddingClient,
            "embed_batch",
            new_callable=AsyncMock,
            side_effect=lambda texts: [[0.01] * 1024 for _ in texts],
        ),
        patch.object(EmbeddingClient, "embed", new_callable=AsyncMock, return_value=[0.01] * 1024),
    ):
        yield tc


class TestAuthStateHelpers:
    def test_set_and_get_status(self):
        set_llm_auth_status("ok")
        assert get_llm_auth_status() == "ok"
        set_llm_auth_status("rejected")
        assert get_llm_auth_status() == "rejected"

    def test_invalid_status_ignored(self):
        set_llm_auth_status("ok")
        set_llm_auth_status("bogus")
        assert get_llm_auth_status() == "ok"

    def test_key_hash_prefix_never_returns_key(self):
        key = "fg-super-secret-48-char-key"
        prefix = key_hash_prefix(key)
        assert len(prefix) == 8
        assert key not in prefix
        assert key_hash_prefix("") == "(none)"


class TestProbeSync:
    def test_probe_401_marks_rejected(self, monkeypatch):
        monkeypatch.setattr(httpx, "get", lambda *a, **k: httpx.Response(401))
        assert probe_llm_auth_sync("http://gw:11432/v1", "k") == "rejected"
        assert get_llm_auth_status() == "rejected"

    def test_probe_200_marks_ok(self, monkeypatch):
        monkeypatch.setattr(httpx, "get", lambda *a, **k: httpx.Response(200))
        assert probe_llm_auth_sync("http://gw:11432/v1", "k") == "ok"
        assert get_llm_auth_status() == "ok"

    def test_probe_unreachable_marks_unknown_not_rejected(self, monkeypatch):
        def _raise(*a, **k):
            raise httpx.ConnectError("down")

        monkeypatch.setattr(httpx, "get", _raise)
        assert probe_llm_auth_sync("http://gw:11432/v1", "k") == "unknown"
        assert get_llm_auth_status() == "unknown"


class _FakeResp:
    def __init__(self, status_code: int):
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            req = httpx.Request("POST", "http://gw/v1/chat/completions")
            raise httpx.HTTPStatusError(
                f"client error {self.status_code}",
                request=req,
                response=httpx.Response(self.status_code, request=req),
            )

    def json(self):
        return {"choices": [{"message": {"content": "answer"}}]}


class TestGenerateAnswerAuthRejected:
    def test_401_raises_auth_rejected_and_flags_state(self):
        from fusion_rag.api import routes

        fake_embed = SimpleNamespace(base_url="http://gw/v1", api_key="fg-stale")
        fake_client = SimpleNamespace(post=AsyncMock(return_value=_FakeResp(401)))

        async def _fake_retry(fn, **kwargs):
            return await fn()

        with (
            patch.object(routes, "_get_embed_client", return_value=fake_embed),
            patch.object(routes, "get_async_client", return_value=fake_client),
            patch.object(routes, "with_retry", _fake_retry),
            pytest.raises(LLMAuthRejected),
        ):
            import asyncio

            asyncio.run(routes._generate_answer("q", "ctx", []))
        assert get_llm_auth_status() == "rejected"

    def test_500_still_raises_generic_unavailable(self):
        from fusion_rag.api import routes
        from fusion_rag.engine.llm_errors import LLMUnavailable

        fake_embed = SimpleNamespace(base_url="http://gw/v1", api_key="fg-key")
        fake_client = SimpleNamespace(post=AsyncMock(return_value=_FakeResp(500)))

        async def _fake_retry(fn, **kwargs):
            return await fn()

        with (
            patch.object(routes, "_get_embed_client", return_value=fake_embed),
            patch.object(routes, "get_async_client", return_value=fake_client),
            patch.object(routes, "with_retry", _fake_retry),
            pytest.raises(LLMUnavailable) as exc_info,
        ):
            import asyncio

            asyncio.run(routes._generate_answer("q", "ctx", []))
        assert not isinstance(exc_info.value, LLMAuthRejected)
        assert get_llm_auth_status() == "unknown"


class TestAskAndHealthSurfaces:
    def test_ask_maps_auth_rejection_distinctly(self, client):
        kb_id = client.post("/kb/bases", json={"name": "auth-kb"}).json()["id"]
        client.post(
            f"/kb/bases/{kb_id}/documents/ingest",
            json={
                "content": "外墙防水施工规范 waterproof spec",
                "content_type": "text",
                "doc_name": "spec.md",
                "contextualize": False,
            },
        )

        async def _rejected(*a, **k):
            raise LLMAuthRejected()

        with patch("fusion_rag.api.routes._generate_answer", _rejected):
            resp = client.post(
                f"/kb/bases/{kb_id}/ask",
                json={"question": "外墙防水怎么做"},
            )
        assert resp.status_code == 502
        assert "credential rejected" in resp.json()["detail"]
        assert "unavailable" not in resp.json()["detail"]

    def test_ask_generic_unavailable_unchanged(self, client):
        from fusion_rag.engine.llm_errors import LLMUnavailable

        kb_id = client.post("/kb/bases", json={"name": "auth-kb2"}).json()["id"]
        client.post(
            f"/kb/bases/{kb_id}/documents/ingest",
            json={
                "content": "外墙防水施工规范 waterproof spec",
                "content_type": "text",
                "doc_name": "spec.md",
                "contextualize": False,
            },
        )

        async def _unavailable(*a, **k):
            raise LLMUnavailable()

        with patch("fusion_rag.api.routes._generate_answer", _unavailable):
            resp = client.post(
                f"/kb/bases/{kb_id}/ask",
                json={"question": "外墙防水怎么做"},
            )
        assert resp.status_code == 502
        assert "upstream LLM unavailable" in resp.json()["detail"]

    def test_health_exposes_llm_auth_field(self, client):
        set_llm_auth_status("rejected")
        body = client.get("/health").json()
        assert body["llm_auth"] == "rejected"
        assert body["status"] == "ok"
