"""Console UI — static RAG console served at /shell/fusion-rag/ (+ /console alias).

The path matches the web-shell module convention so a gateway proxy of
/shell/fusion-rag/ -> :11436 maps 1:1; the page uses relative API urls so it
works identically direct and behind the gateway.
"""

from __future__ import annotations

import tempfile
from unittest.mock import patch

from fastapi.testclient import TestClient

from fusion_rag.api.server import create_app
from fusion_rag.embed.client import EmbeddingClient


async def _fake_embed_batch(self, texts):
    return [[0.01] * 1024 for _ in texts]


async def _fake_health(self):
    return True


def _client():
    app = create_app(kb_storage_dir=tempfile.mkdtemp())
    p1 = patch.object(EmbeddingClient, "embed_batch", _fake_embed_batch)
    p2 = patch.object(EmbeddingClient, "health", _fake_health)
    p1.start()
    p2.start()
    return TestClient(app), p1, p2


def test_console_served_at_shell_path():
    tc, p1, p2 = _client()
    try:
        with tc:
            resp = tc.get("/shell/fusion-rag/")
            assert resp.status_code == 200
            assert "Fusion-RAG" in resp.text
            assert resp.headers["content-type"].startswith("text/html")
    finally:
        p1.stop()
        p2.stop()


def test_console_alias_redirects():
    tc, p1, p2 = _client()
    try:
        with tc:
            resp = tc.get("/console", follow_redirects=False)
            assert resp.status_code == 307
            assert resp.headers["location"] == "/shell/fusion-rag/"
    finally:
        p1.stop()
        p2.stop()


def test_console_page_uses_relative_api_urls():
    # Issue #109: the page must derive its API base from location.pathname —
    # an origin-root fetch("/kb/...") 404s behind the gateway pass-through
    # proxy (origin :11432 only routes /shell/fusion-rag/*).
    from importlib import resources

    html = (resources.files("fusion_rag") / "console" / "index.html").read_text(encoding="utf-8")
    assert "11436" not in html
    assert "API_BASE" in html
    assert "location.pathname" in html
    assert 'fetch("/' not in html  # no origin-root-relative fetch survives


def test_kb_api_served_under_console_subpath():
    # Issue #109: page-relative paths resolve to /shell/fusion-rag/kb/... —
    # the /kb router must answer there too, with the SAME handlers.
    import tempfile
    from unittest.mock import patch

    from fusion_rag.api.server import create_app
    from fusion_rag.embed.client import EmbeddingClient

    async def _fake_embed_batch(self, texts):
        return [[0.01] * 1024 for _ in texts]

    async def _fake_health(self):
        return True

    app = create_app(kb_storage_dir=tempfile.mkdtemp())
    p1 = patch.object(EmbeddingClient, "embed_batch", _fake_embed_batch)
    p2 = patch.object(EmbeddingClient, "health", _fake_health)
    p1.start()
    p2.start()
    try:
        with TestClient(app) as tc:
            tc.post("/kb/bases", json={"kb_id": "sub109", "name": "t"})
            # root path still works...
            assert any(kb["id"] == "sub109" for kb in tc.get("/kb/bases").json())
            # ...and the console subpath serves the SAME data.
            resp = tc.get("/shell/fusion-rag/kb/bases")
            assert resp.status_code == 200
            assert any(kb["id"] == "sub109" for kb in resp.json())
            # health pill path.
            h = tc.get("/shell/fusion-rag/health")
            assert h.status_code == 200
            assert h.json()["service"] == "fusion-rag"
            # static page still wins at the subpath root.
            page = tc.get("/shell/fusion-rag/")
            assert page.status_code == 200
            assert "Fusion-RAG" in page.text
    finally:
        p1.stop()
        p2.stop()
