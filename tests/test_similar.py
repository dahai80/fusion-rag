"""Tests for issue #83: /similar near-duplicate search endpoint."""

from __future__ import annotations

import tempfile
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from fusion_rag.api.server import create_app
from fusion_rag.embed.client import EmbeddingClient


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


def _ingest(client, kb_id: str, doc_name: str, content: str) -> str:
    resp = client.post(
        f"/kb/bases/{kb_id}/documents/ingest",
        json={"content": content, "doc_name": doc_name, "contextualize": False},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["doc_id"]


class TestSimilar:
    def test_text_query(self, client):
        kb_id = client.post("/kb/bases", json={"name": "sim-kb"}).json()["id"]
        doc_id = _ingest(client, kb_id, "reg-v1.txt", "排污许可证 编号 X123 pH 排放限值 6-9 悬浮物 30mg/L")

        resp = client.post(f"/kb/bases/{kb_id}/similar", json={"text": "排污许可证 X123", "top_k": 5})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["count"] >= 1
        m = body["matches"][0]
        assert m["doc_id"] == doc_id
        assert m["chunk_id"].startswith(doc_id)
        assert m["status"] == "active"
        assert "排污许可证" in m["snippet"]
        # float32 cosine can land a hair above 1.0
        assert 0.0 <= m["score"] <= 1.0 + 1e-6
        assert "threshold" in body

    def test_raw_embedding_query(self, client):
        kb_id = client.post("/kb/bases", json={"name": "sim-emb-kb"}).json()["id"]
        _ingest(client, kb_id, "reg-v2.txt", "环境监测数据 数值 42 报告")

        resp = client.post(
            f"/kb/bases/{kb_id}/similar",
            json={"embedding": [0.01] * 1024, "top_k": 5},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["count"] >= 1

    def test_exclude_doc_id(self, client):
        kb_id = client.post("/kb/bases", json={"name": "sim-excl-kb"}).json()["id"]
        doc_a = _ingest(client, kb_id, "a.txt", "环境监测数据 数值 42")
        _ingest(client, kb_id, "b.txt", "环境监测数据 数值 43")

        resp = client.post(
            f"/kb/bases/{kb_id}/similar",
            json={"text": "环境监测数据", "top_k": 10, "exclude_doc_id": doc_a},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["count"] >= 1
        assert all(m["doc_id"] != doc_a for m in body["matches"])

    def test_status_filter_respects_deprecation(self, client):
        kb_id = client.post("/kb/bases", json={"name": "sim-status-kb"}).json()["id"]
        doc_a = _ingest(client, kb_id, "old.txt", "环境监测数据 旧版本 数值 41")
        _ingest(client, kb_id, "new.txt", "环境监测数据 新版本 数值 42")

        resp = client.patch(
            f"/kb/bases/{kb_id}/documents/{doc_a}",
            json={"status": "deprecated", "superseded_by": "new.txt"},
        )
        assert resp.status_code == 200, resp.text

        # default filter drops the deprecated doc
        resp = client.post(f"/kb/bases/{kb_id}/similar", json={"text": "环境监测数据", "top_k": 10})
        assert resp.status_code == 200, resp.text
        assert all(m["doc_id"] != doc_a for m in resp.json()["matches"])

        # status_filter=all keeps it, marked deprecated
        resp = client.post(
            f"/kb/bases/{kb_id}/similar",
            json={"text": "环境监测数据", "top_k": 10, "status_filter": "all"},
        )
        assert resp.status_code == 200, resp.text
        matches = {m["doc_id"]: m for m in resp.json()["matches"]}
        assert doc_a in matches
        assert matches[doc_a]["status"] == "deprecated"

    def test_validation_errors(self, client):
        kb_id = client.post("/kb/bases", json={"name": "sim-val-kb"}).json()["id"]
        # neither text nor embedding
        resp = client.post(f"/kb/bases/{kb_id}/similar", json={})
        assert resp.status_code == 400
        # bad top_k
        resp = client.post(f"/kb/bases/{kb_id}/similar", json={"text": "x", "top_k": 0})
        assert resp.status_code == 400
        # bad threshold
        resp = client.post(f"/kb/bases/{kb_id}/similar", json={"text": "x", "threshold": 1.5})
        assert resp.status_code == 400
        # non-numeric embedding
        resp = client.post(f"/kb/bases/{kb_id}/similar", json={"embedding": ["a", "b"]})
        assert resp.status_code == 400
        # bad status_filter
        resp = client.post(f"/kb/bases/{kb_id}/similar", json={"text": "x", "status_filter": "bogus"})
        assert resp.status_code == 400
