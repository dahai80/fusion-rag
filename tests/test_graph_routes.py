"""Tests for the GraphRAG HTTP surface (issue #79): /kb/bases/{kb_id}/graph/*
— build, stats, search. Entity extraction is LLM-backed, so tests patch
GraphRAG.extract_entities with a deterministic fake; the graph store, routes,
chunk resolution, and evidence fields are all real."""

from __future__ import annotations

import tempfile
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from fusion_rag.api.server import create_app
from fusion_rag.embed.client import EmbeddingClient
from fusion_rag.engine.graph_rag import GraphRAG


def _fake_extraction(text: str) -> dict:
    # Deterministic stand-in for the LLM: the "fusion" entity + a relation.
    return {
        "entities": [
            {"name": "fusion", "type": "CONCEPT"},
            {"name": f"tag_{abs(hash(text)) % 100}", "type": "CONCEPT"},
        ],
        "relations": [{"source": "fusion", "target": f"tag_{abs(hash(text)) % 100}", "label": "mentions"}],
    }


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


@pytest.fixture
def kb_with_docs(client):
    kb_id = client.post("/kb/bases", json={"name": "graph-kb"}).json()["id"]
    for i in range(3):
        resp = client.post(
            f"/kb/bases/{kb_id}/documents/ingest",
            json={
                "content": f"fusion platform component {i} handles data processing for the fusion service",
                "doc_name": f"doc_{i}.txt",
                "contextualize": False,
            },
        )
        assert resp.status_code == 200, resp.text
    return kb_id


class TestGraphRoutes:
    def test_build_stats_search_roundtrip(self, client, kb_with_docs):
        kb_id = kb_with_docs
        with patch.object(GraphRAG, "extract_entities", new=AsyncMock(side_effect=lambda text: _fake_extraction(text))):
            resp = client.post(f"/kb/bases/{kb_id}/graph/build", json={})
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert body["status"] == "ok"
            assert body["chunks_processed"] == 3
            assert body["entities"] > 0
            assert body["relations"] > 0

        resp = client.get(f"/kb/bases/{kb_id}/graph/stats")
        assert resp.status_code == 200
        stats = resp.json()
        assert stats["entities"] == body["entities"]
        assert stats["relations"] == body["relations"]

        resp = client.post(f"/kb/bases/{kb_id}/graph/search", json={"query": "fusion"})
        assert resp.status_code == 200, resp.text
        rows = resp.json()
        assert rows, "expected graph hits for 'fusion'"
        for r in rows:
            assert r["text"], "hit chunk should resolve to its text"
            assert r["doc_id"]
            assert r["chunk_id"] == r["id"]
            assert r["entity_count"] > 0

    def test_build_empty_kb_rejected(self, client):
        kb_id = client.post("/kb/bases", json={"name": "empty-graph-kb"}).json()["id"]
        resp = client.post(f"/kb/bases/{kb_id}/graph/build", json={})
        assert resp.status_code == 400

    def test_build_unknown_kb_404(self, client):
        resp = client.post("/kb/bases/nope/graph/build", json={})
        assert resp.status_code == 404

    def test_search_requires_query(self, client, kb_with_docs):
        resp = client.post(f"/kb/bases/{kb_with_docs}/graph/search", json={})
        assert resp.status_code == 400

    def test_search_no_hits_empty_list(self, client, kb_with_docs):
        resp = client.post(f"/kb/bases/{kb_with_docs}/graph/search", json={"query": "zzz-no-such-entity-zzz"})
        assert resp.status_code == 200
        assert resp.json() == []

    def test_build_respects_max_chunks_cap(self, client, kb_with_docs):
        kb_id = kb_with_docs
        with patch.object(GraphRAG, "extract_entities", new=AsyncMock(side_effect=lambda text: _fake_extraction(text))):
            resp = client.post(f"/kb/bases/{kb_id}/graph/build", json={"max_chunks": 1})
            assert resp.status_code == 200
            assert resp.json()["chunks_processed"] == 1
