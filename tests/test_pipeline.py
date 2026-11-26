"""Issue #163: retrieval-pipeline stats endpoint (/kb/bases/{kb_id}/pipeline).

Live counters (docs/chunks/queries) + per-stage latency assembled from the RED
metrics registry. Uninstrumented stages return null stats — never invented."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from fusion_rag.api.server import create_app
from fusion_rag.embed.client import EmbeddingClient
from fusion_rag.engine.metrics import get_metrics


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("FUSION_RAG_DATA_DIR", str(tmp_path))
    get_metrics().reset()
    with patch.object(
        EmbeddingClient,
        "embed_batch",
        new_callable=AsyncMock,
        side_effect=lambda texts: [[0.01] * 1024 for _ in texts],
    ), patch.object(
        EmbeddingClient, "embed", new_callable=AsyncMock, return_value=[0.01] * 1024
    ), TestClient(create_app()) as tc:
        yield tc


def _kb_with_doc(client, tmp_path):
    doc = tmp_path / "doc.txt"
    doc.write_text("alpha beta gamma delta epsilon zeta", encoding="utf-8")
    kb_id = client.post("/kb/bases", json={"name": "pipe-kb"}).json()["id"]
    resp = client.post(f"/kb/bases/{kb_id}/documents", json={"file_path": str(doc)})
    assert resp.status_code == 200, resp.text
    return kb_id


class TestPipelineEndpoint:
    def test_basic_shape_and_counters(self, client, tmp_path):
        kb_id = _kb_with_doc(client, tmp_path)
        resp = client.get(f"/kb/bases/{kb_id}/pipeline")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["kb_id"] == kb_id
        assert body["docs"] == 1
        assert body["chunks"] > 0
        assert body["queries"] == 0

    def test_query_counter_counts_search(self, client, tmp_path):
        kb_id = _kb_with_doc(client, tmp_path)
        for _ in range(2):
            assert client.post(
                f"/kb/bases/{kb_id}/search", json={"query": "alpha", "top_k": 3}
            ).status_code == 200
        body = client.get(f"/kb/bases/{kb_id}/pipeline").json()
        assert body["queries"] == 2
        # retrieve stage latency recorded from the search endpoint histogram
        retrieve = next(s for s in body["stages"] if s["id"] == "retrieve")
        assert retrieve["calls"] == 2
        assert retrieve["avg_ms"] is not None and retrieve["avg_ms"] >= 0

    def test_uninstrumented_stages_null_not_invented(self, client, tmp_path):
        kb_id = _kb_with_doc(client, tmp_path)
        body = client.get(f"/kb/bases/{kb_id}/pipeline").json()
        for sid in ("rewrite", "filter"):
            stage = next(s for s in body["stages"] if s["id"] == sid)
            assert stage["calls"] is None and stage["avg_ms"] is None

    def test_llm_stage_from_metrics(self, client, tmp_path):
        kb_id = _kb_with_doc(client, tmp_path)
        get_metrics().record_llm_latency("generate", 0.5)
        get_metrics().record_llm_latency("generate", 1.5)
        body = client.get(f"/kb/bases/{kb_id}/pipeline").json()
        gen = next(s for s in body["stages"] if s["id"] == "generate")
        assert gen["calls"] == 2
        assert gen["avg_ms"] == 1.0

    def test_counts_not_cross_kb(self, client, tmp_path):
        kb_id = _kb_with_doc(client, tmp_path)
        other = client.post("/kb/bases", json={"name": "other"}).json()["id"]
        assert client.post(
            f"/kb/bases/{kb_id}/search", json={"query": "alpha", "top_k": 3}
        ).status_code == 200
        assert client.get(f"/kb/bases/{other}/pipeline").json()["queries"] == 0

    def test_unknown_kb_404(self, client):
        assert client.get("/kb/bases/nope/pipeline").status_code == 404

    def test_stage_chain_matches_prototype(self, client, tmp_path):
        kb_id = _kb_with_doc(client, tmp_path)
        body = client.get(f"/kb/bases/{kb_id}/pipeline").json()
        assert [s["id"] for s in body["stages"]] == [
            "rewrite", "retrieve", "rerank", "filter", "generate", "contextualize",
        ]
