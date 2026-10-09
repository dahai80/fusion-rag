"""Tests for BNUP corpus ingestion + retrieval API (issue #74)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from fusion_rag.api.server import create_app
from fusion_rag.bnup.corpus import BnupCorpus, derive_semester
from fusion_rag.embed.client import EmbeddingClient

_FIXTURE = Path(__file__).parent / "fixtures" / "beishi-math-g1-6-mini.json"


def _load_fixture() -> dict:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


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
    ):
        yield tc


# ── corpus parser (pure logic) ──


class TestBnupCorpus:
    def test_derive_semester(self):
        assert derive_semester(1, 4) == 1
        assert derive_semester(2, 4) == 1
        assert derive_semester(3, 4) == 2
        assert derive_semester(4, 4) == 2
        assert derive_semester(1, 0) == 1

    def test_parse_lessons(self):
        c = BnupCorpus(_load_fixture())
        lessons = c.lessons()
        assert len(lessons) == 15
        first = lessons[0]
        assert first["doc_id"] == "bnup_g1_u1_l1"
        assert first["doc_path"] == "bnup/g1/u1/l1"
        assert first["doc_type"] == "bnup_lesson"
        m = first["metadata"]
        assert m["edition"] == "bnup"
        assert m["grade"] == 1
        assert m["unit"] == 1
        assert m["lesson"] == 1
        assert "math-g1-na-01" in m["knowledge_point_ids"]
        assert m["semester"] == 1

    def test_semester_split(self):
        c = BnupCorpus(_load_fixture())
        lessons = c.lessons()
        g1 = [d for d in lessons if d["metadata"]["grade"] == 1]
        sems = {d["metadata"]["semester"] for d in g1}
        assert sems == {1, 2}

    def test_knowledge_graph(self):
        c = BnupCorpus(_load_fixture())
        kg = c.knowledge_graph()
        assert kg["edition"] == "beishi"
        assert "1" in kg["grades"]
        assert kg["knowledge_point_count"] > 0
        assert "math-g1-na-01" in kg["knowledge_points"]
        kp = kg["knowledge_points"]["math-g1-na-01"]
        assert kp["grade"] == 1
        assert kp["strand"] == "na"

    def test_stats(self):
        c = BnupCorpus(_load_fixture())
        s = c.stats()
        assert s["total_lessons"] == 15
        assert s["per_grade"]["1"]["lessons"] == 8
        assert s["per_grade"]["2"]["lessons"] == 7
        assert s["knowledge_point_count"] > 0

    def test_invalid_corpus(self):
        with pytest.raises(ValueError):
            BnupCorpus({"no_grades": True})
        with pytest.raises(ValueError):
            BnupCorpus("not a dict")  # type: ignore[arg-type]


# ── routes ──


class TestBnupRoutes:
    def test_retrieve_before_ingest_404(self, client):
        resp = client.get("/api/v1/bnup/retrieve", params={"knowledge_point": "math-g1-na-01"})
        assert resp.status_code == 404

    def test_ingest_and_retrieve(self, client):
        corpus = _load_fixture()
        resp = client.post("/api/v1/bnup/ingest", json=corpus)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "ok"
        assert body["lessons_ingested"] == 15
        assert body["edition"] == "bnup"

        resp = client.get("/api/v1/bnup/retrieve", params={"knowledge_point": "math-g1-na-01", "limit": 5})
        assert resp.status_code == 200
        results = resp.json()
        assert isinstance(results, list)

    def test_retrieve_by_grade(self, client):
        client.post("/api/v1/bnup/ingest", json=_load_fixture())
        resp = client.get("/api/v1/bnup/retrieve", params={"grade": 2, "limit": 10})
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_knowledge_graph(self, client):
        client.post("/api/v1/bnup/ingest", json=_load_fixture())
        resp = client.get("/api/v1/bnup/knowledge_graph")
        assert resp.status_code == 200
        kg = resp.json()
        assert kg["edition"] == "beishi"
        assert "1" in kg["grades"]
        assert "math-g2-na-02" in kg["knowledge_points"]

    def test_misconceptions_empty_upstream(self, client):
        client.post("/api/v1/bnup/ingest", json=_load_fixture())
        resp = client.get("/api/v1/bnup/misconceptions", params={"knowledge_point": "math-g1-na-01"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["knowledge_point"] == "math-g1-na-01"
        assert body["misconceptions"] == []
        assert body["count"] == 0
        assert "fusion-k12-teacher" in body["dependency"]

    def test_misconceptions_all(self, client):
        client.post("/api/v1/bnup/ingest", json=_load_fixture())
        resp = client.get("/api/v1/bnup/misconceptions")
        assert resp.status_code == 200
        body = resp.json()
        assert "misconceptions_by_kp" in body
        assert body["total"] == 0

    def test_stats(self, client):
        client.post("/api/v1/bnup/ingest", json=_load_fixture())
        resp = client.get("/api/v1/bnup/stats")
        assert resp.status_code == 200
        s = resp.json()
        assert s["total_lessons"] == 15
        assert s["per_grade"]["1"]["lessons"] == 8

    def test_ingest_invalid_corpus(self, client):
        resp = client.post("/api/v1/bnup/ingest", json={"no_grades": True})
        assert resp.status_code == 400

    def test_ingest_idempotent(self, client):
        corpus = _load_fixture()
        r1 = client.post("/api/v1/bnup/ingest", json=corpus)
        assert r1.status_code == 200
        r2 = client.post("/api/v1/bnup/ingest", json=corpus)
        assert r2.status_code == 200
        assert r2.json()["lessons_ingested"] == 15

    def test_limit_validation(self, client):
        client.post("/api/v1/bnup/ingest", json=_load_fixture())
        resp = client.get("/api/v1/bnup/retrieve", params={"limit": 0})
        assert resp.status_code == 400
