"""Tests for issue #86: document active/deprecated status (PATCH + search filtering)
and issue #85: /ask citations + has_low_confidence."""

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


def _search_doc_ids(client, kb_id: str, status_filter: str | None = None) -> set[str]:
    body: dict = {"query": "环境监测", "top_k": 10}
    if status_filter:
        body["status_filter"] = status_filter
    resp = client.post(f"/kb/bases/{kb_id}/search", json=body)
    assert resp.status_code == 200, resp.text
    return {r["doc_id"] for r in resp.json()}


class TestPatchDocument:
    def test_patch_valid_status(self, client):
        kb_id = client.post("/kb/bases", json={"name": "st-kb"}).json()["id"]
        doc_id = _ingest(client, kb_id, "v1.txt", "环境监测数据 版本一 数值 41")

        resp = client.patch(
            f"/kb/bases/{kb_id}/documents/{doc_id}",
            json={"status": "deprecated", "superseded_by": "v2.txt"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["doc_id"] == doc_id
        assert body["status"] == "deprecated"
        assert body["superseded_by"] == "v2.txt"
        assert body["chunks_mirrored"] >= 1

    def test_patch_invalid_status_400(self, client):
        kb_id = client.post("/kb/bases", json={"name": "st-val-kb"}).json()["id"]
        doc_id = _ingest(client, kb_id, "v.txt", "环境监测数据 数值 42")
        resp = client.patch(
            f"/kb/bases/{kb_id}/documents/{doc_id}", json={"status": "archived"}
        )
        assert resp.status_code == 400

    def test_default_active_stamp(self, client):
        kb_id = client.post("/kb/bases", json={"name": "st-active-kb"}).json()["id"]
        _ingest(client, kb_id, "a.txt", "环境监测数据 数值 42")
        rows = client.post(
            f"/kb/bases/{kb_id}/search", json={"query": "环境监测", "top_k": 5}
        ).json()
        assert rows
        assert all(r["status"] == "active" for r in rows)


class TestSearchStatusFilter:
    def test_default_drops_deprecated(self, client):
        kb_id = client.post("/kb/bases", json={"name": "st-filt-kb"}).json()["id"]
        old = _ingest(client, kb_id, "old.txt", "环境监测数据 旧 数值 41")
        new = _ingest(client, kb_id, "new.txt", "环境监测数据 新 数值 42")
        client.patch(f"/kb/bases/{kb_id}/documents/{old}", json={"status": "deprecated"})

        # default = active only
        assert _search_doc_ids(client, kb_id) == {new}
        # deprecated only
        assert _search_doc_ids(client, kb_id, status_filter="deprecated") == {old}
        # all
        assert _search_doc_ids(client, kb_id, status_filter="all") == {old, new}

    def test_invalid_status_filter_400(self, client):
        kb_id = client.post("/kb/bases", json={"name": "st-filt-val-kb"}).json()["id"]
        resp = client.post(
            f"/kb/bases/{kb_id}/search",
            json={"query": "x", "status_filter": "bogus"},
        )
        assert resp.status_code == 400

    def test_deprecated_rows_marked_on_output(self, client):
        kb_id = client.post("/kb/bases", json={"name": "st-mark-kb"}).json()["id"]
        old = _ingest(client, kb_id, "old.txt", "环境监测数据 旧 数值 41")
        client.patch(f"/kb/bases/{kb_id}/documents/{old}", json={"status": "deprecated"})
        rows = client.post(
            f"/kb/bases/{kb_id}/search",
            json={"query": "环境监测", "top_k": 5, "status_filter": "all"},
        ).json()
        by_doc = {r["doc_id"]: r for r in rows}
        assert by_doc[old]["status"] == "deprecated"

    def test_ask_default_hides_deprecated(self, client):
        kb_id = client.post("/kb/bases", json={"name": "st-ask-kb"}).json()["id"]
        old = _ingest(client, kb_id, "old.txt", "环境监测数据 旧 数值 41")
        new = _ingest(client, kb_id, "new.txt", "环境监测数据 新 数值 42")
        client.patch(f"/kb/bases/{kb_id}/documents/{old}", json={"status": "deprecated"})
        client.patch(f"/kb/bases/{kb_id}/documents/{new}", json={"status": "deprecated"})

        # BOTH docs deprecated -> default /ask retrieval is empty (no LLM call
        # happens — the empty-result branch returns before generation).
        resp = client.post(
            f"/kb/bases/{kb_id}/ask",
            json={"question": "环境监测 数值 是多少"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "sources" in body  # /ask contract preserved
        assert body["sources"] == []
        assert body["citations"] == []
        assert body["has_low_confidence"] is False

        # restore one via PATCH (active again) -> default retrieval sees it again
        client.patch(f"/kb/bases/{kb_id}/documents/{old}", json={"status": "active"})
        assert old in _search_doc_ids(client, kb_id)


class TestAskCitations:
    def test_citations_contract(self, client):
        kb_id = client.post("/kb/bases", json={"name": "cite-kb"}).json()["id"]
        _ingest(client, kb_id, "cite.txt", "环境监测数据 数值 42")

        class _FakeResp:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {"choices": [{"message": {"content": "数值是 42。"}}]}

        # /ask calls routes._generate_answer, which uses with_retry for the LLM
        # call — mock it so no live fusion-mlx is needed and the real
        # citation-building code runs.
        with patch(
            "fusion_rag.api.routes.with_retry", new_callable=AsyncMock, return_value=_FakeResp()
        ):
            resp = client.post(
                f"/kb/bases/{kb_id}/ask",
                json={"question": "环境监测数值是多少"},
            )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["answer"] == "数值是 42。"
        assert body["sources"], "expected deduped sources"
        assert body["citations"], "expected per-chunk citations"
        c = body["citations"][0]
        for key in ("doc_id", "doc_name", "chunk_id", "page", "score", "status"):
            assert key in c
        assert body["has_low_confidence"] is False
        # no OCR chunks cited -> no ocr_confidence key on the citation
        assert "ocr_confidence" not in c

    def test_low_confidence_flag_from_ocr(self, client, tmp_path, monkeypatch):
        from fusion_rag.engine.ocr_hook import OcrHook

        monkeypatch.setenv("FUSION_RAG_OCR_URL", "http://127.0.0.1:11491")
        pdf_path = tmp_path / "scanned.pdf"
        import fitz

        doc = fitz.open()
        doc.new_page()
        doc.save(str(pdf_path))
        doc.close()

        ocr_chunks = [
            {
                "task_id": "tsk_9",
                "text": "环境监测数据 数值 42",
                "page": 0,
                "bbox": [0, 0, 100, 100],
                "slice_id": "slc_0_0",
                "ocr_confidence": 0.42,
                "low_confidence": True,
                "kind": "handwriting",
            }
        ]
        with patch.object(OcrHook, "parse_file", new_callable=AsyncMock, return_value=ocr_chunks):
            kb_id = client.post("/kb/bases", json={"name": "cite-ocr-kb"}).json()["id"]
            resp = client.post(
                f"/kb/bases/{kb_id}/documents",
                json={"file_path": str(pdf_path), "contextualize": False},
            )
            assert resp.status_code == 200, resp.text

        class _FakeResp:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {"choices": [{"message": {"content": "数值是 42。"}}]}

        with patch(
            "fusion_rag.api.routes.with_retry", new_callable=AsyncMock, return_value=_FakeResp()
        ):
            resp = client.post(
                f"/kb/bases/{kb_id}/ask",
                json={"question": "环境监测数值是多少"},
            )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["has_low_confidence"] is True
        c = body["citations"][0]
        assert c["ocr_confidence"] == 0.42
        assert c["low_confidence"] is True
        assert c["slice_ref"]["task_id"] == "tsk_9"
        assert c["slice_ref"]["url"] == "http://127.0.0.1:11491/v1/tasks/tsk_9/slices/slc_0_0"
