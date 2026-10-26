"""Tests for issue #90: inline base64 binary payloads (pdf/docx/epub) must go
through the real parse pipeline, not be indexed as base64 text."""

from __future__ import annotations

import base64
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


def _pdf_bytes(text: str) -> bytes:
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def _epub_bytes() -> bytes:
    from ebooklib import epub

    book = epub.EpubBook()
    book.set_identifier("inline-book")
    book.set_title("Inline Book")
    c1 = epub.EpubHtml(title="Chapter One", file_name="chap1.xhtml", lang="en")
    c1.content = "<html><body><h1>Chapter One</h1><p>排放限值 pH 6-9 悬浮物 30mg/L</p></body></html>"
    book.add_item(c1)
    book.toc = (c1,)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", c1]
    import io

    buf = io.BytesIO()
    epub.write_epub(buf, book, {})
    return buf.getvalue()


class TestInlineBinaryIngest:
    def test_base64_pdf_parsed_not_indexed_as_text(self, client):
        payload = base64.b64encode(_pdf_bytes("Env monitor report value 42 pH")).decode()
        kb_id = client.post("/kb/bases", json={"name": "bin-pdf-kb"}).json()["id"]

        resp = client.post(
            f"/kb/bases/{kb_id}/documents/ingest",
            json={"content": payload, "content_type": "pdf", "doc_name": "report.pdf",
                  "contextualize": False},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["chunks"] >= 1
        assert body["doc_name"] == "report.pdf"

        rows = client.post(
            f"/kb/bases/{kb_id}/search", json={"query": "monitor report", "top_k": 5}
        ).json()
        assert rows, "expected search results from the parsed PDF text"
        for r in rows:
            # the base64 blob itself must never be what got chunked
            assert "monitor report" in r["text"] or "monitor report" in r.get("snippet", "")
            assert "JVBERi" not in r["text"]  # base64 of "%PDF" magic

    def test_base64_scanned_pdf_triggers_ocr_path(self, client, monkeypatch):
        from fusion_rag.engine.ocr_hook import OcrHook

        monkeypatch.setenv("FUSION_RAG_OCR_URL", "http://127.0.0.1:11491")
        import fitz

        doc = fitz.open()
        doc.new_page()
        blank = doc.tobytes()
        doc.close()
        payload = base64.b64encode(blank).decode()

        ocr_chunks = [
            {
                "task_id": "tsk_inline",
                "text": "排污许可证 编号 X123 pH 排放限值 6-9",
                "page": 0,
                "bbox": [0, 0, 100, 100],
                "slice_id": "slc_0_0",
                "ocr_confidence": 0.95,
                "low_confidence": False,
                "kind": "text",
            }
        ]
        with patch.object(OcrHook, "parse_file", new_callable=AsyncMock, return_value=ocr_chunks):
            kb_id = client.post("/kb/bases", json={"name": "bin-scan-kb"}).json()["id"]
            resp = client.post(
                f"/kb/bases/{kb_id}/documents/ingest",
                json={"content": payload, "content_type": "pdf", "doc_name": "scan.pdf",
                      "contextualize": False},
            )
        assert resp.status_code == 200, resp.text
        assert resp.json()["chunks"] == 1
        rows = client.post(
            f"/kb/bases/{kb_id}/search", json={"query": "排污许可证", "top_k": 5}
        ).json()
        assert rows
        assert rows[0]["ocr_confidence"] == 0.95
        assert rows[0]["slice_ref"]["task_id"] == "tsk_inline"

    def test_base64_epub_parsed(self, client):
        payload = base64.b64encode(_epub_bytes()).decode()
        kb_id = client.post("/kb/bases", json={"name": "bin-epub-kb"}).json()["id"]

        resp = client.post(
            f"/kb/bases/{kb_id}/documents/ingest",
            json={"content": payload, "content_type": "epub", "doc_name": "book.epub",
                  "contextualize": False},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["chunks"] >= 1
        rows = client.post(
            f"/kb/bases/{kb_id}/search", json={"query": "排放限值", "top_k": 5}
        ).json()
        assert rows, "expected EPUB chapter text to be searchable"

    def test_invalid_base64_rejected_400(self, client):
        kb_id = client.post("/kb/bases", json={"name": "bin-val-kb"}).json()["id"]
        resp = client.post(
            f"/kb/bases/{kb_id}/documents/ingest",
            json={"content": "not!!valid!!base64!!!", "content_type": "pdf"},
        )
        assert resp.status_code == 400
        assert "base64" in resp.json()["detail"]

    def test_plain_text_ingest_unchanged(self, client):
        kb_id = client.post("/kb/bases", json={"name": "bin-text-kb"}).json()["id"]
        resp = client.post(
            f"/kb/bases/{kb_id}/documents/ingest",
            json={"content": "环境监测数据 数值 42", "content_type": "text",
                  "doc_name": "note.txt", "contextualize": False},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["chunks"] >= 1
        rows = client.post(
            f"/kb/bases/{kb_id}/search", json={"query": "环境监测", "top_k": 5}
        ).json()
        assert rows
