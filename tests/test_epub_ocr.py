"""Tests for issues #84/#85: EPUB parsing + scanned-PDF OCR ingest with evidence metadata."""

from __future__ import annotations

import asyncio
import tempfile
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from fusion_rag.api.server import create_app
from fusion_rag.embed.client import EmbeddingClient
from fusion_rag.engine.document import DocumentParser
from fusion_rag.engine.ocr_hook import OcrHook


def _make_epub(path) -> None:
    from ebooklib import epub

    book = epub.EpubBook()
    book.set_identifier("test-book-1")
    book.set_title("Test Book")
    c1 = epub.EpubHtml(title="Chapter One", file_name="chap1.xhtml", lang="en")
    c1.content = "<html><body><h1>Chapter One</h1><p>排放限值 pH 6-9 悬浮物 30mg/L</p></body></html>"
    c2 = epub.EpubHtml(title="Chapter Two", file_name="chap2.xhtml", lang="en")
    c2.content = "<html><body><h1>Chapter Two</h1><p>环境监测数据 数值 42</p></body></html>"
    book.add_item(c1)
    book.add_item(c2)
    book.toc = (c1, c2)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", c1, c2]
    epub.write_epub(str(path), book, {})


def _make_scanned_pdf(path) -> None:
    # Blank pages: pages exist, zero extractable text = scanned PDF.
    import fitz

    doc = fitz.open()
    for _ in range(2):
        doc.new_page()
    doc.save(str(path))
    doc.close()


def _ocr_chunks() -> list[dict]:
    return [
        {
            "task_id": "tsk_001",
            "text": "排污许可证 编号 X123 pH 排放限值 6-9",
            "page": 2,
            "bbox": [10, 20, 300, 80],
            "slice_id": "slc_2_0",
            "ocr_confidence": 0.97,
            "low_confidence": False,
            "kind": "text",
        },
        {
            "task_id": "tsk_001",
            "text": "手写批注 现场照片 不清晰",
            "page": 3,
            "bbox": [10, 90, 300, 200],
            "slice_id": "slc_3_1",
            "ocr_confidence": 0.42,
            "low_confidence": True,
            "kind": "handwriting",
        },
    ]


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


class TestEpub:
    def test_parse_epub(self, tmp_path):
        epub_path = tmp_path / "book.epub"
        _make_epub(epub_path)
        result = asyncio.run(DocumentParser().parse(epub_path))
        assert not result.error, result.error
        assert result.doc_type.value == "epub"
        assert "Chapter One" in result.content
        assert "排放限值 pH 6-9" in result.content
        assert "Chapter Two" in result.content

    def test_epub_ingest_roundtrip(self, client, tmp_path):
        epub_path = tmp_path / "book.epub"
        _make_epub(epub_path)
        kb_id = client.post("/kb/bases", json={"name": "epub-kb"}).json()["id"]
        resp = client.post(
            f"/kb/bases/{kb_id}/documents",
            json={"file_path": str(epub_path), "contextualize": False},
        )
        assert resp.status_code == 200, resp.text
        assert "error" not in resp.json(), resp.json()
        assert resp.json()["chunks"] >= 1

        resp = client.post(f"/kb/bases/{kb_id}/search", json={"query": "排放限值", "top_k": 5})
        assert resp.status_code == 200, resp.text
        assert resp.json(), "expected at least one search result from the EPUB"


class TestScannedPdf:
    def test_scanned_pdf_no_ocr_configured_fails_visibly(self, client, tmp_path, monkeypatch):
        monkeypatch.delenv("FUSION_RAG_OCR_URL", raising=False)
        pdf_path = tmp_path / "scanned.pdf"
        _make_scanned_pdf(pdf_path)

        result = asyncio.run(DocumentParser().parse(pdf_path))
        assert result.error is not None
        assert "no OCR service configured" in result.error
        assert result.ocr_chunks == []

        kb_id = client.post("/kb/bases", json={"name": "scan-kb"}).json()["id"]
        resp = client.post(
            f"/kb/bases/{kb_id}/documents",
            json={"file_path": str(pdf_path), "contextualize": False},
        )
        assert resp.status_code == 400
        assert "no OCR service configured" in resp.json()["detail"]

    def test_ocr_ingest_roundtrip(self, client, tmp_path, monkeypatch):
        monkeypatch.setenv("FUSION_RAG_OCR_URL", "http://127.0.0.1:11491")
        pdf_path = tmp_path / "scanned2.pdf"
        _make_scanned_pdf(pdf_path)

        with patch.object(OcrHook, "parse_file", new_callable=AsyncMock, return_value=_ocr_chunks()):
            kb_id = client.post("/kb/bases", json={"name": "ocr-kb"}).json()["id"]
            resp = client.post(
                f"/kb/bases/{kb_id}/documents",
                json={"file_path": str(pdf_path), "contextualize": True},
            )
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert "error" not in body, body
            assert body["chunks"] == 2

            resp = client.post(f"/kb/bases/{kb_id}/search", json={"query": "排污许可证", "top_k": 10})
            assert resp.status_code == 200, resp.text
            rows = resp.json()
            assert len(rows) == 2
            # pages are 1-based in fusion-rag (drawing-parser is 0-based)
            pages = {r["page"] for r in rows}
            assert pages == {3, 4}
            by_conf = {r["ocr_confidence"]: r for r in rows}
            high = by_conf[0.97]
            assert high["low_confidence"] is False
            assert high["slice_ref"]["task_id"] == "tsk_001"
            assert high["slice_ref"]["slice_id"] == "slc_2_0"
            assert high["slice_ref"]["url"] == "http://127.0.0.1:11491/v1/tasks/tsk_001/slices/slc_2_0"
            low = by_conf[0.42]
            assert low["low_confidence"] is True
            assert low["metadata"]["ocr_kind"] == "handwriting"

    def test_ocr_service_failure_visible(self, client, tmp_path, monkeypatch):
        from fusion_rag.engine.ocr_hook import OcrUnavailable

        monkeypatch.setenv("FUSION_RAG_OCR_URL", "http://127.0.0.1:11491")
        pdf_path = tmp_path / "scanned3.pdf"
        _make_scanned_pdf(pdf_path)

        with patch.object(
            OcrHook,
            "parse_file",
            new_callable=AsyncMock,
            side_effect=OcrUnavailable("OCR task t_x failed: bad render"),
        ):
            kb_id = client.post("/kb/bases", json={"name": "ocr-fail-kb"}).json()["id"]
            resp = client.post(
                f"/kb/bases/{kb_id}/documents",
                json={"file_path": str(pdf_path), "contextualize": False},
            )
            assert resp.status_code == 400
            assert "OCR error" in resp.json()["detail"]
