"""Tests for the evidence metadata contract (issue #77): per-chunk page
tracking + chunk_id/doc_id/page round-trip through search."""

from __future__ import annotations

import tempfile
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from fusion_rag.api.server import create_app
from fusion_rag.embed.client import EmbeddingClient
from fusion_rag.engine.chunker import Chunker
from fusion_rag.engine.document import DocumentParser, DocumentType, ParseResult


def _pdf_result(content: str, spans: list[tuple[int, int, int]]) -> ParseResult:
    return ParseResult(
        file_path="test://doc.pdf",
        file_name="doc.pdf",
        doc_type=DocumentType.PDF,
        content=content,
        page_spans=spans,
    )


# ── chunker page mapping ──


class TestChunkerPages:
    def test_semantic_pages(self):
        p1 = "alpha " * 60
        p2 = "beta " * 60
        content = f"# H1 {p1}\n\n# H2 {p2}"
        spans = [(0, len(p1) + 5, 1), (len(p1) + 7, len(content), 2)]
        chunker = Chunker(strategy="semantic", chunk_size=200)
        chunks = chunker._chunk_semantic(_pdf_result(content, spans))
        assert len(chunks) == 2
        assert chunks[0].page == 1
        assert chunks[1].page == 2

    def test_fixed_pages(self):
        p1 = "x" * 100
        p2 = "y" * 100
        content = p1 + "\n\n" + p2
        spans = [(0, 100, 1), (102, 202, 2)]
        chunker = Chunker(strategy="fixed", chunk_size=50, chunk_overlap=10)
        chunks = chunker._chunk_fixed(_pdf_result(content, spans))
        assert len(chunks) > 2
        assert chunks[0].page == 1
        assert chunks[-1].page == 2
        assert all(c.page in (1, 2) for c in chunks)

    def test_no_spans_page_zero(self):
        result = ParseResult(
            file_path="a.txt",
            file_name="a.txt",
            doc_type=DocumentType.TXT,
            content="# H1 some text here\n\n# H2 more text",
        )
        chunker = Chunker(strategy="semantic", chunk_size=200)
        chunks = chunker._chunk_semantic(result)
        assert all(c.page == 0 for c in chunks)

    def test_pdf_parse_spans(self, tmp_path):
        import fitz

        doc = fitz.open()
        for i, text in enumerate(["page one alpha", "page two beta", "page three gamma"]):
            page = doc.new_page()
            page.insert_text((72, 72), text)
        pdf_path = tmp_path / "three.pdf"
        doc.save(str(pdf_path))
        doc.close()

        import asyncio

        parser = DocumentParser()
        result = asyncio.run(parser.parse(pdf_path))
        assert result.pages == 3
        assert len(result.page_spans) == 3
        assert result.page_spans[0][2] == 1
        assert result.page_spans[2][2] == 3
        assert "page three gamma" in result.content
        # span offsets are exact — content[span] is the page text
        start, end, _ = result.page_spans[1]
        assert result.content[start:end].strip() == "page two beta"


# ── API round-trip ──


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


class TestEvidenceApi:
    def test_inline_ocr_roundtrip(self, client):
        kb_id = client.post("/kb/bases", json={"name": "ev-ocr"}).json()["id"]
        resp = client.post(
            f"/kb/bases/{kb_id}/documents/ingest",
            json={
                "content": "排污许可证 编号 X123 pH 排放限值 6-9 悬浮物 30mg/L",
                "doc_name": "permit_page3.txt",
                "metadata": {"source": "ocr", "page": 3},
                "contextualize": False,
            },
        )
        assert resp.status_code == 200, resp.text
        doc_id = resp.json()["doc_id"]

        resp = client.post(f"/kb/bases/{kb_id}/search", json={"query": "排放限值 pH", "top_k": 5})
        assert resp.status_code == 200, resp.text
        rows = resp.json()
        assert rows, "expected at least one search result"
        row = rows[0]
        assert row["doc_id"] == doc_id
        assert row["chunk_id"] == row["id"]
        assert row["chunk_id"].startswith(doc_id)
        assert row["page"] == 3
        assert row["metadata"]["source"] == "ocr"
        assert row["metadata"]["doc_id"] == doc_id

    def test_pdf_ingest_pages(self, client, tmp_path):
        import fitz

        doc = fitz.open()
        for i, tag in enumerate(["alpha", "beta", "gamma"]):
            page = doc.new_page()
            page.insert_text((72, 72), f"# Section {tag}")
            for line in range(6):
                page.insert_text((72, 100 + 20 * line), f"{tag} regulation clause {line} discharge limit value text.")
        pdf_path = tmp_path / "regulations.pdf"
        doc.save(str(pdf_path))
        doc.close()

        kb_id = client.post("/kb/bases", json={"name": "ev-pdf"}).json()["id"]
        resp = client.post(
            f"/kb/bases/{kb_id}/documents",
            json={"file_path": str(pdf_path), "contextualize": False},
        )
        assert resp.status_code == 200, resp.text
        doc_id = resp.json()["doc_id"]
        assert resp.json()["chunks"] >= 3

        resp = client.post(f"/kb/bases/{kb_id}/search", json={"query": "regulation clause", "top_k": 10})
        assert resp.status_code == 200, resp.text
        rows = resp.json()
        assert len(rows) >= 3
        pages = {r["page"] for r in rows}
        assert pages == {1, 2, 3}
        assert all(r["doc_id"] == doc_id for r in rows)
        assert all(r["chunk_id"] == r["id"] for r in rows)

    def test_metadata_filter_still_works(self, client):
        # Regression: chunk metadata now carries doc_id/chunk_index extras —
        # user metadata_filter equality must keep matching.
        kb_id = client.post("/kb/bases", json={"name": "ev-filter"}).json()["id"]
        for i, proj in enumerate(["projA", "projB"]):
            client.post(
                f"/kb/bases/{kb_id}/documents/ingest",
                json={
                    "content": f"环境监测数据 {proj} 数值 42",
                    "doc_name": f"{proj}.txt",
                    "metadata": {"project": proj},
                    "contextualize": False,
                },
            )
        resp = client.post(
            f"/kb/bases/{kb_id}/search",
            json={"query": "环境监测", "filter": {"project": "projA"}, "top_k": 10},
        )
        assert resp.status_code == 200, resp.text
        rows = resp.json()
        assert rows
        assert all(r["metadata"]["project"] == "projA" for r in rows)


# ── _normalize_evidence unit ──


class TestNormalizeEvidence:
    def test_doc_id_fallback_from_chunk_id(self):
        from fusion_rag.api.routes import _normalize_evidence

        rows = [{"id": "abc123def456_2", "metadata": "{}"}]
        out = _normalize_evidence(rows)
        assert out[0]["doc_id"] == "abc123def456"
        assert out[0]["chunk_id"] == "abc123def456_2"
        assert out[0]["page"] == 0

    def test_metadata_doc_id_wins(self):
        from fusion_rag.api.routes import _normalize_evidence

        rows = [{"id": "x_0", "metadata": {"doc_id": "real-doc", "page": 7}}]
        out = _normalize_evidence(rows)
        assert out[0]["doc_id"] == "real-doc"
        assert out[0]["page"] == 7

    def test_no_underscore_chunk_id(self):
        from fusion_rag.api.routes import _normalize_evidence

        rows = [{"id": "blake2bhash", "metadata": {}}]
        out = _normalize_evidence(rows)
        assert out[0]["doc_id"] == "blake2bhash"
