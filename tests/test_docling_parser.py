"""Issue #131 (R9): Docling PDF backend selection, per-KB pin, fallback."""

from __future__ import annotations

import pytest

from fusion_rag.engine.docling_parser import DoclingParser
from fusion_rag.engine.document import DocumentParser
from fusion_rag.engine.runtime_config import reset_runtime_config


@pytest.fixture(autouse=True)
def _clean_cfg(monkeypatch):
    monkeypatch.delenv("FUSION_RAG_PDF_BACKEND", raising=False)
    reset_runtime_config()
    yield
    reset_runtime_config()


@pytest.mark.asyncio
async def test_docling_backend_used_and_stamped(tmp_path, monkeypatch):
    pdf = tmp_path / "t.pdf"
    _make_real_pdf(pdf)
    parser = DocumentParser()

    async def fake_parse_pdf(self, path):
        return "docling text", 1, [(0, 12, 1)]

    monkeypatch.setattr(DoclingParser, "parse_pdf", fake_parse_pdf)
    result = await parser.parse(pdf, pdf_backend="docling")
    assert result.error == ""
    assert result.content == "docling text"
    assert result.metadata["pdf_backend"] == "docling"


@pytest.mark.asyncio
async def test_docling_failure_falls_back_to_legacy(tmp_path, monkeypatch):
    pdf = tmp_path / "t.pdf"
    _make_real_pdf(pdf)
    parser = DocumentParser()

    async def boom(self, path):
        raise RuntimeError("model download failed")

    monkeypatch.setattr(DoclingParser, "parse_pdf", boom)
    result = await parser.parse(pdf, pdf_backend="docling")
    assert result.error == ""
    assert "hello table" in result.content
    assert result.metadata["pdf_backend"] == "legacy"


@pytest.mark.asyncio
async def test_env_default_resolves_when_param_empty(tmp_path, monkeypatch):
    pdf = tmp_path / "t.pdf"
    _make_real_pdf(pdf)
    monkeypatch.setenv("FUSION_RAG_PDF_BACKEND", "docling")
    reset_runtime_config()
    parser = DocumentParser()
    called = {}

    async def fake(self, path):
        called["hit"] = True
        return "docling text", 1, [(0, 12, 1)]

    monkeypatch.setattr(DoclingParser, "parse_pdf", fake)
    result = await parser.parse(pdf, pdf_backend="")
    assert called.get("hit") is True
    assert result.metadata["pdf_backend"] == "docling"


@pytest.mark.asyncio
async def test_legacy_pin_overrides_env(tmp_path, monkeypatch):
    pdf = tmp_path / "t.pdf"
    _make_real_pdf(pdf)
    monkeypatch.setenv("FUSION_RAG_PDF_BACKEND", "docling")
    reset_runtime_config()
    parser = DocumentParser()
    called = {}

    async def fake(self, path):
        called["hit"] = True
        return "docling text", 1, [(0, 12, 1)]

    monkeypatch.setattr(DoclingParser, "parse_pdf", fake)
    result = await parser.parse(pdf, pdf_backend="legacy")
    assert called == {}
    assert result.metadata["pdf_backend"] == "legacy"


def test_kb_config_pdf_backend_persisted_and_validated(tmp_path):
    import tempfile

    from fusion_rag.engine.knowledge_base import KnowledgeBaseManager

    with tempfile.TemporaryDirectory() as d:
        mgr = KnowledgeBaseManager(storage_dir=d)
        kb = mgr.create(name="t", pdf_backend="docling")
        assert kb.config.pdf_backend == "docling"
        kb2 = mgr.create(name="t2")
        assert kb2.config.pdf_backend == ""


def test_runtime_config_pdf_backend_from_env(monkeypatch):
    from fusion_rag.engine.runtime_config import RuntimeConfig

    monkeypatch.setenv("FUSION_RAG_PDF_BACKEND", "DOCLING")
    assert RuntimeConfig.from_env().pdf_backend == "docling"
    monkeypatch.delenv("FUSION_RAG_PDF_BACKEND")
    assert RuntimeConfig.from_env().pdf_backend == "legacy"


def _make_real_pdf(path):
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "hello table")
    page.insert_text((72, 90), "row one")
    doc.save(str(path))
    doc.close()
