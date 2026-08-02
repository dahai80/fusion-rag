"""Final coverage push — targets remaining uncovered lines for 90%+."""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from fusion_rag.connectors import DatabaseConnector
from fusion_rag.engine.document import DocumentParser
from fusion_rag.store.vector_store import VectorStore

# ── Vector Store: lazy imports ──

class TestVectorStoreLazy:
    @pytest.mark.asyncio
    async def test_vector_store_close(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(tmpdir, dimension=4)
            store.close()
            assert store._backend._db is None

    @pytest.mark.asyncio
    async def test_vector_store_delete_by_doc_escape(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(tmpdir, dimension=4)
            # Test with a path containing single quotes
            result = store.delete_by_doc("/path/with'quote.txt")
            # Should not crash, just return 0 since no matching docs
            assert result == 0


# ── Document Parser: PDF/DOCX fallback ──

class TestDocumentParserFallback:
    @pytest.mark.asyncio
    async def test_parse_pdf_missing_library(self):
        """Test that missing PyMuPDF is handled gracefully."""
        with patch.dict("sys.modules", {"fitz": None}):
            parser = DocumentParser()
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                f.write(b"%PDF-1.4 fake pdf content")
                path = f.name
            try:
                result = await parser.parse(path)
                # Should fail gracefully since fitz is mocked out
                assert result.error or not result.content
            finally:
                Path(path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_parse_docx_missing_library(self):
        """Test that missing python-docx is handled gracefully."""
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "docx":
                raise ImportError("No module named 'docx'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            parser = DocumentParser()
            with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
                f.write(b"fake docx content")
                path = f.name
            try:
                result = await parser.parse(path)
                # Should fail gracefully
                assert result.error
            finally:
                Path(path).unlink(missing_ok=True)


# ── Connectors: edge cases ──

class TestConnectorsEdge:
    @pytest.mark.asyncio
    async def test_database_sqlite_fetch_error(self):
        dc = DatabaseConnector(db_type="sqlite", connection_string="/nonexistent/db.sqlite")
        rows = await dc.fetch_table("test")
        assert rows == []

    @pytest.mark.asyncio
    async def test_database_postgres_list_error(self):
        dc = DatabaseConnector(db_type="postgresql", connection_string="postgresql://localhost/test")
        tables = await dc.list_tables()
        assert tables == []

    @pytest.mark.asyncio
    async def test_database_unknown_type(self):
        dc = DatabaseConnector(db_type="unknown", connection_string="")
        tables = await dc.list_tables()
        assert tables == []
        rows = await dc.fetch_table("test")
        assert rows == []


# ── Knowledge Base: edge cases ──

class TestKnowledgeBaseEdge:
    def test_kb_config_from_dict_partial(self):
        from fusion_rag.engine.knowledge_base import KnowledgeBaseConfig
        cfg = KnowledgeBaseConfig.from_dict({"name": "test"})
        assert cfg.name == "test"
        assert cfg.chunk_size == 512  # default

    def test_kb_to_dict_roundtrip(self):
        from fusion_rag.engine.knowledge_base import KnowledgeBase, KnowledgeBaseConfig
        kb = KnowledgeBase(config=KnowledgeBaseConfig(name="test"))
        d = kb.to_dict()
        assert d["name"] == "test"
        # Roundtrip
        kb2 = KnowledgeBase.from_dict(d)
        assert kb2.id == kb.id
        assert kb2.config.name == "test"


# ── Server: run_server ──

class TestServerEdge:
    def test_run_server_imports(self):
        """Test that server module imports correctly."""
        from fusion_rag.api import server
        assert hasattr(server, "create_app")
        assert hasattr(server, "run_server")
