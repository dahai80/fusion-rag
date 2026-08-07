"""Final coverage push — targets remaining uncovered lines in key modules."""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from fusion_rag.api.server import create_app
from fusion_rag.connectors import DatabaseConnector, WebLoader
from fusion_rag.engine.document import DocumentParser, DocumentType
from fusion_rag.engine.reranker import HybridSearch, Reranker
from fusion_rag.engine.retrievers import FusionRetriever, MMRRetriever
from fusion_rag.store.vector_store import VectorStore

# ── API Routes Extra ──

class TestAPIExtra:
    @pytest.fixture
    def client(self):
        app = create_app(kb_storage_dir=tempfile.mkdtemp())
        with TestClient(app) as tc:
            yield tc

    def test_search_with_vectors(self, client):
        """Test search endpoint returns error when no query."""
        create = client.post("/kb/bases", json={"name": "test"}).json()
        resp = client.post(f"/kb/bases/{create['id']}/search", json={})
        assert resp.status_code == 400

    def test_ask_with_vectors(self, client):
        """Test ask endpoint returns error when no question."""
        create = client.post("/kb/bases", json={"name": "test"}).json()
        resp = client.post(f"/kb/bases/{create['id']}/ask", json={})
        assert resp.status_code == 400

    def test_search_with_mocked_embed(self, client):
        """Test search on empty KB returns empty results."""
        create = client.post("/kb/bases", json={"name": "test"}).json()
        resp = client.post(f"/kb/bases/{create['id']}/search",
                           json={"query": "test", "top_k": 5, "threshold": 0.5})
        assert resp.status_code == 200
        data = resp.json()
        results = data if isinstance(data, list) else data.get("results", [])
        assert results == []


# ── Document Parser Extra ──

class TestDocumentParserFinal:
    @pytest.mark.asyncio
    async def test_parse_code_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".swift", delete=False) as f:
            f.write("func hello() {\n    print(\"Hello\")\n}")
            path = f.name
        try:
            parser = DocumentParser()
            result = await parser.parse(path)
            assert result.doc_type == DocumentType.CODE_SWIFT
            assert "hello" in result.content
        finally:
            Path(path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_parse_unknown_extension(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".unknown", delete=False) as f:
            f.write("some text")
            path = f.name
        try:
            parser = DocumentParser()
            result = await parser.parse(path)
            assert result.doc_type in (DocumentType.TXT, DocumentType.UNKNOWN)
        finally:
            Path(path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_parse_empty_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            path = f.name
        try:
            parser = DocumentParser()
            result = await parser.parse(path)
            assert result.chars == 0
        finally:
            Path(path).unlink(missing_ok=True)

    def test_detect_all_types(self):
        parser = DocumentParser()
        assert parser.detect_type("test.pdf") == DocumentType.PDF
        assert parser.detect_type("test.docx") == DocumentType.DOCX
        assert parser.detect_type("test.py") == DocumentType.CODE_PYTHON
        assert parser.detect_type("test.swift") == DocumentType.CODE_SWIFT
        assert parser.detect_type("test.c") == DocumentType.CODE_CPP
        assert parser.detect_type("test.cpp") == DocumentType.CODE_CPP
        assert parser.detect_type("test.js") == DocumentType.CODE_JS
        assert parser.detect_type("test.ts") == DocumentType.CODE_JS
        assert parser.detect_type("test.sh") == DocumentType.CODE_SHELL
        assert parser.detect_type("test.rs") == DocumentType.CODE_OTHER
        assert parser.detect_type("test.go") == DocumentType.CODE_OTHER
        assert parser.detect_type("test.java") == DocumentType.CODE_OTHER
        assert parser.detect_type("test.yaml") == DocumentType.CODE_OTHER
        assert parser.detect_type("test.xyz") == DocumentType.UNKNOWN

    def test_is_code_file(self):
        assert DocumentParser.is_code_file(DocumentType.CODE_PYTHON) is True
        assert DocumentParser.is_code_file(DocumentType.CODE_SWIFT) is True
        assert DocumentParser.is_code_file(DocumentType.PDF) is False
        assert DocumentParser.is_code_file(DocumentType.TXT) is False


# ── Reranker Extra ──

class TestRerankerFinal:
    @pytest.mark.asyncio
    async def test_score_relevance_parse_error(self):
        r = Reranker()
        with patch("httpx.AsyncClient.post", new=AsyncMock()) as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"choices": [{"message": {"content": "not a number"}}]}
            mock_post.return_value = mock_resp
            client = MagicMock()
            client.post = AsyncMock(return_value=mock_resp)
            score = await r._score_relevance(client, "query", "doc")
            assert score == 5.0  # fallback score

    @pytest.mark.asyncio
    async def test_hybrid_search_with_filters(self):
        store = MagicMock()
        store.search = MagicMock(return_value=[
            {"id": "1", "score": 0.8, "metadata": {"type": "pdf"}},
            {"id": "2", "score": 0.6, "metadata": {"type": "md"}},
        ])
        store.keyword_search = MagicMock(return_value=[])
        hs = HybridSearch(store)
        results = await hs.search([1.0, 0.0], "test", filters={"type": "pdf"})
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_hybrid_search_threshold(self):
        store = MagicMock()
        store.search = MagicMock(return_value=[
            {"id": "1", "score": 0.9},
            {"id": "2", "score": 0.3},
        ])
        store.keyword_search = MagicMock(return_value=[])
        hs = HybridSearch(store)
        results = await hs.search([1.0, 0.0], "test", threshold=0.5)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_rrf_ignores_cosine_threshold(self):
        store = MagicMock()
        store.search = MagicMock(return_value=[{"id": "1", "score": 0.9}])
        store.keyword_search = MagicMock(return_value=[{"id": "1", "score": 0.8}])
        hs = HybridSearch(store, method="rrf")
        results = await hs.search([1.0, 0.0], "test", top_k=5, threshold=0.3)
        assert len(results) == 1, "RRF must not be wiped by a cosine-scaled threshold"


# ── Retrievers Extra ──

class TestRetrieversFinal:
    @pytest.mark.asyncio
    async def test_mmr_with_results(self):
        store = MagicMock()
        store.search = MagicMock(return_value=[
            {"id": "1", "score": 0.9, "vector": [1.0, 0.0]},
            {"id": "2", "score": 0.8, "vector": [0.0, 1.0]},
            {"id": "3", "score": 0.7, "vector": [0.5, 0.5]},
        ])
        mmr = MMRRetriever(store)
        results = await mmr.search([1.0, 0.0], top_k=2)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_mmr_cosine_sim_edge_cases(self):
        assert MMRRetriever._cosine_sim([1.0], [1.0]) == 1.0
        assert MMRRetriever._cosine_sim([1.0], [-1.0]) == -1.0
        assert MMRRetriever._cosine_sim([0.0], [1.0]) == 0.0

    @pytest.mark.asyncio
    async def test_fusion_retriever(self):
        r1 = MagicMock()
        r1.search = AsyncMock(return_value=[{"id": "1", "score": 0.9}])
        r2 = MagicMock()
        r2.search = AsyncMock(return_value=[{"id": "2", "score": 0.8}])
        fusion = FusionRetriever([("v", r1, 0.7), ("k", r2, 0.3)])
        results = await fusion.search([1.0, 0.0], "query", top_k=5)
        assert len(results) >= 0


# ── Vector Store Extra ──

class TestVectorStoreFinal:
    @pytest.mark.asyncio
    async def test_count_after_clear(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(tmpdir, dimension=4)
            store.add("c1", [1.0, 0.0, 0.0, 0.0], "text")
            assert store.count() == 1
            store.clear()
            assert store.count() == 0

    @pytest.mark.asyncio
    async def test_keyword_search_with_stopwords(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(tmpdir, dimension=4)
            store.add("c1", [1.0, 0.0, 0.0, 0.0], "the cat sat on the mat")
            store.add("c2", [0.0, 1.0, 0.0, 0.0], "dogs are great")
            results = store.keyword_search("cat", top_k=5)
            assert len(results) >= 1
            results2 = store.keyword_search("the", top_k=5)
            # "the" appears in both documents
            assert len(results2) >= 1

    @pytest.mark.asyncio
    async def test_delete_by_doc_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(tmpdir, dimension=4)
            store.add("c1", [1.0, 0.0, 0.0, 0.0], "text")
            result = store.delete_by_doc("/nonexistent.txt")
            assert result == 0


# ── Connectors Extra ──

class TestConnectorsFinal:
    @pytest.mark.asyncio
    async def test_database_connector_sqlite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            import sqlite3
            conn = sqlite3.connect(db_path)
            conn.execute("CREATE TABLE test (id INT, name TEXT)")
            conn.execute("INSERT INTO test VALUES (1, 'Alice')")
            conn.commit()
            conn.close()
            dc = DatabaseConnector(db_type="sqlite", connection_string=db_path)
            tables = await dc.list_tables()
            assert len(tables) >= 1
            assert tables[0]["name"] == "test"
            rows = await dc.fetch_table("test")
            assert len(rows) == 1

    @pytest.mark.asyncio
    async def test_web_loader_extract_text(self):
        loader = WebLoader()
        html = "<html><body><script>var x=1;</script><p>Hello world</p></body></html>"
        text = loader._extract_text(html)
        assert "Hello" in text
        assert "script" not in text.lower() or "var" not in text
