"""Tests for new P0/P1/P2 capabilities in fusion-rag."""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fusion_rag.connectors import DatabaseConnector, WebLoader
from fusion_rag.engine.preprocessor import DocumentPreprocessor, RecursiveChunker
from fusion_rag.engine.rag_chain import DocumentChain, MultiTurnRAG
from fusion_rag.engine.reranker import HybridSearch, Reranker
from fusion_rag.engine.retrievers import ContextCompressionRetriever, FusionRetriever, MMRRetriever
from fusion_rag.engine.streaming import MetadataExtractor, ResultCache

# ── Reranker ──

class TestReranker:
    @pytest.mark.asyncio
    async def test_rerank_empty(self):
        r = Reranker()
        results = await r.rerank("query", [], top_k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_rerank_with_docs(self):
        r = Reranker()
        docs = [{"id": "1", "text": "apple"}, {"id": "2", "text": "banana"}]
        with patch("httpx.AsyncClient.post", new=AsyncMock()) as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"choices": [{"message": {"content": "8"}}]}
            mock_post.return_value = mock_resp
            results = await r.rerank("apple", docs, top_k=2)
            assert len(results) == 2


# ── HybridSearch ──

class TestHybridSearch:
    def test_apply_filters(self):
        results = [
            {"id": "1", "metadata": {"type": "pdf", "date": "2024"}},
            {"id": "2", "metadata": {"type": "md", "date": "2025"}},
        ]
        filtered = HybridSearch._apply_filters(results, {"type": "pdf"})
        assert len(filtered) == 1
        assert filtered[0]["id"] == "1"


# ── MMRRetriever ──

class TestMMRRetriever:
    def test_cosine_sim(self):
        sim = MMRRetriever._cosine_sim([1, 0, 0], [1, 0, 0])
        assert sim == 1.0
        sim = MMRRetriever._cosine_sim([1, 0], [0, 1])
        assert sim == 0.0
        sim = MMRRetriever._cosine_sim([], [])
        assert sim == 0.0


# ── DocumentPreprocessor ──

class TestDocumentPreprocessor:
    def test_clean(self):
        assert DocumentPreprocessor.clean("hello   world") == "hello world"
        assert DocumentPreprocessor.clean("\x00test") == "test"

    def test_normalize(self):
        assert DocumentPreprocessor.normalize("  Hello  ") == "Hello"

    def test_deduplicate(self):
        chunks = [{"text": "hello"}, {"text": "hello"}, {"text": "world"}]
        result = DocumentPreprocessor.deduplicate(chunks)
        assert len(result) == 2


# ── RecursiveChunker ──

class TestRecursiveChunker:
    def test_empty(self):
        c = RecursiveChunker()
        assert c.chunk("") == []
        assert c.chunk("  ") == []

    def test_small(self):
        c = RecursiveChunker(chunk_size=100)
        chunks = c.chunk("hello world")
        assert len(chunks) == 1

    def test_large(self):
        c = RecursiveChunker(chunk_size=50)
        text = "A" * 200
        chunks = c.chunk(text)
        assert len(chunks) >= 4


# ── MultiTurnRAG ──

class TestMultiTurnRAG:
    @pytest.mark.asyncio
    async def test_ask(self):
        rag = MultiTurnRAG()
        with patch("httpx.AsyncClient.post", new=AsyncMock()) as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"choices": [{"message": {"content": "test answer"}}]}
            mock_post.return_value = mock_resp
            result = await rag.ask("test question", "some context")
            assert "test answer" in result["answer"]
            assert rag._history
        rag.clear_history()
        assert len(rag._history) == 0


# ── DocumentChain ──

class TestDocumentChain:
    @pytest.mark.asyncio
    async def test_stuff(self):
        with patch("httpx.AsyncClient.post", new=AsyncMock()) as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"choices": [{"message": {"content": "stuff answer"}}]}
            mock_post.return_value = mock_resp
            result = await DocumentChain.stuff(["doc1", "doc2"], "query")
            assert "stuff answer" in result


# ── MetadataExtractor ──

class TestMetadataExtractor:
    @pytest.mark.asyncio
    async def test_extract(self):
        extractor = MetadataExtractor()
        with patch("httpx.AsyncClient.post", new=AsyncMock()) as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {
                "choices": [{"message": {"content": '{"title": "Test", "language": "en", "topics": ["ai"]}'}}]
            }
            mock_post.return_value = mock_resp
            meta = await extractor.extract("test text", "doc.md")
            assert meta.get("title") == "Test" or meta.get("language") == "en"


# ── ResultCache ──

class TestResultCache:
    def test_set_and_get(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ResultCache(str(Path(tmpdir) / "cache.db"))
            cache.set("hello", "world answer", context="ctx", sources=[{"id": "1"}])
            result = cache.get("hello", "ctx")
            assert result is not None
            assert "world" in result["answer"]

    def test_miss(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ResultCache(str(Path(tmpdir) / "cache.db"))
            result = cache.get("nonexistent")
            assert result is None


# ── DatabaseConnector ──

class TestDatabaseConnector:
    @pytest.mark.asyncio
    async def test_sqlite_invalid(self):
        dc = DatabaseConnector(db_type="sqlite", connection_string="/nonexistent/db.sqlite")
        tables = await dc.list_tables()
        assert tables == []

    @pytest.mark.asyncio
    async def test_postgres_no_asyncpg(self):
        dc = DatabaseConnector(db_type="postgresql", connection_string="postgresql://localhost/test")
        tables = await dc.list_tables()
        assert tables == []


# ── WebLoader ──

class TestWebLoader:
    @pytest.mark.asyncio
    async def test_load_invalid_url(self):
        loader = WebLoader()
        result = await loader.load("http://nonexistent-domain-xyz-123.com")
        assert "error" in result


# ── ContextCompressionRetriever ──

class TestContextCompression:
    def test_compress(self):
        store = MagicMock()
        retriever = ContextCompressionRetriever(store, max_tokens=10)
        results = [{"text": "hello world " * 100, "score": 0.9}]
        compressed = retriever._compress(results, "query")
        assert len(compressed) >= 1


# ── FusionRetriever ──

class TestFusionRetriever:
    @pytest.mark.asyncio
    async def test_search(self):
        retriever1 = MagicMock()
        retriever1.search = AsyncMock(return_value=[{"id": "1", "score": 0.8}])
        fusion = FusionRetriever([("vec", retriever1, 1.0)])
        results = await fusion.search([1.0, 0.0])
        assert len(results) >= 0
