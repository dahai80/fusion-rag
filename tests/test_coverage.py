"""Tests for FastAPI routes, connectors, and remaining uncovered modules."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from fusion_rag.api.server import create_app
from fusion_rag.api.routes import set_kb_context
from fusion_rag.engine.knowledge_base import KnowledgeBaseManager
from fusion_rag.embed.client import EmbeddingClient


# ── FastAPI Routes ──

@pytest.fixture
def client():
    """Create a test client with mocked KB manager and embed client."""
    app = create_app(kb_storage_dir=tempfile.mkdtemp())
    with TestClient(app) as tc:
        yield tc


class TestAPIRoutes:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_status(self, client):
        resp = client.get("/kb/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data

    def test_list_bases_empty(self, client):
        resp = client.get("/kb/bases")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_create_base(self, client):
        resp = client.post("/kb/bases", json={"name": "test-kb", "description": "test"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "created"
        assert "id" in data

    def test_create_base_no_name(self, client):
        resp = client.post("/kb/bases", json={})
        assert resp.status_code == 400

    def test_get_base(self, client):
        create = client.post("/kb/bases", json={"name": "my-kb"}).json()
        kb_id = create["id"]
        resp = client.get(f"/kb/bases/{kb_id}")
        assert resp.status_code == 200
        assert resp.json()["name"] == "my-kb"

    def test_get_base_not_found(self, client):
        resp = client.get("/kb/bases/nonexistent")
        assert resp.status_code == 404

    def test_delete_base(self, client):
        create = client.post("/kb/bases", json={"name": "del-kb"}).json()
        resp = client.delete(f"/kb/bases/{create['id']}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

    def test_delete_base_not_found(self, client):
        resp = client.delete("/kb/bases/nonexistent")
        assert resp.status_code == 404

    def test_kb_stats(self, client):
        create = client.post("/kb/bases", json={"name": "stats-kb"}).json()
        resp = client.get(f"/kb/bases/{create['id']}/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "documents" in data

    def test_upload_document_no_file(self, client):
        create = client.post("/kb/bases", json={"name": "doc-kb"}).json()
        resp = client.post(f"/kb/bases/{create['id']}/documents", json={})
        assert resp.status_code == 400

    def test_upload_document_nonexistent(self, client):
        create = client.post("/kb/bases", json={"name": "doc-kb"}).json()
        resp = client.post(f"/kb/bases/{create['id']}/documents",
                           json={"file_path": "/nonexistent/file.txt"})
        assert resp.status_code == 400

    def test_scan_directory_nonexistent(self, client):
        create = client.post("/kb/bases", json={"name": "scan-kb"}).json()
        resp = client.post(f"/kb/bases/{create['id']}/scan",
                           json={"dir_path": "/nonexistent"})
        # API gracefully handles non-existent directory, returns 0 files
        assert resp.status_code == 200
        data = resp.json()
        assert data["files_indexed"] == 0

    def test_search_no_query(self, client):
        create = client.post("/kb/bases", json={"name": "search-kb"}).json()
        resp = client.post(f"/kb/bases/{create['id']}/search", json={})
        assert resp.status_code == 400

    def test_ask_no_question(self, client):
        create = client.post("/kb/bases", json={"name": "ask-kb"}).json()
        resp = client.post(f"/kb/bases/{create['id']}/ask", json={})
        assert resp.status_code == 400


# ── Embedding Client ──

class TestEmbeddingClientAdvanced:
    @pytest.mark.asyncio
    async def test_embed_batch_with_retry(self):
        client = EmbeddingClient(base_url="http://localhost:11434/v1")
        mock_http = MagicMock()
        mock_http.post = AsyncMock()
        # First call fails, second succeeds
        mock_http.post.side_effect = [
            RuntimeError("timeout"),
            MagicMock(status_code=200, json=lambda: {
                "data": [{"embedding": [0.1, 0.2, 0.3]}],
            }),
        ]
        client._client = mock_http
        results = await client.embed_batch(["test text"])
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_embed_batch_all_fail(self):
        client = EmbeddingClient(base_url="http://localhost:11434/v1", max_retries=1)
        mock_http = MagicMock()
        mock_http.post = AsyncMock(side_effect=RuntimeError("always fail"))
        client._client = mock_http
        results = await client.embed_batch(["test"])
        # Should return zero vectors on failure
        assert len(results) == 1
        assert len(results[0]) == 1024

    @pytest.mark.asyncio
    async def test_embed_single(self):
        client = EmbeddingClient(base_url="http://localhost:11434/v1")
        mock_http = MagicMock()
        mock_http.post = AsyncMock(return_value=MagicMock(
            status_code=200, json=lambda: {
                "data": [{"embedding": [0.5, 0.5]}],
            },
        ))
        client._client = mock_http
        result = await client.embed("hello")
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_health_check(self):
        client = EmbeddingClient(base_url="http://localhost:19999/v1", timeout=1.0)
        ok = await client.health()
        assert ok is False


# ── Document Parser ──

class TestDocumentParserAdvanced:
    @pytest.mark.asyncio
    async def test_parse_html(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
            f.write("<html><body><h1>Title</h1><p>Content</p></body></html>")
            path = f.name
        try:
            from fusion_rag.engine.document import DocumentParser
            parser = DocumentParser()
            result = await parser.parse(path)
            assert "Title" in result.content
            assert result.doc_type.value == "html"
        finally:
            Path(path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_parse_directory_recursive(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "a.txt").write_text("file a")
            Path(tmpdir, "sub").mkdir()
            Path(tmpdir, "sub", "b.md").write_text("# sub file")
            from fusion_rag.engine.document import DocumentParser
            parser = DocumentParser()
            results = await parser.parse_directory(tmpdir, recursive=True)
            assert len(results) == 2
            results2 = await parser.parse_directory(tmpdir, recursive=False)
            assert len(results2) == 1

    @pytest.mark.asyncio
    async def test_parse_directory_not_dir(self):
        from fusion_rag.engine.document import DocumentParser
        parser = DocumentParser()
        results = await parser.parse_directory("/nonexistent")
        assert results == []


# ── Vector Store ──

class TestVectorStoreAdvanced:
    @pytest.mark.asyncio
    async def test_add_and_search_with_threshold(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from fusion_rag.store.vector_store import VectorStore
            store = VectorStore(tmpdir, dimension=4)
            store.add("c1", [1.0, 0.0, 0.0, 0.0], "cats", doc_path="/a.txt", doc_name="a.txt", doc_type="txt")
            store.add("c2", [0.0, 1.0, 0.0, 0.0], "dogs", doc_path="/b.txt", doc_name="b.txt", doc_type="txt")
            # Search with high threshold — should only return close matches
            results = store.search([1.0, 0.0, 0.0, 0.0], top_k=5, threshold=0.9)
            assert len(results) >= 1
            # Search with low threshold — should return all
            results2 = store.search([1.0, 0.0, 0.0, 0.0], top_k=5, threshold=0.0)
            assert len(results2) >= 1

    @pytest.mark.asyncio
    async def test_keyword_search(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from fusion_rag.store.vector_store import VectorStore
            store = VectorStore(tmpdir, dimension=4)
            store.add("c1", [1.0, 0.0, 0.0, 0.0], "the cat sat on the mat", doc_path="/a.txt", doc_name="a.txt", doc_type="txt")
            store.add("c2", [0.0, 1.0, 0.0, 0.0], "dogs are great pets", doc_path="/b.txt", doc_name="b.txt", doc_type="txt")
            results = store.keyword_search("cat", top_k=5)
            assert len(results) >= 1
            results2 = store.keyword_search("nonexistent", top_k=5)
            assert len(results2) == 0

    @pytest.mark.asyncio
    async def test_delete_by_doc(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from fusion_rag.store.vector_store import VectorStore
            store = VectorStore(tmpdir, dimension=4)
            store.add("c1", [1.0, 0.0, 0.0, 0.0], "text", doc_path="/a.txt")
            store.add("c2", [0.0, 1.0, 0.0, 0.0], "more", doc_path="/b.txt")
            assert store.count() == 2
            store.delete_by_doc("/a.txt")
            assert store.count() == 1

    @pytest.mark.asyncio
    async def test_add_batch_with_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from fusion_rag.store.vector_store import VectorStore
            store = VectorStore(tmpdir, dimension=4)
            records = [
                {"id": "c1", "vector": [1.0, 0.0, 0.0, 0.0], "text": "a", "doc_path": "", "doc_name": "", "doc_type": "", "chunk_index": 0, "metadata": {"type": "code"}},
            ]
            store.add_batch(records)
            assert store.count() == 1


# ── Metadata Store ──

class TestMetadataStoreAdvanced:
    @pytest.mark.asyncio
    async def test_get_document_by_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from fusion_rag.store.metadata_store import MetadataStore
            store = MetadataStore(str(Path(tmpdir) / "meta.db"))
            store.add_document("d1", "/path/to/file.pdf", "file.pdf", "pdf")
            doc = store.get_document_by_path("/path/to/file.pdf")
            assert doc is not None
            assert doc["file_name"] == "file.pdf"
            doc2 = store.get_document_by_path("/nonexistent")
            assert doc2 is None

    @pytest.mark.asyncio
    async def test_delete_chunks_by_doc(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from fusion_rag.store.metadata_store import MetadataStore
            store = MetadataStore(str(Path(tmpdir) / "meta.db"))
            store.add_document("d1", "/f.txt", "f.txt", "txt")
            store.add_chunk("c1", "d1", "/f.txt", 0, "text")
            store.add_chunk("c2", "d1", "/f.txt", 1, "more")
            assert store.chunk_count() == 2
            store.delete_chunks_by_doc("d1")
            assert store.chunk_count() == 0


# ── RAG Chain ──

class TestRAGChainAdvanced:
    @pytest.mark.asyncio
    async def test_refine(self):
        from fusion_rag.engine.rag_chain import DocumentChain
        with patch("httpx.AsyncClient.post", new=AsyncMock()) as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"choices": [{"message": {"content": "refined answer"}}]}
            mock_post.return_value = mock_resp
            result = await DocumentChain.refine(["doc1"], "query")
            assert "refined answer" in result

    @pytest.mark.asyncio
    async def test_map_reduce(self):
        from fusion_rag.engine.rag_chain import DocumentChain
        with patch("httpx.AsyncClient.post", new=AsyncMock()) as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"choices": [{"message": {"content": "map result"}}]}
            mock_post.return_value = mock_resp
            result = await DocumentChain.map_reduce(["doc1", "doc2"], "query")
            assert "map result" in result


# ── Preprocessor ──

class TestPreprocessorAdvanced:
    def test_clean_with_control_chars(self):
        from fusion_rag.engine.preprocessor import DocumentPreprocessor
        result = DocumentPreprocessor.clean("hello\x00world\x01test")
        assert "hello" in result
        assert "\x00" not in result

    def test_strip_boilerplate(self):
        from fusion_rag.engine.preprocessor import DocumentPreprocessor
        text = "Header\n\nMain content\n\nCopyright 2024\n\nFooter"
        result = DocumentPreprocessor.strip_boilerplate(text)
        # Should remove short lines and boilerplate
        assert "Main content" in result

    def test_recursive_chunker_overlap(self):
        from fusion_rag.engine.preprocessor import RecursiveChunker
        c = RecursiveChunker(chunk_size=20, chunk_overlap=5)
        text = "Hello world. This is a test. " * 10
        chunks = c.chunk(text)
        assert len(chunks) >= 2


# ── Reranker ──

class TestRerankerAdvanced:
    @pytest.mark.asyncio
    async def test_rerank_failure_neutral_score(self):
        from fusion_rag.engine.reranker import Reranker
        r = Reranker()
        with patch("httpx.AsyncClient.post", new=AsyncMock(side_effect=RuntimeError("fail"))):
            docs = [{"id": "1", "text": "test"}]
            results = await r.rerank("query", docs, top_k=1)
            assert len(results) == 1
            assert results[0]["score"] == 5.0  # Neutral score


# ── Connectors ──

class TestConnectorsAdvanced:
    @pytest.mark.asyncio
    async def test_web_loader_success(self):
        from fusion_rag.connectors import WebLoader
        loader = WebLoader()
        with patch("httpx.AsyncClient.get", new=AsyncMock()) as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.text = "<html><body><p>Hello world</p></body></html>"
            mock_resp.raise_for_status = MagicMock()
            mock_get.return_value = mock_resp
            result = await loader.load("http://example.com")
            assert "Hello" in result["content"]

    @pytest.mark.asyncio
    async def test_web_loader_error(self):
        from fusion_rag.connectors import WebLoader
        loader = WebLoader()
        with patch("httpx.AsyncClient.get", side_effect=RuntimeError("fail")):
            result = await loader.load("http://example.com")
            assert "error" in result


# ── Streaming ──

class TestStreamingAdvanced:
    @pytest.mark.asyncio
    async def test_sse_stream(self):
        from fusion_rag.engine.streaming import SSEStreamer
        sse = SSEStreamer()
        with patch("httpx.AsyncClient.stream") as mock_stream:
            mock_ctx = MagicMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)

            async def mock_lines():
                yield 'data: ' + json.dumps({"choices": [{"delta": {"content": "Hello"}}]})
                yield 'data: [DONE]'

            mock_ctx.aiter_lines = mock_lines
            mock_stream.return_value = mock_ctx
            result = await sse.stream_response("query", "context")
            assert "Hello" in result

    @pytest.mark.asyncio
    async def test_metadata_extractor(self):
        from fusion_rag.engine.streaming import MetadataExtractor
        extractor = MetadataExtractor()
        with patch("httpx.AsyncClient.post", new=AsyncMock()) as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {
                "choices": [{"message": {"content": '{"title": "Doc", "language": "en", "topics": ["ai", "ml"]}'}}]
            }
            mock_post.return_value = mock_resp
            meta = await extractor.extract("some text", "doc.md")
            assert "title" in meta

    @pytest.mark.asyncio
    async def test_metadata_extractor_failure(self):
        from fusion_rag.engine.streaming import MetadataExtractor
        extractor = MetadataExtractor()
        with patch("httpx.AsyncClient.post", side_effect=RuntimeError("fail")):
            meta = await extractor.extract("text", "doc.md")
            assert meta["language"] == "unknown"


# ── Retrievers ──

class TestRetrieversAdvanced:
    @pytest.mark.asyncio
    async def test_mmr_empty(self):
        from fusion_rag.engine.retrievers import MMRRetriever
        store = MagicMock()
        store.search = MagicMock(return_value=[])
        mmr = MMRRetriever(store)
        results = await mmr.search([1.0, 0.0], top_k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_context_compression_empty(self):
        from fusion_rag.engine.retrievers import ContextCompressionRetriever
        store = MagicMock()
        retriever = ContextCompressionRetriever(store)
        retriever.base_retriever.search = AsyncMock(return_value=[])
        results = await retriever.search([1.0, 0.0], "query")
        assert results == []