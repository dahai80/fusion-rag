"""Tests for Fusion-KB core modules."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fusion_kb.engine.knowledge_base import KnowledgeBase, KnowledgeBaseConfig, KnowledgeBaseManager
from fusion_kb.engine.document import DocumentParser, DocumentType, ParseResult
from fusion_kb.engine.chunker import Chunker, Chunk
from fusion_kb.embed.client import EmbeddingClient
from fusion_kb.store.vector_store import VectorStore
from fusion_kb.store.metadata_store import MetadataStore


# ── KnowledgeBaseConfig ──

class TestKnowledgeBaseConfig:
    def test_defaults(self):
        cfg = KnowledgeBaseConfig(name="test")
        assert cfg.name == "test"
        assert cfg.chunk_strategy == "semantic"
        assert cfg.chunk_size == 512

    def test_to_dict(self):
        cfg = KnowledgeBaseConfig(name="test", description="desc")
        d = cfg.to_dict()
        assert d["name"] == "test"
        assert d["description"] == "desc"

    def test_from_dict(self):
        cfg = KnowledgeBaseConfig.from_dict({"name": "test", "chunk_size": 256})
        assert cfg.name == "test"
        assert cfg.chunk_size == 256


# ── KnowledgeBase ──

class TestKnowledgeBase:
    def test_create(self):
        kb = KnowledgeBase(config=KnowledgeBaseConfig(name="test"))
        assert kb.id
        assert kb.config.name == "test"
        assert kb.created_at > 0

    def test_to_dict_roundtrip(self):
        kb = KnowledgeBase(config=KnowledgeBaseConfig(name="test"))
        d = kb.to_dict()
        assert d["name"] == "test"
        kb2 = KnowledgeBase.from_dict(d)
        assert kb2.id == kb.id
        assert kb2.config.name == "test"


# ── KnowledgeBaseManager ──

class TestKnowledgeBaseManager:
    def test_create_and_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = KnowledgeBaseManager(storage_dir=tmpdir)
            kb = mgr.create("test-kb", "A test knowledge base")
            assert kb.id
            bases = mgr.list()
            assert len(bases) == 1
            assert bases[0]["name"] == "test-kb"

    def test_get(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = KnowledgeBaseManager(storage_dir=tmpdir)
            kb = mgr.create("test-kb")
            loaded = mgr.get(kb.id)
            assert loaded.id == kb.id

    def test_get_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = KnowledgeBaseManager(storage_dir=tmpdir)
            with pytest.raises(KeyError):
                mgr.get("nonexistent")

    def test_delete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = KnowledgeBaseManager(storage_dir=tmpdir)
            kb = mgr.create("test-kb")
            assert mgr.delete(kb.id) is True
            assert len(mgr.list()) == 0

    def test_delete_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = KnowledgeBaseManager(storage_dir=tmpdir)
            assert mgr.delete("nonexistent") is False

    def test_update(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = KnowledgeBaseManager(storage_dir=tmpdir)
            kb = mgr.create("test-kb")
            mgr.update(kb.id, chunk_size=1024)
            updated = mgr.get(kb.id)
            assert updated.config.chunk_size == 1024

    def test_persistence(self):
        """Test that KB metadata survives manager re-creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr1 = KnowledgeBaseManager(storage_dir=tmpdir)
            mgr1.create("persistent-kb")
            mgr2 = KnowledgeBaseManager(storage_dir=tmpdir)
            assert len(mgr2.list()) == 1


# ── DocumentParser ──

class TestDocumentParser:
    def test_detect_type(self):
        assert DocumentParser.detect_type("test.pdf") == DocumentType.PDF
        assert DocumentParser.detect_type("test.py") == DocumentType.CODE_PYTHON
        assert DocumentParser.detect_type("test.md") == DocumentType.MARKDOWN
        assert DocumentParser.detect_type("test.unknown") == DocumentType.UNKNOWN

    def test_is_code_file(self):
        assert DocumentParser.is_code_file(DocumentType.CODE_PYTHON) is True
        assert DocumentParser.is_code_file(DocumentType.PDF) is False

    @pytest.mark.asyncio
    async def test_parse_txt(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Hello, world!")
            path = f.name
        try:
            parser = DocumentParser()
            result = await parser.parse(path)
            assert result.content == "Hello, world!"
            assert result.doc_type == DocumentType.TXT
            assert result.chars == 13
        finally:
            Path(path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_parse_markdown(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# Title\n\nSome content")
            path = f.name
        try:
            parser = DocumentParser()
            result = await parser.parse(path)
            assert "# Title" in result.content
            assert result.doc_type == DocumentType.MARKDOWN
        finally:
            Path(path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_parse_code(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def hello():\n    print('hi')")
            path = f.name
        try:
            parser = DocumentParser()
            result = await parser.parse(path)
            assert "def hello" in result.content
            assert result.doc_type == DocumentType.CODE_PYTHON
        finally:
            Path(path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_parse_nonexistent(self):
        parser = DocumentParser()
        result = await parser.parse("/nonexistent/file.txt")
        assert "Error" in result.error or "not found" in result.error

    @pytest.mark.asyncio
    async def test_parse_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "a.txt").write_text("file a")
            Path(tmpdir, "b.py").write_text("print('b')")
            Path(tmpdir, "c.md").write_text("# C")
            parser = DocumentParser()
            results = await parser.parse_directory(tmpdir)
            assert len(results) == 3


# ── Chunker ──

class TestChunker:
    @pytest.mark.asyncio
    async def test_semantic_chunking(self):
        text = "# Section 1\n\n" + "A" * 500 + "\n\n# Section 2\n\n" + "B" * 500
        result = ParseResult(file_path="test.md", file_name="test.md",
                             doc_type=DocumentType.MARKDOWN, content=text)
        chunker = Chunker(strategy="semantic", chunk_size=300)
        chunks = await chunker.chunk(result)
        assert len(chunks) >= 2

    @pytest.mark.asyncio
    async def test_fixed_chunking(self):
        text = "A" * 2000
        result = ParseResult(file_path="test.txt", file_name="test.txt",
                             doc_type=DocumentType.TXT, content=text)
        chunker = Chunker(strategy="fixed", chunk_size=500, chunk_overlap=50)
        chunks = await chunker.chunk(result)
        assert len(chunks) >= 4

    @pytest.mark.asyncio
    async def test_code_chunking_python(self):
        text = "def func1():\n    pass\n\ndef func2():\n    pass"
        result = ParseResult(file_path="test.py", file_name="test.py",
                             doc_type=DocumentType.CODE_PYTHON, content=text)
        chunker = Chunker(strategy="code", chunk_size=1000)
        chunks = await chunker.chunk(result)
        assert len(chunks) >= 2

    @pytest.mark.asyncio
    async def test_empty_content(self):
        result = ParseResult(file_path="empty.txt", file_name="empty.txt",
                             doc_type=DocumentType.TXT, content="")
        chunker = Chunker()
        chunks = await chunker.chunk(result)
        assert len(chunks) == 0

    @pytest.mark.asyncio
    async def test_error_content(self):
        result = ParseResult(file_path="err.txt", file_name="err.txt",
                             doc_type=DocumentType.TXT, content="test",
                             error="Failed")
        chunker = Chunker()
        chunks = await chunker.chunk(result)
        assert len(chunks) == 0


# ── EmbeddingClient ──

class TestEmbeddingClient:
    def test_init(self):
        client = EmbeddingClient(base_url="http://localhost:8000/v1", model="BGE-M3")
        assert client.model == "BGE-M3"
        assert client.max_retries == 3

    @pytest.mark.asyncio
    async def test_health_no_server(self):
        client = EmbeddingClient(base_url="http://localhost:19999/v1", timeout=1.0)
        ok = await client.health()
        assert ok is False

    @pytest.mark.asyncio
    async def test_embed_empty(self):
        client = EmbeddingClient()
        results = await client.embed_batch([])
        assert results == []

    @pytest.mark.asyncio
    async def test_close(self):
        client = EmbeddingClient()
        _ = client.client  # trigger lazy init
        await client.close()
        assert client._client is None


# ── VectorStore ──

class TestVectorStore:
    def test_create_and_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(tmpdir)
            assert store.count() == 0

    def test_add_and_search(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(tmpdir, dimension=4)
            store.add("chunk1", [1.0, 0.0, 0.0, 0.0], "text about cats")
            store.add("chunk2", [0.0, 1.0, 0.0, 0.0], "text about dogs")
            results = store.search([1.0, 0.0, 0.0, 0.0], top_k=5, threshold=0.0)
            assert len(results) >= 1
            assert results[0]["id"] == "chunk1"

    def test_add_batch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(tmpdir, dimension=4)
            records = [
                {"id": "c1", "vector": [1.0, 0.0, 0.0, 0.0], "text": "a", "doc_path": "", "doc_name": "", "doc_type": "", "chunk_index": 0, "metadata": {}},
                {"id": "c2", "vector": [0.0, 1.0, 0.0, 0.0], "text": "b", "doc_path": "", "doc_name": "", "doc_type": "", "chunk_index": 0, "metadata": {}},
            ]
            store.add_batch(records)
            assert store.count() == 2

    def test_clear(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(tmpdir, dimension=4)
            store.add("c1", [1.0, 0.0, 0.0, 0.0], "text")
            store.clear()
            assert store.count() == 0

    def test_close(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(tmpdir)
            store.close()
            assert store._db is None


# ── MetadataStore ──

class TestMetadataStore:
    def test_add_and_get_document(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MetadataStore(str(Path(tmpdir) / "meta.db"))
            store.add_document("doc1", "/path/to/file.pdf", "file.pdf", "pdf", 1024)
            doc = store.get_document("doc1")
            assert doc is not None
            assert doc["file_name"] == "file.pdf"

    def test_get_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MetadataStore(str(Path(tmpdir) / "meta.db"))
            assert store.get_document("nonexistent") is None

    def test_list_documents(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MetadataStore(str(Path(tmpdir) / "meta.db"))
            store.add_document("doc1", "/a.pdf", "a.pdf", "pdf")
            store.add_document("doc2", "/b.md", "b.md", "markdown")
            docs = store.list_documents()
            assert len(docs) == 2

    def test_add_chunk(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MetadataStore(str(Path(tmpdir) / "meta.db"))
            store.add_document("doc1", "/f.txt", "f.txt", "txt")
            store.add_chunk("chunk1", "doc1", "/f.txt", 0, "hello world", 5)
            chunks = store.get_chunks_by_doc("doc1")
            assert len(chunks) == 1
            assert chunks[0]["text"] == "hello world"

    def test_delete_document(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MetadataStore(str(Path(tmpdir) / "meta.db"))
            store.add_document("doc1", "/f.txt", "f.txt", "txt")
            store.add_chunk("c1", "doc1", "/f.txt", 0, "text")
            store.delete_document("doc1")
            assert store.get_document("doc1") is None
            assert len(store.get_chunks_by_doc("doc1")) == 0

    def test_update_chunk_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MetadataStore(str(Path(tmpdir) / "meta.db"))
            store.add_document("doc1", "/f.txt", "f.txt", "txt")
            store.update_chunk_count("doc1", 10, 5000)
            doc = store.get_document("doc1")
            assert doc["chunk_count"] == 10

    def test_doc_and_chunk_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MetadataStore(str(Path(tmpdir) / "meta.db"))
            store.add_document("d1", "/a.txt", "a.txt", "txt")
            store.add_document("d2", "/b.txt", "b.txt", "txt")
            store.add_chunk("c1", "d1", "/a.txt", 0, "text")
            assert store.doc_count() == 2
            assert store.chunk_count() == 1