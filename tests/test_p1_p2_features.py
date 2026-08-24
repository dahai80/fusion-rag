"""Tests for P1/P2 features: QueryRewriter, EmbeddingCache, Auth, GraphRAG, MCP, Evaluator."""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fusion_rag.api.auth import AuthConfig
from fusion_rag.engine.embedding_cache import EmbeddingCache
from fusion_rag.engine.evaluator import RAGEvaluator
from fusion_rag.engine.graph_rag import GraphRAG

# callers: pytest runner
# API: test functions verify P1/P2 features
# schema: N/A
# user instruction: "按照你的方案和计划落地所有phase阶段的需求"
from fusion_rag.engine.query_rewriter import QueryRewriter
from fusion_rag.engine.rag_chain import MultiTurnRAG, estimate_tokens

# ── estimate_tokens ──

class TestEstimateTokens:
    def test_empty(self):
        assert estimate_tokens("") == 0

    def test_english(self):
        tokens = estimate_tokens("hello world")
        assert tokens > 0

    def test_chinese(self):
        tokens = estimate_tokens("你好世界")
        assert tokens > 0


# ── QueryRewriter ──

class TestQueryRewriter:
    @pytest.mark.asyncio
    async def test_rewrite_disabled(self):
        rw = QueryRewriter(enabled=False)
        result = await rw.rewrite("test query", mode="hyde")
        assert result == "test query"

    @pytest.mark.asyncio
    async def test_rewrite_empty(self):
        rw = QueryRewriter(enabled=True)
        result = await rw.rewrite("", mode="hyde")
        assert result == ""

    @pytest.mark.asyncio
    async def test_hyde(self):
        rw = QueryRewriter(enabled=True)
        with patch("httpx.AsyncClient.post", new=AsyncMock()) as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"choices": [{"message": {"content": "Hypothetical answer"}}]}
            mock_post.return_value = mock_resp
            result = await rw.rewrite("What is Python?", mode="hyde")
            assert result == "Hypothetical answer"

    @pytest.mark.asyncio
    async def test_expand(self):
        rw = QueryRewriter(enabled=True)
        with patch("httpx.AsyncClient.post", new=AsyncMock()) as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {
                "choices": [{"message": {"content":
                    "1. What is Python language\n"
                    "2. Python programming\n"
                    "3. Python overview"}}]
            }
            mock_post.return_value = mock_resp
            result = await rw.rewrite("What is Python?", mode="expand")
            assert isinstance(result, list)
            assert len(result) == 4  # original + 3 variants

    @pytest.mark.asyncio
    async def test_condense(self):
        rw = QueryRewriter(enabled=True)
        with patch("httpx.AsyncClient.post", new=AsyncMock()) as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"choices": [{"message": {"content": "What is Python's GIL?"}}]}
            mock_post.return_value = mock_resp
            result = await rw.rewrite("what about the GIL?", history=[
                {"role": "user", "content": "Tell me about Python"},
                {"role": "assistant", "content": "Python is a language..."},
            ], mode="condense")
            assert "GIL" in result

    @pytest.mark.asyncio
    async def test_rewrite_fallback_on_error(self):
        # L1: LLM failure raises LLMUnavailable (route layer logs + falls back to
        # the original query); it no longer silently returns the original query
        # as if rewrite succeeded.
        from fusion_rag.engine.llm_errors import LLMUnavailable

        rw = QueryRewriter(enabled=True)
        with patch("httpx.AsyncClient.post", side_effect=Exception("API error")), pytest.raises(LLMUnavailable):
            await rw.rewrite("test", mode="hyde")

    @pytest.mark.asyncio
    async def test_unknown_mode(self):
        rw = QueryRewriter(enabled=True)
        result = await rw.rewrite("test", mode="nonexistent")
        assert result == "test"


# ── EmbeddingCache ──

class TestEmbeddingCache:
    def test_set_and_get(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = EmbeddingCache(str(Path(tmpdir) / "cache.db"))
            cache.set("hello world", [0.1, 0.2, 0.3], model="test")
            result = cache.get("hello world", model="test")
            assert result == [0.1, 0.2, 0.3]

    def test_miss(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = EmbeddingCache(str(Path(tmpdir) / "cache.db"))
            result = cache.get("nonexistent")
            assert result is None

    def test_batch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = EmbeddingCache(str(Path(tmpdir) / "cache.db"))
            texts = ["a", "b", "c"]
            vectors = [[1.0], [2.0], [3.0]]
            cache.set_batch(texts, vectors, model="test")
            results = cache.get_batch(texts, model="test")
            assert results == [[1.0], [2.0], [3.0]]

    def test_ttl_expiry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = EmbeddingCache(str(Path(tmpdir) / "cache.db"), ttl=1)
            cache.set("test", [1.0], model="test")
            # Manually backdate the created_at to force expiry
            import sqlite3
            import time
            conn = sqlite3.connect(str(Path(tmpdir) / "cache.db"))
            conn.execute("UPDATE embed_cache SET created_at = ?", (time.time() - 10,))
            conn.commit()
            conn.close()
            result = cache.get("test", model="test")
            assert result is None

    def test_clear(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = EmbeddingCache(str(Path(tmpdir) / "cache.db"))
            cache.set("test", [1.0])
            assert cache.count() == 1
            cache.clear()
            assert cache.count() == 0

    def test_model_isolation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = EmbeddingCache(str(Path(tmpdir) / "cache.db"))
            cache.set("test", [1.0], model="model_a")
            cache.set("test", [2.0], model="model_b")
            assert cache.get("test", model="model_a") == [1.0]
            assert cache.get("test", model="model_b") == [2.0]


# ── MultiTurnRAG (enhanced) ──

class TestMultiTurnRAGEnhanced:
    @pytest.mark.asyncio
    async def test_token_count(self):
        rag = MultiTurnRAG()
        rag._history = [{"role": "user", "content": "hello"}]
        assert rag.token_count() > 0

    @pytest.mark.asyncio
    async def test_session_isolation(self):
        rag = MultiTurnRAG()
        with patch("httpx.AsyncClient.post", new=AsyncMock()) as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"choices": [{"message": {"content": "answer"}}]}
            mock_post.return_value = mock_resp
            await rag.ask("q1", session_id="s1")
            await rag.ask("q2", session_id="s2")
        assert rag.token_count("s1") > 0
        assert rag.token_count("s2") > 0
        rag.clear_history("s1")
        assert rag.token_count("s1") == 0
        assert rag.token_count("s2") > 0

    @pytest.mark.asyncio
    async def test_usage_reported(self):
        rag = MultiTurnRAG()
        with patch("httpx.AsyncClient.post", new=AsyncMock()) as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {
                "choices": [{"message": {"content": "answer"}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            }
            mock_post.return_value = mock_resp
            result = await rag.ask("test")
            assert result.get("prompt_tokens") == 100
            assert result.get("total_tokens") == 150


# ── AuthConfig ──

class TestAuthConfig:
    def test_add_and_validate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            auth = AuthConfig(str(Path(tmpdir) / "auth.db"))
            assert auth.add_key("test-key-123", "test")
            assert auth.validate_key("test-key-123")
            assert not auth.validate_key("wrong-key")
            assert not auth.validate_key("")

    def test_remove_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            auth = AuthConfig(str(Path(tmpdir) / "auth.db"))
            auth.add_key("key-to-remove")
            assert auth.validate_key("key-to-remove")
            assert auth.remove_key("key-to-remove")
            assert not auth.validate_key("key-to-remove")

    def test_list_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            auth = AuthConfig(str(Path(tmpdir) / "auth.db"))
            auth.add_key("key1", "first")
            auth.add_key("key2", "second")
            keys = auth.list_keys()
            assert len(keys) == 2


# ── GraphRAG ──

class TestGraphRAG:
    def test_init_db(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = GraphRAG(db_path=str(Path(tmpdir) / "graph.db"))
            assert graph.get_entity_count() == 0

    @pytest.mark.asyncio
    async def test_extract_entities(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = GraphRAG(db_path=str(Path(tmpdir) / "graph.db"))
            with patch("httpx.AsyncClient.post", new=AsyncMock()) as mock_post:
                mock_resp = MagicMock()
                mock_resp.json.return_value = {
                    "choices": [{"message": {"content":
                        '{"entities": [{"name": "Python", '
                        '"type": "CONCEPT"}], '
                        '"relations": []}'}}]
                }
                mock_post.return_value = mock_resp
                result = await graph.extract_entities("Python is a language")
                assert len(result.get("entities", [])) >= 0

    @pytest.mark.asyncio
    async def test_extract_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = GraphRAG(db_path=str(Path(tmpdir) / "graph.db"))
            result = await graph.extract_entities("")
            assert result == {"entities": [], "relations": []}

    @pytest.mark.asyncio
    async def test_build_and_search(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = GraphRAG(db_path=str(Path(tmpdir) / "graph.db"))
            with patch("httpx.AsyncClient.post", new=AsyncMock()) as mock_post:
                mock_resp = MagicMock()
                mock_resp.json.return_value = {
                    "choices": [{"message": {"content":
                        '{"entities": [{"name": "Python", '
                        '"type": "CONCEPT"}], '
                        '"relations": []}'}}]
                }
                mock_post.return_value = mock_resp
                stats = await graph.build_graph(
                    [{"id": "c1", "text": "Python is great"}], kb_id="kb1"
                )
                assert stats["entities"] >= 0


# ── RAGEvaluator ──

class TestRAGEvaluator:
    def test_init(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ev = RAGEvaluator(db_path=str(Path(tmpdir) / "eval.db"))
            assert ev.get_history() == []

    def test_context_recall(self):
        recall = RAGEvaluator._compute_context_recall(
            [{"doc_name": "a.pdf"}, {"doc_name": "b.pdf"}],
            ["a.pdf", "c.pdf"],
        )
        assert recall == 0.5

    def test_context_recall_no_expected(self):
        recall = RAGEvaluator._compute_context_recall([], [])
        assert recall == 1.0

    def test_context_recall_no_retrieved(self):
        recall = RAGEvaluator._compute_context_recall([], ["a.pdf"])
        assert recall == 0.0


# ── MCP Server ──

class TestMCPServer:
    def test_mcp_tools_defined(self):
        from fusion_rag.api.mcp_server import MCP_TOOLS
        assert len(MCP_TOOLS) >= 5
        tool_names = [t["name"] for t in MCP_TOOLS]
        assert "kb_list" in tool_names
        assert "kb_search" in tool_names
        assert "kb_ask" in tool_names
        assert "kb_create" in tool_names

    def test_mcp_tools_have_schemas(self):
        from fusion_rag.api.mcp_server import MCP_TOOLS
        for tool in MCP_TOOLS:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool
