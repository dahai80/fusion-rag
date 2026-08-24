"""D-H3 silent-failure guard tests — verify empty LLM content triggers fail/degrade, not silent success.

Phase C fusion-core migration: success-path empty-content guards across 5 migrated LLM modules.
Each test mocks httpx at class level (pooled instance inherits the patch) and asserts the guard
fires (exception or degraded return) instead of returning empty content as a valid result.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fusion_rag.engine.graph_rag import GraphRAG
from fusion_rag.engine.rag_chain import MultiTurnRAG
from fusion_rag.engine.reranker import Reranker
from fusion_rag.engine.streaming import MetadataExtractor


class TestDH3EmptyContent:
    @pytest.mark.asyncio
    async def test_multiturnrag_ask_empty_content_degrades(self):
        rag = MultiTurnRAG()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": ""}}]}
        with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_resp)):
            result = await rag.ask("test question", "context")
            assert "Error" in result["answer"]
            assert "empty_content" in result["answer"]

    @pytest.mark.asyncio
    async def test_reranker_batch_score_empty_content_fallback(self):
        r = Reranker()
        docs = [{"id": "1", "text": "apple", "score": 7.0}]
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": ""}}]}
        with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_resp)):
            scores = await r._batch_score("query", docs)
            assert scores == [7.0]

    @pytest.mark.asyncio
    async def test_graphrag_extract_entities_empty_content_empty_result(self, tmp_path):
        graph = GraphRAG(db_path=str(tmp_path / "graph.db"))
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": ""}}]}
        with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_resp)):
            result = await graph.extract_entities("some text")
            assert result == {"entities": [], "relations": []}

    @pytest.mark.asyncio
    async def test_metadata_extractor_empty_content_default(self):
        extractor = MetadataExtractor()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": ""}}]}
        with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_resp)):
            meta = await extractor.extract("some text", "doc.md")
            assert meta["language"] == "unknown"

    @pytest.mark.asyncio
    async def test_multiturnrag_ask_whitespace_content_degrades(self):
        rag = MultiTurnRAG()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "   \n  "}}]}
        with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_resp)):
            result = await rag.ask("test question", "context")
            assert "empty_content" in result["answer"]
