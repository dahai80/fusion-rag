"""D-H3 silent-failure guard tests — empty LLM content must fail visibly, not degrade to a valid result.

L1 (systemic) rework: the prior version asserted "empty content → fabricated
default" (neutral score / empty graph / default metadata / error-string answer).
That is exactly the error-as-success anti-pattern L1 removes. These tests now
assert the L1 contract: empty/whitespace LLM content propagates as
LLMUnavailable so the route layer can map it to an error response — never a
200 that looks like a successful (if empty) result.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fusion_rag.engine.graph_rag import GraphRAG
from fusion_rag.engine.llm_errors import LLMUnavailable
from fusion_rag.engine.rag_chain import MultiTurnRAG
from fusion_rag.engine.reranker import Reranker
from fusion_rag.engine.streaming import MetadataExtractor


def _empty_content_resp():
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"choices": [{"message": {"content": ""}}]}
    return mock_resp


class TestDH3EmptyContent:
    @pytest.mark.asyncio
    async def test_multiturnrag_ask_empty_content_propagates(self):
        # L1/L3: empty content raises LLMUnavailable, AND history is not
        # poisoned with an "Error: ..." assistant turn.
        rag = MultiTurnRAG()
        with (
            patch("httpx.AsyncClient.post", new=AsyncMock(return_value=_empty_content_resp())),
            pytest.raises(LLMUnavailable),
        ):
            await rag.ask("test question", "context")
        assert rag._history == []

    @pytest.mark.asyncio
    async def test_reranker_batch_score_empty_content_propagates(self):
        # L1: empty content raises LLMUnavailable, not a neutral-score array.
        r = Reranker()
        docs = [{"id": "1", "text": "apple", "score": 7.0}]
        with (
            patch("httpx.AsyncClient.post", new=AsyncMock(return_value=_empty_content_resp())),
            pytest.raises(LLMUnavailable),
        ):
            await r._batch_score("query", docs)

    @pytest.mark.asyncio
    async def test_graphrag_extract_entities_empty_content_propagates(self, tmp_path):
        # L1: empty content raises LLMUnavailable, not {"entities":[],"relations":[]}.
        graph = GraphRAG(db_path=str(tmp_path / "graph.db"))
        with (
            patch("httpx.AsyncClient.post", new=AsyncMock(return_value=_empty_content_resp())),
            pytest.raises(LLMUnavailable),
        ):
            await graph.extract_entities("some text")

    @pytest.mark.asyncio
    async def test_metadata_extractor_empty_content_propagates(self):
        # L1: empty content raises LLMUnavailable, not default metadata.
        extractor = MetadataExtractor()
        with (
            patch("httpx.AsyncClient.post", new=AsyncMock(return_value=_empty_content_resp())),
            pytest.raises(LLMUnavailable),
        ):
            await extractor.extract("some text", "doc.md")

    @pytest.mark.asyncio
    async def test_multiturnrag_ask_whitespace_content_propagates(self):
        # L1: whitespace-only content is treated as empty → LLMUnavailable.
        rag = MultiTurnRAG()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "   \n  "}}]}
        with (
            patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_resp)),
            pytest.raises(LLMUnavailable),
        ):
            await rag.ask("test question", "context")
        assert rag._history == []
