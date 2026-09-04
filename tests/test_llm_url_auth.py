"""Issue #72 — LLM engines must honor embed.base_url + embed.api_key.

Regression: Contextualizer / QueryRewriter / Reranker previously hardcoded the
gateway URL (http://127.0.0.1:11432/v1) and sent no auth headers, so a non-default
FUSION_MLX_URL + key (e.g. fusion-mlx directly at :11434 with a different key)
401'd on every chat-completions call. The route layer now passes embed.base_url
+ embed.api_key into every LLM engine it constructs. This test locks that wiring
without a live fusion-mlx — it captures the Authorization header + target URL
each engine posts to.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from fusion_rag.engine.contextualizer import Contextualizer
from fusion_rag.engine.query_rewriter import QueryRewriter
from fusion_rag.engine.reranker import Reranker


class _CapturingResponse:
    status_code = 200

    def json(self):
        # chat/completions shape: choices[0].message.content
        return {"choices": [{"message": {"content": "ctx"}}]}

    def raise_for_status(self):
        return None


def _captured_post(url_holder, header_holder):
    async def _post(url, json=None, headers=None):
        url_holder["url"] = url
        header_holder["headers"] = headers or {}
        return _CapturingResponse()

    return _post


@pytest.mark.asyncio
async def test_contextualizer_uses_passed_url_and_api_key():
    url_h, hdr_h = {}, {}
    rer = Contextualizer(mlx_url="http://127.0.0.1:11434/v1", api_key="fg-admin-key")
    with patch("fusion_rag.engine.contextualizer.get_async_client") as mock_pool, \
            patch("fusion_rag.engine.contextualizer.with_retry") as mock_retry:
        mock_client = mock_pool.return_value
        mock_client.post = _captured_post(url_h, hdr_h)
        async def _run(fn, **_):
            return await fn()

        mock_retry.side_effect = _run
        await rer._generate_context(mock_client, "chunk", "doc")
    assert "11434" in url_h["url"], f"contextualizer ignored base_url: {url_h}"
    assert hdr_h["headers"].get("Authorization") == "Bearer fg-admin-key"


@pytest.mark.asyncio
async def test_query_rewriter_uses_passed_url_and_api_key():
    url_h, hdr_h = {}, {}
    rer = QueryRewriter(mlx_url="http://127.0.0.1:11434/v1", api_key="fg-admin-key")
    with patch("fusion_rag.engine.query_rewriter.get_async_client") as mock_pool, \
            patch("fusion_rag.engine.query_rewriter.with_retry") as mock_retry:
        mock_client = mock_pool.return_value
        mock_client.post = _captured_post(url_h, hdr_h)
        async def _run(fn, **_):
            return await fn()

        mock_retry.side_effect = _run
        out = await rer.rewrite("query", mode="hyde")
    assert "11434" in url_h["url"], f"rewriter ignored base_url: {url_h}"
    assert hdr_h["headers"].get("Authorization") == "Bearer fg-admin-key"
    assert out == "ctx"


@pytest.mark.asyncio
async def test_reranker_uses_passed_url_and_api_key():
    url_h, hdr_h = {}, {}
    rer = Reranker(mlx_url="http://127.0.0.1:11434/v1", api_key="fg-admin-key")
    with patch("fusion_rag.engine.reranker.get_async_client") as mock_pool, \
            patch("fusion_rag.engine.reranker.with_retry") as mock_retry:
        mock_client = mock_pool.return_value
        mock_client.post = _captured_post(url_h, hdr_h)
        async def _run(fn, **_):
            return await fn()

        mock_retry.side_effect = _run
        await rer._batch_score("query", [{"text": "doc"}])
    assert "11434" in url_h["url"], f"reranker ignored base_url: {url_h}"
    assert hdr_h["headers"].get("Authorization") == "Bearer fg-admin-key"


@pytest.mark.asyncio
async def test_contextualizer_no_key_sends_no_auth_header():
    url_h, hdr_h = {}, {}
    rer = Contextualizer(mlx_url="http://127.0.0.1:11432/v1", api_key="")
    with patch("fusion_rag.engine.contextualizer.get_async_client") as mock_pool, \
            patch("fusion_rag.engine.contextualizer.with_retry") as mock_retry:
        mock_client = mock_pool.return_value
        mock_client.post = _captured_post(url_h, hdr_h)
        async def _run(fn, **_):
            return await fn()

        mock_retry.side_effect = _run
        await rer._generate_context(mock_client, "chunk", "doc")
    assert "Authorization" not in hdr_h["headers"]
