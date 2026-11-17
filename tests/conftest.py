"""Shared test fixtures.

R10 gap exposed by the live embeddings endpoint: EmbeddingClient
.embed_batch_sparse (issue #132) makes its own HTTP call instead of
delegating to embed_batch, so test files that mock embed_batch alone
silently fall through to the REAL embeddings endpoint at ingest while the
query side stays mocked — real BGE-M3 vectors vs constant mock vectors
score below the KB similarity_threshold (0.3) and every search returns [].
(The suite passed while the endpoint was unreachable only because
embed_batch_sparse exhausted retries and fell back to the mocked
embed_batch — an accident, not coverage.)

This autouse fixture makes embed_batch_sparse delegate to whatever
embed_batch resolves to AT CALL TIME (the per-test mock when one is
active), restoring the intended mock semantics for every pre-R10 fixture.
Sparse-specific tests (tests/test_sparse_channel.py) are excluded — they
exercise the real method against MockTransport.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from fusion_rag.embed.client import EmbeddingClient


@pytest.fixture(autouse=True)
def _sparse_delegates_to_embed_batch(request):
    # tests/ is imported without a package prefix in some layouts — match
    # the module NAME, not its dotted path.
    if request.module.__name__.split(".")[-1] == "test_sparse_channel":
        yield
        return

    async def _delegate(self, texts):
        return await self.embed_batch(texts), [None] * len(texts)

    with patch.object(EmbeddingClient, "embed_batch_sparse", _delegate):
        yield
