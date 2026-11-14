"""Issue #130 (R8): chunk-profile resolution + KB wiring tests."""

from __future__ import annotations

import pytest

from fusion_rag.engine.chunker import Chunker


def test_profile_values_bench_derived():
    # markdown/prose = bench winner 512/64; code/table finer (structural).
    assert Chunker.resolve_profile("markdown") == {"chunk_size": 512, "chunk_overlap": 64}
    assert Chunker.resolve_profile("prose") == {"chunk_size": 512, "chunk_overlap": 64}
    assert Chunker.resolve_profile("code")["chunk_size"] < 512
    assert Chunker.resolve_profile("table")["chunk_size"] < Chunker.resolve_profile("code")["chunk_size"]


def test_profile_empty_returns_none():
    assert Chunker.resolve_profile("") is None
    assert Chunker.resolve_profile(None) is None


def test_profile_unknown_raises():
    with pytest.raises(ValueError, match="unknown chunk_profile"):
        Chunker.resolve_profile("nano")


def test_profile_case_insensitive():
    assert Chunker.resolve_profile("CODE") == Chunker.resolve_profile("code")


def test_kb_create_resolves_profile():
    import tempfile

    from fusion_rag.engine.knowledge_base import KnowledgeBaseManager

    with tempfile.TemporaryDirectory() as d:
        mgr = KnowledgeBaseManager(storage_dir=d)
        kb = mgr.create(name="t", chunk_profile="code")
        assert kb.config.chunk_profile == "code"
        assert kb.config.chunk_size == 384
        assert kb.config.chunk_overlap == 48
        # explicit overrides win over profile
        kb2 = mgr.create(name="t2", chunk_profile="code", chunk_size=300)
        assert kb2.config.chunk_size == 300
        assert kb2.config.chunk_overlap == 48


def test_kb_create_rejects_unknown_profile():
    import tempfile

    from fusion_rag.engine.knowledge_base import KnowledgeBaseManager

    with tempfile.TemporaryDirectory() as d:
        mgr = KnowledgeBaseManager(storage_dir=d)
        with pytest.raises(ValueError, match="unknown chunk_profile"):
            mgr.create(name="t", chunk_profile="nano")
