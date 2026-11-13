# Issue #117: citation audit trail tests.

from __future__ import annotations

import os

import pytest

from fusion_rag.api.routes_search import _citation_audit_async
from fusion_rag.engine.citation_audit import CitationAuditStore
from fusion_rag.engine.runtime_config import get_runtime_config, reset_runtime_config


def _store(tmp_path):
    return CitationAuditStore(str(tmp_path / "citation_audit.db"))


def _results(n=3):
    return [
        {"id": f"doc1_{i}", "chunk_id": f"doc1_{i}", "doc_id": "doc1", "score": 0.9 - i * 0.1}
        for i in range(n)
    ]


class TestCitationAuditStore:
    def test_record_and_reconstruct_with_rewrites(self, tmp_path):
        store = _store(tmp_path)
        root_id = store.record_retrieval("kb1", "original query", ["rewrite one", "rewrite two"], _results())
        trace = store.get_trace("kb1", root_id)
        assert trace is not None
        assert trace["query_id"] == root_id
        queries = [c["query"] for c in trace["query_chain"]]
        assert queries[0] == "original query"
        assert "rewrite one" in queries and "rewrite two" in queries
        depths = [c["depth"] for c in trace["query_chain"]]
        assert depths == sorted(depths) and depths[0] == 0
        # citations link to the FINAL query version
        final_row = next(c for c in trace["query_chain"] if c["id"] == trace["final_query_id"])
        assert final_row["query"] == "rewrite two"
        assert len(trace["citations"]) == 3
        assert trace["citations"][0]["chunk_id"] == "doc1_0"
        assert trace["citations"][0]["rank"] == 1
        scores = [c["similarity"] for c in trace["citations"]]
        assert scores == sorted(scores, reverse=True)

    def test_record_without_rewrites_links_root(self, tmp_path):
        store = _store(tmp_path)
        root_id = store.record_retrieval("kb1", "plain query", [], _results(2))
        trace = store.get_trace("kb1", root_id)
        assert trace["final_query_id"] == root_id
        assert len(trace["query_chain"]) == 1
        assert len(trace["citations"]) == 2

    def test_get_trace_missing_returns_none(self, tmp_path):
        store = _store(tmp_path)
        assert store.get_trace("kb1", 999) is None

    def test_list_traces(self, tmp_path):
        store = _store(tmp_path)
        store.record_retrieval("kb1", "q1", ["r1"], _results(2))
        store.record_retrieval("kb1", "q2", [], _results(1))
        traces = store.list_traces("kb1")
        assert len(traces) == 2
        assert traces[0]["query"] == "q2"
        assert traces[0]["rewrite_count"] == 0
        assert traces[1]["rewrite_count"] == 1
        assert traces[1]["citation_count"] == 2

    def test_kb_isolation(self, tmp_path):
        store = _store(tmp_path)
        rid = store.record_retrieval("kbA", "qa", [], _results(1))
        assert store.get_trace("kbB", rid) is None
        assert store.list_traces("kbB") == []


class TestAuditHook:
    async def test_flag_off_writes_nothing(self, tmp_path, monkeypatch):
        monkeypatch.delenv("FUSION_RAG_CITATION_AUDIT", raising=False)
        monkeypatch.setattr("fusion_rag.api.routes._get_kb_storage_path", lambda kb_id: str(tmp_path / kb_id))
        reset_runtime_config()
        try:
            await _citation_audit_async("kb1", "q", [], _results())
            assert not list(tmp_path.rglob("citation_audit.db"))
        finally:
            reset_runtime_config()

    async def test_flag_on_writes_rows(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FUSION_RAG_CITATION_AUDIT", "1")
        monkeypatch.setattr("fusion_rag.api.routes._get_kb_storage_path", lambda kb_id: str(tmp_path / kb_id))
        reset_runtime_config()
        try:
            assert get_runtime_config().citation_audit is True
            (tmp_path / "kb1").mkdir(parents=True, exist_ok=True)
            await _citation_audit_async("kb1", "q", ["rw"], _results(2))
            db = tmp_path / "kb1" / "citation_audit.db"
            assert db.exists()
            store = CitationAuditStore(str(db))
            traces = store.list_traces("kb1")
            assert len(traces) == 1
            assert traces[0]["citation_count"] == 2
        finally:
            reset_runtime_config()

    async def test_write_failure_never_raises(self, tmp_path, monkeypatch):
        # a broken storage path must not break the search response
        monkeypatch.setenv("FUSION_RAG_CITATION_AUDIT", "1")
        monkeypatch.setattr("fusion_rag.api.routes._get_kb_storage_path", lambda kb_id: "/dev/null/forbidden")
        reset_runtime_config()
        try:
            await _citation_audit_async("kb1", "q", [], _results())
        finally:
            reset_runtime_config()

    @pytest.mark.parametrize("env", ["0", "1"])
    def test_env_parsing(self, env, monkeypatch):
        monkeypatch.setenv("FUSION_RAG_CITATION_AUDIT", env)
        reset_runtime_config()
        try:
            assert get_runtime_config().citation_audit == (env == "1")
        finally:
            reset_runtime_config()
            os.environ.pop("FUSION_RAG_CITATION_AUDIT", None)
