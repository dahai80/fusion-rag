"""L5-L18 logic-bug fix tests -- lock the new contracts from the audit remediation.

Each test anchors to one audit finding so a regression that reverts the fix
fails loudly here.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fusion_rag.engine.ast_chunker import ASTChunker
from fusion_rag.engine.bm25_index import BM25Index
from fusion_rag.engine.graph_rag import GraphRAG
from fusion_rag.engine.search_template import SearchTemplate, SearchTemplateManager
from fusion_rag.engine.version_manager import VersionManager

# ── L5: bm25 remove_document must match doc_path exactly, not substring ──


class TestBM25ExactRemoval:
    def test_remove_exact_doc_path_only(self, tmp_path):
        idx = BM25Index(str(tmp_path / "bm25.db"))
        # chunk A belongs to /docs/a.txt
        idx.add_documents(
            [
                {"id": "docA_0", "text": "alpha beta", "doc_path": "/docs/a.txt"},
                # chunk B belongs to /docs/b.txt but its TEXT mentions /docs/a.txt
                {"id": "docB_0", "text": "see also /docs/a.txt for alpha", "doc_path": "/docs/b.txt"},
            ]
        )
        assert idx.count() == 2
        removed = idx.remove_document("/docs/a.txt")
        # Only docA_0 must go; docB_0's text containing the path must survive.
        assert removed == 1
        assert idx.count() == 1
        remaining = {did for did, _ in idx._doc_texts.items()}
        assert "docB_0" in remaining
        assert "docA_0" not in remaining

    def test_remove_unknown_path_returns_zero(self, tmp_path):
        idx = BM25Index(str(tmp_path / "bm25.db"))
        idx.add_documents([{"id": "x_0", "text": "hello", "doc_path": "/x.txt"}])
        assert idx.remove_document("/nonexistent.txt") == 0
        assert idx.count() == 1


# ── L6: bm25 _save_to_db failure must propagate, not be swallowed ──


class TestBM25SavePropagates:
    def test_save_failure_raises(self, tmp_path):
        idx = BM25Index(str(tmp_path / "bm25.db"))
        with patch.object(idx, "_db", MagicMock()):
            idx._db.execute = MagicMock(side_effect=sqlite3_operational_error)
            with pytest.raises(Exception):
                idx._save_to_db()


def sqlite3_operational_error(*args, **kwargs):
    import sqlite3

    raise sqlite3.OperationalError("simulated disk I/O error")


# ── L8: same-second snapshots must not collide ──


class TestVersionSnapshotCollision:
    def test_same_second_snapshots_distinct(self, tmp_path):
        storage = tmp_path / "kb1"
        storage.mkdir()
        (storage / "metadata.db").write_text("x")
        vm = VersionManager(str(tmp_path / "versions.db"))
        with patch("fusion_rag.engine.version_manager.time.time", return_value=1700000000.0):
            s1 = vm.create_snapshot("kb1", str(storage), "first")
            s2 = vm.create_snapshot("kb1", str(storage), "second")
        assert s1["version_id"] != s2["version_id"]
        assert Path(s1["snapshot_path"]).exists()
        assert Path(s2["snapshot_path"]).exists()


# ── L13: graph_rag _parse_extraction must handle LLM preamble ──


class TestGraphParseExtraction:
    def test_preamble_before_json(self):
        content = 'Here is the JSON:\n{"entities": [{"name": "Python", "type": "CONCEPT"}], "relations": []}'
        result = GraphRAG._parse_extraction(content)
        assert len(result.get("entities", [])) == 1
        assert result["entities"][0]["name"] == "Python"

    def test_trailing_notes_after_json(self):
        content = '{"entities": [{"name": "Py", "type": "CONCEPT"}], "relations": []} notes: done'
        result = GraphRAG._parse_extraction(content)
        assert len(result.get("entities", [])) == 1

    def test_two_objects_picks_first_balanced(self):
        content = '{"entities": [{"name": "A"}], "relations": []} then {"entities": [{"name": "B"}]}'
        result = GraphRAG._parse_extraction(content)
        names = [e["name"] for e in result.get("entities", [])]
        assert "A" in names

    def test_no_json_returns_empty(self):
        result = GraphRAG._parse_extraction("no json here at all")
        assert result == {"entities": [], "relations": []}


# ── L16: mcp _dispatch_tool must raise typed error, not return {"error": str(e)} ──


class TestMCPDispatchErrors:
    @pytest.mark.asyncio
    async def test_missing_param_raises_mcp_error(self):
        from fusion_rag.api.mcp_server import _dispatch_tool, _MCPError

        with pytest.raises(_MCPError) as exc_info:
            await _dispatch_tool("kb_search", {"query": "x"})  # no kb_id
        # -32602 invalid params
        assert exc_info.value.code == -32602
        assert "kb_id" in exc_info.value.message

    @pytest.mark.asyncio
    async def test_unknown_tool_raises_mcp_error(self):
        from fusion_rag.api.mcp_server import _dispatch_tool, _MCPError

        with pytest.raises(_MCPError) as exc_info:
            await _dispatch_tool("no_such_tool", {})
        assert exc_info.value.code == -32601


# ── L18: template param validation ──


class TestTemplateValidation:
    def test_alpha_out_of_range_rejected(self, tmp_path):
        mgr = SearchTemplateManager(str(tmp_path / "tpl.db"))
        bad = SearchTemplate(
            name="bad", description="", alpha=5.0, rerank=False, top_k=10, threshold=0.5, rewrite_mode=""
        )
        with pytest.raises(ValueError, match="alpha"):
            mgr.create_template("kb1", bad)

    def test_threshold_negative_rejected(self, tmp_path):
        mgr = SearchTemplateManager(str(tmp_path / "tpl.db"))
        bad = SearchTemplate(
            name="bad", description="", alpha=0.5, rerank=False, top_k=10, threshold=-0.1, rewrite_mode=""
        )
        with pytest.raises(ValueError, match="threshold"):
            mgr.create_template("kb1", bad)

    def test_top_k_too_large_rejected(self, tmp_path):
        mgr = SearchTemplateManager(str(tmp_path / "tpl.db"))
        bad = SearchTemplate(
            name="bad", description="", alpha=0.5, rerank=False, top_k=9999, threshold=0.5, rewrite_mode=""
        )
        with pytest.raises(ValueError, match="top_k"):
            mgr.create_template("kb1", bad)

    def test_valid_template_accepted(self, tmp_path):
        mgr = SearchTemplateManager(str(tmp_path / "tpl.db"))
        good = SearchTemplate(
            name="ok", description="", alpha=0.6, rerank=True, top_k=20, threshold=0.4, rewrite_mode="hyde"
        )
        result = mgr.create_template("kb1", good)
        assert result is not None


# ── L18: ast_chunker syntax-error fallback uses RecursiveChunker, not whole-file ──


class TestASTFallback:
    def test_syntax_error_splits_not_whole_file(self):
        chunker = ASTChunker()
        big = "def broken(\n" + "x = 1\n" * 200  # syntax error at line 1
        chunks = chunker.chunk(big, "broken.py")
        # Must be split into multiple embed-sized chunks, not one whole-file chunk.
        assert len(chunks) > 1
        for c in chunks:
            assert c.line_start >= 1
            assert c.line_end >= c.line_start

    def test_empty_source_returns_empty(self):
        chunker = ASTChunker()
        assert chunker.chunk("", "empty.py") == []


# ── L18: bench top_k cap ──


class TestBenchTopKCap:
    @pytest.mark.asyncio
    async def test_top_k_capped(self, tmp_path):
        from fusion_rag.engine.bench import BenchRunner

        bench = BenchRunner(str(tmp_path / "bench.db"))
        vec_store = MagicMock()
        embed_client = MagicMock()
        embed_client.embed = AsyncMock(return_value=[0.1, 0.2])
        vec_store.keyword_search = MagicMock(return_value=[])
        vec_store.search = MagicMock(return_value=[])

        captured = {}

        class FakeHS:
            def __init__(self, *a, **k):
                pass

            async def search(self, qv, qt, top_k, **k):
                captured["top_k"] = top_k
                return []

        with patch("fusion_rag.engine.reranker.HybridSearch", FakeHS):
            await bench.run_search_bench(
                "kb1",
                vec_store,
                embed_client,
                [{"query": "q", "top_k": 99999}],
            )
        # 99999 must be capped to 100.
        assert captured["top_k"] == 100
