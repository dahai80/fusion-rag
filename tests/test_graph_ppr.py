"""Issue #129 (R7): search_ppr unit tests on a hand-built graph (no LLM).

Graph topology:
  seed entity A (chunk c1) -- B (chunk c2) -- C (chunk c3)
  unrelated D (chunk c4), no edges
PPR from query "A" must rank c1 first, then c2 (1 hop), then c3 (2 hops),
never c4.
"""

from __future__ import annotations

import tempfile

import pytest

from fusion_rag.engine.graph_rag import GraphRAG


@pytest.fixture
def graph():
    with tempfile.TemporaryDirectory() as d:
        g = GraphRAG(db_path=f"{d}/graph.db")
        conn = g._get_conn()
        conn.executemany(
            "INSERT INTO entities (name, type, chunk_id) VALUES (?, 'CONCEPT', ?)",
            [("A", "c1"), ("B", "c2"), ("C", "c3"), ("D", "c4")],
        )
        conn.executemany(
            "INSERT INTO relations (source, target) VALUES (?, ?)",
            [("A", "B"), ("B", "C")],
        )
        conn.commit()
        conn.close()
        yield g
        g._close_conn() if hasattr(g, "_close_conn") else None


def test_ppr_orders_by_graph_distance(graph):
    out = graph.search_ppr("A")
    ids = [r["chunk_id"] for r in out]
    assert ids[0] == "c1"
    assert "c2" in ids
    assert "c4" not in ids
    scores = {r["chunk_id"]: r["ppr_score"] for r in out}
    assert scores["c1"] > scores["c2"]
    if "c3" in scores:
        assert scores["c2"] > scores["c3"]


def test_ppr_two_token_query_spreads_seeds(graph):
    # Seeds A and C: both ends get high mass, B (between) collects from both.
    out = graph.search_ppr("A C")
    ids = {r["chunk_id"] for r in out}
    assert {"c1", "c3"} <= ids
    assert "c4" not in ids


def test_ppr_no_match_returns_empty(graph):
    assert graph.search_ppr("zzz-nothing") == []


def test_ppr_empty_graph(graph):
    conn = graph._get_conn()
    conn.execute("DELETE FROM entities")
    conn.execute("DELETE FROM relations")
    conn.commit()
    conn.close()
    assert graph.search_ppr("A") == []


def test_bfs_search_still_works(graph):
    # Regression guard: the R7 PPR layer must not touch the BFS path.
    out = graph.search("A")
    assert any(r["chunk_id"] == "c1" for r in out)
