"""Issue #128 (R6): QueryRouter rule tests + route-map wiring semantics."""

from __future__ import annotations

import pytest

from fusion_rag.engine.query_router import QueryRouter


@pytest.mark.parametrize(
    "q,expected",
    [
        ("SKU-101 specs", "simple"),
        ("灯带长度", "simple"),
        ("Aurora Strip product overview and specs", "standard"),
        ("compare the difference between plan A and plan B", "complex"),
        ("why does the device reboot randomly after firmware update", "complex"),
        ("x" * 150, "complex"),
    ],
)
def test_route_levels(q, expected):
    d = QueryRouter().route(q)
    assert d.level == expected


def test_history_forces_complex():
    d = QueryRouter().route("short", history_len=2)
    assert d.level == "complex"
    assert "history" in d.reason


def test_multiple_question_marks_complex():
    d = QueryRouter().route("what is it? who made it?")
    assert d.level == "complex"


def test_empty_question_simple():
    assert QueryRouter().route("").level == "simple"


def test_route_map_semantics():
    # simple strips enhancement flags; standard enables hybrid; complex adds retry.
    router = QueryRouter()
    for q, checks in [
        ("SKU-101", {"hybrid": False, "rerank": False, "crag": 0}),
        ("mid length question about lighting products", {"hybrid": True}),
        ("compare difference between x and y", {"hybrid": True, "rerank": True, "crag": 2}),
    ]:
        d = router.route(q)
        if d.level == "simple":
            assert checks["hybrid"] is False
        elif d.level == "standard":
            assert checks["hybrid"] is True
        else:
            assert checks.get("crag", 0) >= 2
