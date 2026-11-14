"""Issue #127 (R5): CorrectiveRAG loop + RetrievalGrader wiring tests.

Fake grader/rewriter/retrieve_fn — no network. Covers: sufficient first
round (no retry), insufficient round (rewrite + re-retrieve + dedupe),
grader unavailable (degrade, no retry), bounded rounds, JSON parse
tolerance.
"""

from __future__ import annotations

import pytest

from fusion_rag.engine.crag import CorrectiveRAG, RetrievalGrader
from fusion_rag.engine.llm_errors import LLMUnavailable


def _chunk(cid: str) -> dict:
    return {"chunk_id": cid, "text": f"text {cid}", "score": 0.5}


class FakeGrader:
    def __init__(self, relevant_per_round: list[set[int]]):
        self.relevant_per_round = relevant_per_round
        self.calls = 0

    async def grade(self, query, chunks):
        idx = min(self.calls, len(self.relevant_per_round) - 1)
        self.calls += 1
        return self.relevant_per_round[idx]


class FakeRewriter:
    def __init__(self, outputs: list[str]):
        self.outputs = outputs
        self.calls = 0

    async def rewrite(self, query, history=None, mode="expand"):
        out = self.outputs[min(self.calls, len(self.outputs) - 1)]
        self.calls += 1
        return out


@pytest.mark.asyncio
async def test_sufficient_first_round_no_retry():
    retrieves = []

    async def retrieve_fn(q):
        retrieves.append(q)
        return [_chunk("a"), _chunk("b")]

    grader = FakeGrader([{0}])
    crag = CorrectiveRAG(grader=grader, rewriter=FakeRewriter(["x"]), max_rounds=2)
    chunks, trace = await crag.run("q", retrieve_fn)
    assert [c["chunk_id"] for c in chunks] == ["a", "b"]
    assert [t["action"] for t in trace] == ["initial", "sufficient"]
    assert retrieves == ["q"]


@pytest.mark.asyncio
async def test_insufficient_rewrites_and_merges_dedupe():
    calls = {"n": 0}

    async def retrieve_fn(q):
        calls["n"] += 1
        if calls["n"] == 1:
            return [_chunk("a"), _chunk("b")]
        return [_chunk("b"), _chunk("c")]

    grader = FakeGrader([set(), {0}])
    rewriter = FakeRewriter(["q rewritten"])
    crag = CorrectiveRAG(grader=grader, rewriter=rewriter, max_rounds=2)
    chunks, trace = await crag.run("q", retrieve_fn)
    # dedupe: b seen in round 0 keeps its position; c appended
    assert [c["chunk_id"] for c in chunks] == ["a", "b", "c"]
    actions = [t["action"] for t in trace]
    assert actions == ["initial", "retried", "sufficient"]
    assert trace[1]["query"] == "q rewritten"
    assert trace[1]["n_new"] == 1


@pytest.mark.asyncio
async def test_grader_unavailable_degrades():
    class DeadGrader:
        async def grade(self, query, chunks):
            raise LLMUnavailable("down")

    async def retrieve_fn(q):
        return [_chunk("a")]

    crag = CorrectiveRAG(grader=DeadGrader(), rewriter=FakeRewriter(["x"]), max_rounds=2)
    chunks, trace = await crag.run("q", retrieve_fn)
    assert [c["chunk_id"] for c in chunks] == ["a"]
    assert trace[-1]["action"] == "grader_unavailable"


@pytest.mark.asyncio
async def test_rounds_bounded():
    async def retrieve_fn(q):
        return [_chunk("a")]

    grader = FakeGrader([set()])  # never sufficient
    rewriter = FakeRewriter(["q1", "q2", "q3"])
    crag = CorrectiveRAG(grader=grader, rewriter=rewriter, max_rounds=2)
    _chunks, trace = await crag.run("q", retrieve_fn)
    retried = [t for t in trace if t["action"] == "retried"]
    assert len(retried) == 2
    assert grader.calls == 2


@pytest.mark.asyncio
async def test_zero_rounds_is_single_shot():
    async def retrieve_fn(q):
        return [_chunk("a")]

    grader = FakeGrader([set()])
    crag = CorrectiveRAG(grader=grader, rewriter=FakeRewriter(["x"]), max_rounds=0)
    chunks, trace = await crag.run("q", retrieve_fn)
    assert [c["chunk_id"] for c in chunks] == ["a"]
    assert [t["action"] for t in trace] == ["initial"]
    assert grader.calls == 0


def test_grader_parse_relevant_tolerant():
    g = RetrievalGrader()
    assert g._parse_relevant('{"relevant": [0, 2]}', 3) == {0, 2}
    assert g._parse_relevant('Sure! ```json\n{"relevant": [1]}\n```', 3) == {1}
    assert g._parse_relevant('{"relevant": [0, 9, "x"]}', 3) == {0}  # out-of-range/non-int dropped
    assert g._parse_relevant('{"relevant": []}', 3) == set()
    with pytest.raises(LLMUnavailable):
        g._parse_relevant("no json here", 3)
