# R12 (issue #134): NDCG/MAP metric helpers + per-source trust weights.

from __future__ import annotations

import pytest
from fastapi import HTTPException

from fusion_rag.api.routes_search import _apply_source_weights, _read_source_weights
from fusion_rag.engine.recall_bench import _map_at_k, _ndcg_at_k

TRUTH = {"a", "b", "c"}


class TestNdcg:
    def test_perfect_ranking(self):
        assert _ndcg_at_k(["a", "b", "c"], TRUTH, 3) == 1.0

    def test_all_relevant_but_late(self):
        # all 3 relevant at ranks 4-6, k=3 -> nothing credited
        assert _ndcg_at_k(["x", "y", "z"], TRUTH, 3) == 0.0

    def test_partial_discounted(self):
        # one hit at rank 2: dcg = 1/log2(3); idcg = 1 + 1/log2(3) + 1/log2(4)... k=3, 3 truth
        got = _ndcg_at_k(["x", "a"], TRUTH, 3)
        import math

        idcg = 1 + 1 / math.log2(3) + 1 / math.log2(4)
        assert got == pytest.approx((1 / math.log2(3)) / idcg, abs=1e-6)

    def test_empty_truth(self):
        assert _ndcg_at_k(["a"], set(), 3) == 0.0

    def test_more_truth_than_k_capped(self):
        # truth bigger than k: ideal only counts top-k positions
        big = {f"d{i}" for i in range(10)}
        assert _ndcg_at_k(["d0", "d1", "d2"], big, 3) == 1.0


class TestMap:
    def test_perfect_ranking(self):
        assert _map_at_k(["a", "b", "c"], TRUTH, 3) == 1.0

    def test_one_hit_rank1(self):
        # precision@1 = 1, capped by min(len(truth), k)=3 -> 1/3
        assert _map_at_k(["a", "x", "y"], TRUTH, 3) == pytest.approx(1 / 3)

    def test_two_hits(self):
        # hits at rank1 (1/1) and rank3 (2/3); sum/3 = (1 + 2/3)/3
        assert _map_at_k(["a", "x", "b"], TRUTH, 3) == pytest.approx((1 + 2 / 3) / 3)

    def test_no_hits(self):
        assert _map_at_k(["x", "y"], TRUTH, 3) == 0.0

    def test_empty_truth(self):
        assert _map_at_k(["a"], set(), 3) == 0.0


class TestSourceWeights:
    def test_read_valid(self):
        assert _read_source_weights({"source_weights": {"pdf": 1.5, "md": "0.8"}}) == {"pdf": 1.5, "md": 0.8}

    def test_read_empty_or_missing(self):
        assert _read_source_weights({}) == {}
        assert _read_source_weights({"source_weights": None}) == {}

    def test_read_non_dict_raises_400(self):
        with pytest.raises(HTTPException) as e:
            _read_source_weights({"source_weights": [1, 2]})
        assert e.value.status_code == 400

    def test_invalid_entries_ignored_not_fatal(self):
        w = _read_source_weights({"source_weights": {"pdf": "abc", "md": 99, "code": -1, "txt": 2.0}})
        assert w == {"txt": 2.0}

    def test_apply_reorders_and_stamps(self):
        rows = [
            {"id": "1", "doc_type": "md", "score": 0.9},
            {"id": "2", "doc_type": "pdf", "score": 0.6},
        ]
        out = _apply_source_weights(rows, {"pdf": 2.0})
        assert out[0]["id"] == "2"
        assert out[0]["source_weight"] == 2.0
        assert out[0]["score"] == pytest.approx(1.2)

    def test_apply_no_weights_is_identity_order(self):
        rows = [{"id": "1", "doc_type": "md", "score": 0.9}]
        assert _apply_source_weights(rows, {}) is rows
