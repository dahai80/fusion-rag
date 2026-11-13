# Issue #117: eval set + offline harness tests.

from __future__ import annotations

from fusion_rag.engine.eval_set import (
    EVAL_METRICS,
    EvalStore,
    run_offline_eval,
    seed_default_cases,
)


class TestOfflineEval:
    async def test_run_offline_eval_deterministic(self, tmp_path):
        m1, cases1 = await run_offline_eval(str(tmp_path / "w1"))
        m2, _cases2 = await run_offline_eval(str(tmp_path / "w2"))
        assert set(EVAL_METRICS).issubset(m1)
        assert len(cases1) == 55
        assert m1 == m2
        assert m1["mrr@10"] >= 0.9
        assert m1["recall@10"] >= 0.8
        for c in cases1:
            assert c["gold_doc_id"]
            assert set(EVAL_METRICS).issubset(c)


class TestEvalStore:
    def test_seed_idempotent(self, tmp_path):
        import asyncio

        store = EvalStore(str(tmp_path / "eval.db"))
        n1 = asyncio.run(seed_default_cases(store))
        n2 = asyncio.run(seed_default_cases(store))
        assert n1 > 0
        assert n2 == 0
        assert store.case_count() == n1
        cases = store.list_cases()
        assert all(c["gold_chunk_ids"] for c in cases)

    def test_record_and_latest_run(self, tmp_path):
        store = EvalStore(str(tmp_path / "eval.db"))
        assert store.latest_run() is None
        run_id = store.record_run({"recall@10": 0.92, "mrr@10": 1.0}, case_count=55)
        latest = store.latest_run()
        assert latest["run_id"] == run_id
        assert latest["metrics"] == {"recall@10": 0.92, "mrr@10": 1.0}
        assert latest["case_count"] == 55
