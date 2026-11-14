"""Issue #132 (R10): BGE-M3 native sparse channel tests.

Covers: sparse_dot scorer, SparseIndex SQLite roundtrip + teardown, HybridSearch
3-way RRF fusion (and unchanged 2-channel path), EmbeddingClient.embed_batch_sparse
feature-detect (400 downgrade / missing field / total failure), flag-off
byte-identical payload.
"""

from __future__ import annotations

import httpx
import pytest

from fusion_rag.embed.client import EmbeddingClient
from fusion_rag.engine.runtime_config import reset_runtime_config
from fusion_rag.engine.sparse_index import SparseIndex, get_sparse_index, sparse_dot


@pytest.fixture(autouse=True)
def _reset_cfg():
    reset_runtime_config()
    yield
    reset_runtime_config()


class TestSparseDot:
    def test_shared_tokens_sum(self):
        q = {"1": 2.0, "5": 1.0}
        d = {"1": 3.0, "9": 10.0}
        assert sparse_dot(q, d) == pytest.approx(6.0)

    def test_empty_sides_zero(self):
        assert sparse_dot({}, {"1": 1.0}) == 0.0
        assert sparse_dot({"1": 1.0}, {}) == 0.0

    def test_string_keys(self):
        # API returns token ids as JSON string keys.
        assert sparse_dot({"33600": 2.0}, {"33600": 1.5}) == pytest.approx(3.0)


class TestSparseIndex:
    def test_roundtrip_and_get_map(self, tmp_path):
        idx = SparseIndex(str(tmp_path / "sparse.db"))
        idx.add_batch([("c1", "d1", {"1": 0.5, "7": 2.0}), ("c2", "d1", {"3": 1.0}), ("c3", "d1", {})])
        m = idx.get_map(["c1", "c2", "c3", "missing"])
        assert m["c1"] == {"1": 0.5, "7": 2.0}
        assert m["c2"] == {"3": 1.0}
        assert "c3" not in m  # empty weights skipped
        assert idx.count() == 2

    def test_replace_is_upsert(self, tmp_path):
        idx = SparseIndex(str(tmp_path / "sparse.db"))
        idx.add_batch([("c1", "d1", {"1": 1.0})])
        idx.add_batch([("c1", "d1", {"2": 2.0})])
        assert idx.get_map(["c1"])["c1"] == {"2": 2.0}
        assert idx.count() == 1

    def test_delete_by_chunk_ids_and_doc(self, tmp_path):
        idx = SparseIndex(str(tmp_path / "sparse.db"))
        idx.add_batch([("c1", "d1", {"1": 1.0}), ("c2", "d1", {"1": 1.0}), ("c3", "d2", {"1": 1.0})])
        assert idx.delete_by_chunk_ids(["c1", "c3", "nope"]) == 2
        assert idx.delete_by_doc("d1") == 1
        assert idx.count() == 0

    def test_pooled_instance_reused(self, tmp_path):
        p = str(tmp_path / "sparse.db")
        assert get_sparse_index(p) is get_sparse_index(p)

    def test_corrupt_row_skipped(self, tmp_path):
        idx = SparseIndex(str(tmp_path / "sparse.db"))
        idx.add_batch([("c1", "d1", {"1": 1.0})])
        conn = idx._get_conn()
        conn.execute("UPDATE sparse_weights SET weights = 'not-json' WHERE chunk_id = 'c1'")
        conn.commit()
        assert idx.get_map(["c1"]) == {}


class TestHybridSearchSparse:
    def _store(self, vector_results, keyword_results):
        class _Store:
            def search(self, vec, top_k=10, threshold=0.0):
                return list(vector_results)

            def keyword_search(self, q, top_k=10):
                return list(keyword_results)

        return _Store()

    def test_third_channel_fuses(self):
        from fusion_rag.engine.reranker import HybridSearch

        hs = HybridSearch(self._store([{"id": "a", "score": 0.9}], [{"id": "b", "score": 5.0}]))
        out = hs._rrf_fusion(
            [{"id": "a", "score": 0.9}],
            [{"id": "b", "score": 5.0}],
            10,
            0.0,
            [{"id": "c", "score": 3.0}],
        )
        ids = [r["id"] for r in out]
        assert set(ids) == {"a", "b", "c"}
        # a appears in all three lists -> top rank
        assert ids[0] == "a"

    async def test_search_skips_sparse_without_query_weights(self):
        from fusion_rag.engine.reranker import HybridSearch

        hs = HybridSearch(self._store([{"id": "a", "score": 0.9}], []))
        called = []

        def _lookup(ids):
            called.append(ids)
            return {}

        out = await hs.search([0.1], "q", top_k=5, sparse_lookup=_lookup)
        assert called == []
        assert [r["id"] for r in out] == ["a"]

    async def test_search_scores_candidates(self):
        from fusion_rag.engine.reranker import HybridSearch

        vec = [{"id": "a", "score": 0.9}, {"id": "b", "score": 0.8}]
        hs = HybridSearch(self._store(vec, []))

        def _lookup(ids):
            assert set(ids) == {"a", "b"}
            return {"a": {"1": 1.0}, "b": {"1": 0.1}}

        out = await hs.search(
            [0.1], "q", top_k=5, query_sparse={"1": 2.0}, sparse_lookup=_lookup
        )
        # sparse ranking: a >> b; vector ranking also a > b — a first either way
        assert out[0]["id"] == "a"
        assert len(out) == 2


class TestEmbedBatchSparse:
    def _client(self):
        return EmbeddingClient(base_url="http://test", api_key="k", cache_enabled=False)

    @pytest.fixture(autouse=True)
    def _flag_on(self, monkeypatch):
        # Sparse parsing only happens when FUSION_RAG_SPARSE_CHANNEL=1.
        monkeypatch.setenv("FUSION_RAG_SPARSE_CHANNEL", "1")
        reset_runtime_config()
        yield
        reset_runtime_config()

    async def test_parses_sparse(self, monkeypatch):
        c = self._client()

        def _resp(request):
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"index": 0, "embedding": [0.1, 0.2], "sparse": {"1": 0.5}},
                        {"index": 1, "embedding": [0.3, 0.4], "sparse": {"2": 1.5}},
                    ]
                },
                request=request,
            )

        transport = httpx.MockTransport(_resp)
        c._client = httpx.AsyncClient(base_url="http://test", transport=transport)
        vecs, sparse = await c.embed_batch_sparse(["a", "b"])
        assert len(vecs) == 2 and vecs[0] == [0.1, 0.2]
        assert sparse[0] == {"1": 0.5} and sparse[1] == {"2": 1.5}
        await c.close()

    async def test_400_downgrades_dense(self, monkeypatch):
        c = self._client()
        calls = {"n": 0}

        def _resp(request):
            calls["n"] += 1
            body = request.read()
            if b"return_sparse" in body:
                return httpx.Response(400, json={"error": "unknown field"}, request=request)
            return httpx.Response(
                200, json={"data": [{"index": 0, "embedding": [0.1, 0.2]}]}, request=request
            )

        c._client = httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(_resp))
        vecs, sparse = await c.embed_batch_sparse(["a"])
        assert calls["n"] == 2
        assert vecs == [[0.1, 0.2]]
        assert sparse == [None]
        assert c._sparse_unsupported is True
        await c.close()

    async def test_missing_field_marks_unsupported(self, monkeypatch):
        c = self._client()

        def _resp(request):
            return httpx.Response(
                200, json={"data": [{"index": 0, "embedding": [0.1, 0.2]}]}, request=request
            )

        c._client = httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(_resp))
        vecs, sparse = await c.embed_batch_sparse(["a"])
        assert vecs == [[0.1, 0.2]] and sparse == [None]
        assert c._sparse_unsupported is True
        await c.close()

    async def test_total_failure_degrades_dense(self, monkeypatch):
        c = self._client()
        # Retry loop exhausts -> falls back to embed_batch, which we stub.

        class _Fail:
            async def post(self, *a, **k):
                raise httpx.ConnectError("down")

        c._client = _Fail()

        async def _ok_async(texts):
            return [[0.1] * 4 for _ in texts]

        monkeypatch.setattr(c, "embed_batch", _ok_async)
        vecs, sparse = await c.embed_batch_sparse(["a"])
        assert vecs[0] == [0.1] * 4
        assert sparse == [None]

    async def test_flag_off_never_requests_sparse(self, monkeypatch):
        # Flag off (default) = payload must NOT contain return_sparse.
        monkeypatch.setenv("FUSION_RAG_SPARSE_CHANNEL", "0")
        reset_runtime_config()
        c = self._client()
        seen = {}

        def _resp(request):
            import json as _json

            seen["payload"] = _json.loads(request.read())
            return httpx.Response(
                200, json={"data": [{"index": 0, "embedding": [0.1, 0.2]}]}, request=request
            )

        c._client = httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(_resp))
        _vecs, sparse = await c.embed_batch_sparse(["a"])
        assert "return_sparse" not in seen["payload"]
        assert sparse == [None]
        await c.close()

    def test_flag_on_requests_sparse(self, monkeypatch):

        monkeypatch.setenv("FUSION_RAG_SPARSE_CHANNEL", "1")
        from fusion_rag.engine.runtime_config import get_runtime_config

        assert get_runtime_config().sparse_channel is True
