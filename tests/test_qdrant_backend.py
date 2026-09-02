"""Issue #66 — Qdrant vector backend + per-tenant collection isolation tests.

Uses QdrantClient(":memory:") — no Qdrant server needed. Verifies the ABC
contract (add/search/keyword/delete/count/clear) AND the #66 acceptance:
tenant A's upserts land in `tenant_id_A`, tenant B's search of `tenant_id_B`
returns zero hits (physical collection-per-tenant isolation).
"""
from __future__ import annotations

import contextlib

import pytest

_qdrant = pytest.importorskip("qdrant_client", reason="qdrant-client not installed")

from fusion_rag.api.tenant import _request_tenant, reset_request_tenant
from fusion_rag.store.qdrant_backend import QdrantBackend


def _record(chunk_id, vector, text="hello world", doc_path="doc1.txt", doc_name="doc1", doc_type="txt"):
    return {
        "id": chunk_id,
        "vector": vector,
        "text": text,
        "doc_path": doc_path,
        "doc_name": doc_name,
        "doc_type": doc_type,
        "chunk_index": 0,
        "metadata": {"src": "test"},
        "context": "",
    }


@contextlib.contextmanager
def _tenant(t):
    token = _request_tenant.set(t)
    try:
        yield
    finally:
        _request_tenant.reset(token)


@pytest.fixture
def backend(tmp_path, monkeypatch):
    # :memory: keeps each test isolated (no shared on-disk collection state).
    monkeypatch.setenv("QDRANT_URL", ":memory:")
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)
    reset_request_tenant()
    store_dir = str(tmp_path / "vectors")
    b = QdrantBackend(vector_path=store_dir, dimension=4)
    yield b
    b.close()
    reset_request_tenant()


class TestQdrantBackendContract:
    def test_add_and_count(self, backend):
        backend.add("d1_0", [1.0, 0.0, 0.0, 0.0], text="first chunk")
        assert backend.count() == 1
        backend.add_batch([_record("d1_1", [0.0, 1.0, 0.0, 0.0]), _record("d1_2", [0.0, 0.0, 1.0, 0.0])])
        assert backend.count() == 3

    def test_search_recall_and_score(self, backend):
        backend.add_batch(
            [
                _record("d1_0", [1.0, 0.0, 0.0, 0.0], text="cats"),
                _record("d1_1", [0.0, 1.0, 0.0, 0.0], text="dogs"),
                _record("d1_2", [0.9, 0.1, 0.0, 0.0], text="kittens"),
            ]
        )
        res = backend.search([1.0, 0.0, 0.0, 0.0], top_k=2)
        assert len(res) == 2
        ids = [r["id"] for r in res]
        assert ids[0] == "d1_0", f"top match must be d1_0: {ids}"
        assert res[0]["score"] >= 0.99, f"identical vector score ~1.0: {res[0]['score']}"
        assert "d1_2" in ids, f"kittens (near-parallel) should rank: {ids}"

    def test_search_threshold_filters(self, backend):
        backend.add_batch([_record("d1_0", [1.0, 0.0, 0.0, 0.0]), _record("d1_1", [0.0, 1.0, 0.0, 0.0])])
        res = backend.search([1.0, 0.0, 0.0, 0.0], top_k=5, threshold=0.99)
        assert len(res) == 1 and res[0]["id"] == "d1_0"

    def test_result_shape(self, backend):
        backend.add("d1_0", [1.0, 0.0, 0.0, 0.0], text="hi", doc_path="a.md",
                    doc_name="a", doc_type="md", metadata={"k": "v"}, context="ctx")
        res = backend.search([1.0, 0.0, 0.0, 0.0], top_k=1)
        r = res[0]
        assert isinstance(r["id"], str)
        assert r["text"] == "hi"
        assert r["doc_path"] == "a.md"
        assert r["metadata"] == {"k": "v"}
        assert r["context"] == "ctx"
        assert isinstance(r["score"], float)

    def test_keyword_search_via_bm25(self, backend):
        backend.add_batch(
            [
                _record("d1_0", [1.0, 0.0, 0.0, 0.0], text="machine learning basics", doc_path="ml.md"),
                _record("d1_1", [0.0, 1.0, 0.0, 0.0], text="vector database design", doc_path="vdb.md"),
            ]
        )
        res = backend.keyword_search("vector database", top_k=5)
        assert len(res) >= 1
        # BM25 returns id/score/text; the vector DB doc was d1_1 (vdb.md).
        assert res[0]["id"] == "d1_1"
        assert "vector database" in res[0]["text"]

    def test_delete_by_doc(self, backend):
        backend.add_batch(
            [
                _record("d1_0", [1.0, 0.0, 0.0, 0.0], doc_path="keep.md"),
                _record("d1_1", [0.0, 1.0, 0.0, 0.0], doc_path="drop.md"),
                _record("d2_0", [0.0, 0.0, 1.0, 0.0], doc_path="drop.md"),
            ]
        )
        assert backend.count() == 3
        n = backend.delete_by_doc("drop.md")
        assert n == 2, f"delete_by_doc should return 2: {n}"
        assert backend.count() == 1

    def test_clear(self, backend):
        backend.add_batch([_record("d1_0", [1.0, 0.0, 0.0, 0.0]), _record("d1_1", [0.0, 1.0, 0.0, 0.0])])
        assert backend.count() == 2
        backend.clear()
        assert backend.count() == 0

    def test_dim_mismatch_raises(self, backend):
        with pytest.raises(ValueError, match="dimension mismatch"):
            backend.add("d1_0", [1.0, 0.0, 0.0], text="short vector")


class TestQdrantTenantIsolation:
    def test_cross_tenant_search_returns_zero(self, backend):
        # acceptance #66: tenant A upserts, tenant B searches → no hits.
        with _tenant("tenant-a"):
            backend.add_batch(
                [
                    _record("a_0", [1.0, 0.0, 0.0, 0.0], text="secret A", doc_path="a.md"),
                    _record("a_1", [0.9, 0.1, 0.0, 0.0], text="secret A2", doc_path="a2.md"),
                ]
            )
            assert backend.count() == 2
        with _tenant("tenant-b"):
            assert backend.count() == 0
            res = backend.search([1.0, 0.0, 0.0, 0.0], top_k=10)
            assert res == [], f"tenant B must not see tenant A vectors: {res}"

    def test_same_tenant_sees_own_vectors(self, backend):
        with _tenant("tenant-a"):
            backend.add_batch([_record("a_0", [1.0, 0.0, 0.0, 0.0], text="A secret", doc_path="a.md")])
        with _tenant("tenant-a"):
            res = backend.search([1.0, 0.0, 0.0, 0.0], top_k=5)
            assert len(res) == 1 and res[0]["id"] == "a_0"

    def test_default_collection_when_no_tenant(self, backend):
        # no tenant set (single-tenant dev) → shared fusion_rag_kb_<kb_id> collection.
        backend.add_batch([_record("d1_0", [1.0, 0.0, 0.0, 0.0])])
        assert backend.count() == 1
        res = backend.search([1.0, 0.0, 0.0, 0.0], top_k=5)
        assert len(res) == 1 and res[0]["id"] == "d1_0"

    def test_delete_is_tenant_scoped(self, backend):
        with _tenant("tenant-a"):
            backend.add_batch([_record("a_0", [1.0, 0.0, 0.0, 0.0], doc_path="shared.md")])
        with _tenant("tenant-b"):
            backend.add_batch([_record("b_0", [1.0, 0.0, 0.0, 0.0], doc_path="shared.md")])
        # tenant-b deletes shared.md → only its own collection affected.
        with _tenant("tenant-b"):
            n = backend.delete_by_doc("shared.md")
            assert n == 1
        with _tenant("tenant-a"):
            assert backend.count() == 1, "tenant-a's shared.md must survive tenant-b's delete"
