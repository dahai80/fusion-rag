
import pytest

_fusion_store = pytest.importorskip("fusion_store", reason="fusion_store binding not installed (monorepo-only)")
_np = pytest.importorskip("numpy", reason="numpy not installed")

from fusion_rag.store.fusion_store_backend import FusionStoreBackend
from fusion_rag.store.vector_store import VectorStore


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


@pytest.fixture
def backend(tmp_path):
    store_dir = str(tmp_path / "vectors")
    b = FusionStoreBackend(vector_path=store_dir, dimension=4)
    yield b
    b.close()


class TestFusionStoreBackend:
    def test_add_and_count(self, backend):
        backend.add("d1_0", [1.0, 0.0, 0.0, 0.0], text="first chunk")
        assert backend.count() == 1
        backend.add_batch([_record("d1_1", [0.0, 1.0, 0.0, 0.0]), _record("d1_2", [0.0, 0.0, 1.0, 0.0])])
        assert backend.count() == 3

    def test_search_recall_and_score(self, backend):
        backend.add_batch(
            [
                _record("a", [1.0, 0.0, 0.0, 0.0]),
                _record("b", [0.0, 1.0, 0.0, 0.0]),
                _record("c", [1.0, 1.0, 0.0, 0.0]),
            ]
        )
        # query closest to "a" -> a first
        results = backend.search([1.0, 0.0, 0.0, 0.0], top_k=1)
        assert len(results) == 1
        assert results[0]["id"] == "a"
        # cosine identical -> score ~ 1.0 (1 - dist)
        assert results[0]["score"] == pytest.approx(1.0, abs=1e-5)

    def test_search_result_shape_contract(self, backend):
        backend.add_batch([_record("a", [1.0, 0.0, 0.0, 0.0], text="ctx text", doc_path="p.txt")])
        results = backend.search([1.0, 0.0, 0.0, 0.0], top_k=1)
        r = results[0]
        # id must be the str chunk_id, not the int internal id
        assert isinstance(r["id"], str)
        assert r["id"] == "a"
        assert r["text"] == "ctx text"
        assert r["doc_path"] == "p.txt"
        assert r["doc_name"] == "doc1"
        assert r["doc_type"] == "txt"
        assert isinstance(r["metadata"], dict)
        assert r["metadata"] == {"src": "test"}
        assert "score" in r

    def test_search_threshold_filter(self, backend):
        backend.add_batch([_record("a", [1.0, 0.0, 0.0, 0.0]), _record("b", [0.0, 1.0, 0.0, 0.0])])
        # a is identical (score 1.0), b is orthogonal (score ~0.5); threshold 0.9 drops b
        results = backend.search([1.0, 0.0, 0.0, 0.0], top_k=2, threshold=0.9)
        assert len(results) == 1
        assert results[0]["id"] == "a"

    def test_keyword_search_via_bm25(self, backend):
        backend.add_batch(
            [
                _record("a", [1.0, 0.0, 0.0, 0.0], text="apple banana"),
                _record("b", [0.0, 1.0, 0.0, 0.0], text="cherry date"),
            ]
        )
        results = backend.keyword_search("apple", top_k=2)
        assert len(results) >= 1
        assert results[0]["text"] == "apple banana"

    def test_delete_by_doc(self, backend):
        backend.add_batch(
            [
                _record("a", [1.0, 0.0, 0.0, 0.0], doc_path="doc1.txt"),
                _record("b", [0.0, 1.0, 0.0, 0.0], doc_path="doc2.txt"),
            ]
        )
        assert backend.count() == 2
        deleted = backend.delete_by_doc("doc1.txt")
        assert deleted == 1
        assert backend.count() == 1
        # remaining is doc2's chunk
        results = backend.search([0.0, 1.0, 0.0, 0.0], top_k=1)
        assert results[0]["id"] == "b"

    def test_clear(self, backend):
        backend.add_batch([_record("a", [1.0, 0.0, 0.0, 0.0]), _record("b", [0.0, 1.0, 0.0, 0.0])])
        assert backend.count() == 2
        backend.clear()
        assert backend.count() == 0

    def test_reopen_recovers_dim_and_counter(self, tmp_path):
        store_dir = str(tmp_path / "vectors")
        b = FusionStoreBackend(vector_path=store_dir, dimension=4)
        b.add_batch([_record("a", [1.0, 0.0, 0.0, 0.0]), _record("b", [0.0, 1.0, 0.0, 0.0])])
        b.close()
        # reopen
        b2 = FusionStoreBackend(vector_path=store_dir, dimension=4)
        assert b2.dimension == 4
        assert b2.count() == 2
        # new inserts get fresh int_ids (counter persisted)
        b2.add("c", [0.0, 0.0, 1.0, 0.0], text="third chunk")
        assert b2.count() == 3
        results = b2.search([0.0, 0.0, 1.0, 0.0], top_k=1)
        assert results[0]["id"] == "c"
        b2.close()

    def test_vector_store_wrapper_routes_to_fusion_store(self, tmp_path):
        store_dir = str(tmp_path / "vectors")
        vs = VectorStore(store_dir, dimension=4, backend_type="fusion-store")
        assert vs.backend_type == "fusion-store"
        vs.add_batch([_record("a", [1.0, 0.0, 0.0, 0.0])])
        assert vs.count() == 1
        # wrapper .bm25 property generalized (hasattr path)
        assert vs.bm25 is not None
        results = vs.search([1.0, 0.0, 0.0, 0.0], top_k=1)
        assert results[0]["id"] == "a"
        vs.close()
