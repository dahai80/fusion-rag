"""Tests for issue #94: deleting one doc must not remove the physically-shared
chunks of an identical-content doc (two docs sharing one doc_path)."""

from __future__ import annotations

import tempfile
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from fusion_rag.api.server import create_app
from fusion_rag.embed.client import EmbeddingClient


@pytest.fixture
def client():
    app = create_app(kb_storage_dir=tempfile.mkdtemp())
    with (
        TestClient(app) as tc,
        patch.object(EmbeddingClient, "health", new_callable=AsyncMock, return_value=True),
        patch.object(
            EmbeddingClient,
            "embed_batch",
            new_callable=AsyncMock,
            side_effect=lambda texts: [[0.01] * 1024 for _ in texts],
        ),
        patch.object(EmbeddingClient, "embed", new_callable=AsyncMock, return_value=[0.01] * 1024),
    ):
        yield tc


class TestBM25RemoveChunks:
    def test_remove_chunks_only_listed_ids(self):
        from fusion_rag.engine.bm25_index import BM25Index

        with tempfile.TemporaryDirectory() as td:
            idx = BM25Index(f"{td}/bm25.db")
            idx.add_documents(
                [
                    {"id": "docA_0", "text": "waterproof membrane 外墙防水 spec", "doc_path": "inline://same.md"},
                    {"id": "docB_0", "text": "waterproof membrane 外墙防水 spec", "doc_path": "inline://same.md"},
                ]
            )
            assert idx.remove_chunks(["docA_0"]) == 1
            hits = idx.search("waterproof", top_k=10)
            assert [h["id"] for h in hits] == ["docB_0"]
            assert idx.count() == 1
            idx._close_conn()

    def test_remove_chunks_unknown_id_noop(self):
        from fusion_rag.engine.bm25_index import BM25Index

        with tempfile.TemporaryDirectory() as td:
            idx = BM25Index(f"{td}/bm25.db")
            idx.add_documents([{"id": "docA_0", "text": "alpha beta", "doc_path": "p"}])
            assert idx.remove_chunks(["nope_0", ""]) == 0
            assert idx.count() == 1
            idx._close_conn()

    def test_remove_document_still_removes_all_shared_path(self):
        from fusion_rag.engine.bm25_index import BM25Index

        with tempfile.TemporaryDirectory() as td:
            idx = BM25Index(f"{td}/bm25.db")
            idx.add_documents(
                [
                    {"id": "docA_0", "text": "alpha", "doc_path": "inline://same.md"},
                    {"id": "docB_0", "text": "beta", "doc_path": "inline://same.md"},
                ]
            )
            assert idx.remove_document("inline://same.md") == 2
            assert idx.count() == 0
            idx._close_conn()


class TestLocalBackendDeleteByChunkIds:
    def test_delete_by_chunk_ids_removes_only_listed(self):
        from fusion_rag.store.local_backend import LocalBackend

        with tempfile.TemporaryDirectory() as td:
            be = LocalBackend(vector_path=f"{td}/vectors", dimension=4)
            base = {
                "vector": [0.1, 0.2, 0.3, 0.4],
                "text": "shared spec text 外墙防水",
                "doc_path": "inline://same.md",
                "doc_name": "same.md",
                "doc_type": "markdown",
                "chunk_index": 0,
                "metadata": {},
                "context": "",
            }
            be.add_batch([dict(base, id="docA_0"), dict(base, id="docB_0")])
            assert be.count() == 2
            assert be.delete_by_chunk_ids(["docA_0"]) == 1
            assert be.count() == 1
            hits = be.search([0.1, 0.2, 0.3, 0.4], top_k=10)
            assert [h["id"] for h in hits] == ["docB_0"]
            # BM25 parity: the surviving chunk is still keyword-retrievable.
            kw = be.bm25.search("防水", top_k=10)
            assert [h["id"] for h in kw] == ["docB_0"]
            assert be.delete_by_chunk_ids([]) == 0
            be.close()


class TestSharedPathDeleteViaAPI:
    def test_delete_one_doc_keeps_identical_content_doc(self, client):
        kb_id = client.post("/kb/bases", json={"name": "dedup-kb"}).json()["id"]
        payload = {
            "content": "外墙防水施工规范与验收标准 waterproof membrane spec",
            "content_type": "text",
            "doc_name": "same-spec.md",
            "contextualize": False,
        }
        doc_a = client.post(f"/kb/bases/{kb_id}/documents/ingest", json=payload).json()["doc_id"]
        doc_b = client.post(f"/kb/bases/{kb_id}/documents/ingest", json=payload).json()["doc_id"]
        assert doc_a != doc_b

        rows = client.post(
            f"/kb/bases/{kb_id}/search", json={"query": "外墙防水", "top_k": 10}
        ).json()
        assert {r["doc_id"] for r in rows} == {doc_a, doc_b}

        resp = client.delete(f"/kb/bases/{kb_id}/documents/{doc_a}")
        assert resp.status_code == 200

        # Issue #94 regression: doc B was never deleted and must stay retrievable
        # on BOTH the vector and the keyword path.
        rows = client.post(
            f"/kb/bases/{kb_id}/search", json={"query": "外墙防水", "top_k": 10}
        ).json()
        assert rows, "doc B lost retrievability after deleting doc A"
        assert {r["doc_id"] for r in rows} == {doc_b}

        rows = client.post(
            f"/kb/bases/{kb_id}/search", json={"query": "外墙防水", "top_k": 10, "hybrid": True}
        ).json()
        assert {r["doc_id"] for r in rows} == {doc_b}

        # Doc B still listed with its chunks intact.
        docs = client.get(f"/kb/bases/{kb_id}/documents").json()
        remaining = {d["id"]: d for d in docs if isinstance(d, dict)}
        assert doc_b in remaining
        assert remaining[doc_b].get("chunk_count", 1) >= 1
