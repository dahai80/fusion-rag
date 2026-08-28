from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .store_backend import StoreBackend

logger = logging.getLogger(__name__)


def _fusion_store():
    try:
        import fusion_store

        return fusion_store
    except ImportError:
        raise ImportError(
            "fusion-store backend requires the 'fusion_store' PyO3 binding. "
            "Install the in-tree project: pip install -e ../fusion-store (maturin). "
            "Not on PyPI; only available in the Fusion monorepo venv."
        )


def _numpy():
    try:
        import numpy as np

        return np
    except ImportError:
        raise ImportError("fusion-store backend requires numpy: pip install numpy")


class FusionStoreBackend(StoreBackend):
    # KV key prefixes — metadata JSON keyed by int_id; str→int id map keyed by chunk_id.
    _META_PREFIX = b"m:"   # m:<int_id> → metadata JSON
    _IDMAP_PREFIX = b"i:"  # i:<chunk_id_str> → int_id string

    def __init__(self, vector_path: str, dimension: int = 1024):
        self.vector_path = vector_path
        self.dimension = dimension
        self._store = None
        self._bm25_index = None
        self._id_counter = 0
        self._meta_path = Path(vector_path) / "fusion_store.meta.json"
        self._store_path = str(Path(vector_path) / "fusion_store.lmdb")
        self._connect()

    def _connect(self) -> None:
        fs = _fusion_store()
        Path(self.vector_path).mkdir(parents=True, exist_ok=True)
        if self._meta_path.exists():
            # Reopen: dim recovered from sidecar, fusion_store.Store.open(path, dim=None)
            try:
                meta = json.loads(self._meta_path.read_text(encoding="utf-8"))
                self.dimension = int(meta.get("dim", self.dimension))
                self._id_counter = int(meta.get("id_counter", 0))
            except Exception as e:
                logger.warning("FusionStoreBackend sidecar parse failed, resetting: %s", e)
                self._id_counter = 0
            logger.info("FusionStoreBackend reopening store: path=%s dim=%s", self._store_path, self.dimension)
            self._store = fs.Store.open(self._store_path, dim=None)
        else:
            # Create new: lock schema dim.
            logger.info("FusionStoreBackend creating new store: path=%s dim=%s", self._store_path, self.dimension)
            self._store = fs.Store.open(self._store_path, dim=self.dimension)
            self._save_meta()

    def _save_meta(self) -> None:
        self._meta_path.write_text(
            json.dumps({"dim": self.dimension, "id_counter": self._id_counter}),
            encoding="utf-8",
        )

    @property
    def bm25(self):
        if self._bm25_index is None:
            from ..engine.bm25_index import BM25Index

            bm25_path = str(Path(self.vector_path).parent / "bm25_index.db")
            self._bm25_index = BM25Index(bm25_path)
        return self._bm25_index

    def _insert_one(self, record: dict[str, Any]) -> None:
        chunk_id = record["id"]
        int_id = self._id_counter
        self._id_counter += 1
        np = _numpy()
        vec = np.asarray(record["vector"], dtype=np.float32)
        self._store.insert_vector(int_id, vec)
        meta = {
            "chunk_id": chunk_id,
            "text": record.get("text", ""),
            "doc_path": record.get("doc_path", ""),
            "doc_name": record.get("doc_name", ""),
            "doc_type": record.get("doc_type", ""),
            "chunk_index": record.get("chunk_index", 0),
            "metadata": record.get("metadata", {}),
            "context": record.get("context", ""),
        }
        meta_bytes = json.dumps(meta, ensure_ascii=False).encode("utf-8")
        self._store.put_kv(self._META_PREFIX + str(int_id).encode(), meta_bytes)
        self._store.put_kv(self._IDMAP_PREFIX + chunk_id.encode("utf-8"), str(int_id).encode("utf-8"))

    def add(
        self,
        chunk_id: str,
        vector: list[float],
        text: str,
        doc_path: str = "",
        doc_name: str = "",
        doc_type: str = "",
        chunk_index: int = 0,
        metadata: dict | None = None,
        context: str = "",
    ) -> None:
        record = {
            "id": chunk_id,
            "vector": vector,
            "text": text,
            "doc_path": doc_path,
            "doc_name": doc_name,
            "doc_type": doc_type,
            "chunk_index": chunk_index,
            "metadata": metadata or {},
            "context": context,
        }
        self._insert_one(record)
        self._save_meta()
        self.bm25.add_documents([record])
        logger.info("FusionStoreBackend add: chunk_id=%s int_id=%d", chunk_id, self._id_counter - 1)

    def add_batch(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        for r in records:
            self._insert_one(r)
        self._save_meta()
        # BM25 expects the same record shape (id/text/doc_path/...) it gets from LocalBackend.
        self.bm25.add_documents(records)
        logger.info("FusionStoreBackend add_batch: %d records, counter now %d", len(records), self._id_counter)

    def search(self, query_vector: list[float], top_k: int = 10, threshold: float = 0.0) -> list[dict[str, Any]]:
        np = _numpy()
        query = np.asarray(query_vector, dtype=np.float32)
        try:
            ids, dists = self._store.search_knn(query, top_k)
        except Exception as e:
            logger.error("FusionStoreBackend search failed (propagating, no silent []): %s", e)
            raise
        results = []
        for int_id, dist in zip(ids, dists):
            score = 1.0 - float(dist)
            if score < threshold:
                continue
            raw = self._store.get_kv(self._META_PREFIX + str(int_id).encode())
            if raw is None:
                logger.warning("FusionStoreBackend search: missing meta for int_id=%s, skipping", int_id)
                continue
            try:
                meta = json.loads(raw.decode("utf-8"))
            except Exception as e:
                logger.warning("FusionStoreBackend search: bad meta JSON for int_id=%s: %s", int_id, e)
                continue
            results.append(
                {
                    "id": meta.get("chunk_id", str(int_id)),
                    "text": meta.get("text", ""),
                    "doc_path": meta.get("doc_path", ""),
                    "doc_name": meta.get("doc_name", ""),
                    "doc_type": meta.get("doc_type", ""),
                    "chunk_index": meta.get("chunk_index", 0),
                    "metadata": meta.get("metadata", {}),
                    "context": meta.get("context", ""),
                    "score": score,
                }
            )
        return results

    def keyword_search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        return self.bm25.search(query, top_k)

    def delete_by_doc(self, doc_path: str) -> int:
        deleted = 0
        for int_id in list(self._store.list_vector_ids()):
            raw = self._store.get_kv(self._META_PREFIX + str(int_id).encode())
            if raw is None:
                continue
            try:
                meta = json.loads(raw.decode("utf-8"))
            except Exception as e:
                logger.warning("FusionStoreBackend delete_by_doc: bad meta for int_id=%s: %s", int_id, e)
                continue
            if meta.get("doc_path") != doc_path:
                continue
            chunk_id = meta.get("chunk_id", "")
            self._store.delete_vector(int_id)
            # KV has no delete in the Python binding; stale meta/idmap entries
            # remain but are unreachable (int_id gone). Acceptable: delete is
            # rare, and search only reads meta for live ids returned by KNN.
            deleted += 1
            logger.debug("FusionStoreBackend delete_by_doc: removed int_id=%s chunk_id=%s", int_id, chunk_id)
        self.bm25.remove_document(doc_path)
        logger.info("FusionStoreBackend delete_by_doc: doc_path=%s deleted=%d", doc_path, deleted)
        return deleted

    def count(self) -> int:
        return self._store.vector_count()

    def clear(self) -> None:
        # Soft-delete every live vector via the binding (no mmap unlink needed;
        # LMDB files can't be removed while the env is open on macOS).
        ids = list(self._store.list_vector_ids())
        for int_id in ids:
            self._store.delete_vector(int_id)
        logger.info("FusionStoreBackend clear: soft-deleted %d vectors", len(ids))

    def close(self) -> None:
        if self._store is not None:
            try:
                self._store.checkpoint()
            except Exception as e:
                logger.warning("FusionStoreBackend checkpoint on close failed: %s", e)
            self._store = None
        self._bm25_index = None
        self._save_meta()
