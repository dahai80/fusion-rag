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
    # P3-1: doc_path → int_id list index. delete_by_doc used to scan EVERY live
    # vector's meta (O(n) per delete) to find a doc's chunks. The binding has no
    # KV delete, so stale entries can't be purged — but a stale entry only points
    # at an int_id whose meta is re-checked before deletion (defense against a
    # reused id). Value = comma-joined int_ids; rebuild on demand if absent.
    _DOC_INDEX_PREFIX = b"d:"  # d:<doc_path> → "int_id1,int_id2,..."

    def __init__(self, vector_path: str, dimension: int = 1024, **_):
        # P4-5: accept and ignore **_ for uniform factory routing (see LocalBackend).
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
            # Reopen: the store recovers its own schema dim (dim=None). The
            # sidecar only carries id_counter + dim (for add-time validation).
            sidecar_counter = None
            sidecar_dim = None
            try:
                meta = json.loads(self._meta_path.read_text(encoding="utf-8"))
                sidecar_dim = int(meta.get("dim", self.dimension))
                sidecar_counter = int(meta.get("id_counter", 0))
            except Exception as e:
                # P1-3: a corrupt/truncated sidecar MUST NOT reset the counter to
                # 0 — that makes the next insert reuse int_id=0 and clobber every
                # existing vector (insert_vector is upsert-by-id). Open the store
                # and recover the counter from the live vector ids. Fail visibly
                # (ERROR), never silently reset to 0.
                logger.error("FusionStoreBackend sidecar corrupt, recovering counter from store: %s", e)
            logger.info("FusionStoreBackend reopening store: path=%s dim=%s", self._store_path, self.dimension)
            self._store = fs.Store.open(self._store_path, dim=None)
            if sidecar_dim is not None:
                self.dimension = sidecar_dim
            self._reconcile_counter(sidecar_counter)
        else:
            # Create new: lock schema dim.
            logger.info("FusionStoreBackend creating new store: path=%s dim=%s", self._store_path, self.dimension)
            self._store = fs.Store.open(self._store_path, dim=self.dimension)
            self._save_meta()

    def _reconcile_counter(self, sidecar_counter: int | None) -> None:
        # P1-2: the truth for "next int_id" lives in the store (live vector ids),
        # not the sidecar JSON. The sidecar is written AFTER insert_vector
        # persists, so a crash between the two leaves the sidecar stale
        # (counter < real max). Reusing a stale counter overwrites an existing
        # vector and silently corrupts retrieval (idmap now points a chunk_id at
        # a different embedding). Recover from list_vector_ids and, if the
        # sidecar disagrees, trust the store and rewrite a clean sidecar.
        existing = self._store.list_vector_ids()
        recovered = max(existing, default=-1) + 1
        if sidecar_counter is None or sidecar_counter < recovered:
            if sidecar_counter is not None:
                logger.error(
                    "FusionStoreBackend counter drift: sidecar=%s recovered=%s — using recovered (sidecar stale/short)",
                    sidecar_counter,
                    recovered,
                )
            else:
                logger.warning("FusionStoreBackend counter recovered from store: %s", recovered)
            self._id_counter = recovered
        else:
            self._id_counter = sidecar_counter
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
        # P1-4: pre-flight dimension check. fusion_store's Rust side only
        # debug-asserts vector length (compiled out in release → reads adjacent
        # memory = UB), so a model swap (1024→768) or a zero/short fallback
        # vector silently corrupts the HNSW or raises an opaque PyRuntimeError
        # deep in the binding. Validate here with an actionable message before
        # any FFI call.
        if vec.ndim != 1 or vec.shape[0] != self.dimension:
            raise ValueError(
                f"vector dimension mismatch for chunk {chunk_id}: got shape {vec.shape}, "
                f"expected ({self.dimension},). Embedding model changed or provider returned a short vector."
            )
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
        # P3-1: append int_id to the doc_path index so delete_by_doc reads a
        # short candidate list, not every live vector. Value is comma-joined.
        doc_path = meta["doc_path"]
        if doc_path:
            idx_key = self._DOC_INDEX_PREFIX + doc_path.encode("utf-8")
            existing = self._store.get_kv(idx_key)
            existing_ids = existing.decode("utf-8") if existing else ""
            ids_list = [s for s in existing_ids.split(",") if s] if existing_ids else []
            if str(int_id) not in ids_list:
                ids_list.append(str(int_id))
            self._store.put_kv(idx_key, ",".join(ids_list).encode("utf-8"))

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
            # P0-10: fusion_store.search_knn returns squared-L2 distance, NOT
            # cosine. Runtime proof on unit vectors: identical=0, 2x-same-dir=1,
            # orthogonal=2, opposite=4. (Cosine would give 0 for 2x-same-dir
            # since it is scale-invariant — it does not, so this is not cosine.)
            # The prior `1 - dist` mapped 2x-same-dir → 0.0 (same score as
            # orthogonal) and identical → 1.0, opposite → -3.0: a scaled copy of
            # the query scored the same as a perpendicular vector — silent recall
            # break. Stopgap: 1/(1+dist) is monotonic in distance, maps to (0,1]
            # (identical=1.0, 2x=0.5, orthogonal≈0.33, opposite=0.2). NOTE range
            # differs from LocalBackend cosine [-1,1]; hybrid alpha-fusion must
            # account for that. Real fix = upstream Store.open(metric="cosine")
            # param (Rust binding lacks it) — tracked as upstream issue.
            score = 1.0 / (1.0 + float(dist))
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
        # P3-1: use the doc_path→int_ids KV index to avoid an O(n) scan of every
        # live vector. The binding has no KV delete, so the index entry can go
        # stale (an int_id already deleted, or reused after a crash). Guard by
        # re-checking each candidate's meta doc_path before deleting — a stale
        # entry pointing at a reused id would have a different doc_path and is
        # skipped. If the index is missing (old data pre-index), fall back to
        # the full scan so deletes still work after an upgrade.
        candidate_ids: list[int] = []
        idx_key = self._DOC_INDEX_PREFIX + doc_path.encode("utf-8")
        raw_idx = self._store.get_kv(idx_key)
        if raw_idx is not None:
            for s in raw_idx.decode("utf-8", errors="ignore").split(","):
                if s.strip():
                    try:
                        candidate_ids.append(int(s))
                    except ValueError:
                        logger.warning("FusionStoreBackend delete_by_doc: bad id %r in doc index", s)
            logger.debug("FusionStoreBackend delete_by_doc: index hit, %d candidates", len(candidate_ids))
        else:
            logger.debug("FusionStoreBackend delete_by_doc: no doc index, falling back to full scan")
            candidate_ids = list(self._store.list_vector_ids())
        deleted = 0
        for int_id in candidate_ids:
            raw = self._store.get_kv(self._META_PREFIX + str(int_id).encode())
            if raw is None:
                # int_id already deleted (stale index) — skip.
                continue
            try:
                meta = json.loads(raw.decode("utf-8"))
            except Exception as e:
                logger.warning("FusionStoreBackend delete_by_doc: bad meta for int_id=%s: %s", int_id, e)
                continue
            if meta.get("doc_path") != doc_path:
                # Stale index entry or reused id — skip, do not delete.
                continue
            chunk_id = meta.get("chunk_id", "")
            self._store.delete_vector(int_id)
            # KV has no delete in the Python binding; stale meta/idmap/index
            # entries remain but are unreachable (int_id gone). Acceptable:
            # delete is rare, search only reads meta for live ids from KNN.
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
        # P2-3: checkpoint is the only crash-safe sync point (kv.rs recover_inner
        # replays WAL on reopen). Swallowing a checkpoint failure → null the store
        # (drop the Engine handle with no clean flush) → save a sidecar claiming
        # counter=N → reopen finds WAL corrupt or max durable id < N → next insert
        # reuses an id → silent collision/corruption. The worst kind of silent
        # failure: ops see a warn, close "succeeds", data may be lost.
        # Fail visibly instead: on checkpoint failure do NOT null the store nor
        # save a clean meta — raise so the pool-shutdown caller logs a real
        # inconsistent-close. The store handle stays live (caller may retry close).
        if self._store is not None:
            try:
                self._store.checkpoint()
            except Exception as e:
                logger.error("FusionStoreBackend checkpoint on close FAILED (store NOT closed, meta NOT saved): %s", e)
                raise
            self._store = None
        if self._bm25_index is not None:
            try:
                self._bm25_index._close_conn()
            except Exception as e:
                logger.warning("FusionStoreBackend close: BM25 conn close failed: %s", e)
            self._bm25_index = None
        self._save_meta()
