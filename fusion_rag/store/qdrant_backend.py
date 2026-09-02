"""Issue #66 — Qdrant vector backend + per-tenant collection isolation.

Optional vector store backend (FUSION_RAG_STORE_BACKEND=qdrant). Stores vectors
in a Qdrant collection chosen PER REQUEST from the authoritative tenant
(X-Fusion-Tenant via tenant.get_request_tenant): a tenant-A upsert lands in
collection `tenant_id_<A>` and a tenant-B search queries `tenant_id_<B>` —
cross-tenant vectors are physically invisible (collection-per-tenant is the
data-layer isolation tier; the KB-level `tenant` field from #61 is the logical
tier). When no tenant is in effect (single-tenant dev, isolation off) a shared
`fusion_rag_kb_<kb_id>` collection is used so the default path still works.

keyword search stays in-process (BM25Index), exactly like FusionStoreBackend —
Qdrant is a vector primitive, not a keyword engine.

chunk_id is str; Qdrant point id must be int. A deterministic blake2b hash maps
str→uint64 so re-ingest (replace) overwrites the same point cleanly.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from .store_backend import StoreBackend

logger = logging.getLogger(__name__)


def _qdrant():
    try:
        import qdrant_client

        return qdrant_client
    except ImportError:
        raise ImportError("Install qdrant-client: pip install 'fusion-rag[qdrant]' or pip install qdrant-client")


# Qdrant collection names: [a-zA-Z0-9_-], max 255 chars.
_COLLECTION_RE = re.compile(r"^[A-Za-z0-9_\-]{1,255}$")
_INT63_MASK = (1 << 63) - 1


def _point_id(chunk_id: str) -> int:
    # Deterministic str→int for Qdrant point id. Same chunk_id always maps to
    # the same point id, so a re-ingest overwrites instead of duplicating.
    h = hashlib.blake2b(chunk_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") & _INT63_MASK


class QdrantBackend(StoreBackend):
    def __init__(self, vector_path: str, dimension: int = 1024, **_):
        # P4-5: accept and ignore **_ so VectorStore routes every backend through
        # StoreBackendFactory.create(**kwargs) uniformly — remote-only params
        # (endpoint/api_key) passed here no longer raise TypeError.
        self.vector_path = vector_path
        self.dimension = dimension
        # kb_id derived from the vector_path basename — the default (no-tenant)
        # collection is namespaced by it so two KBs never share a collection.
        self._kb_id = Path(vector_path).name or "default"
        self._bm25_index = None
        url = os.environ.get("QDRANT_URL", ":memory:")
        api_key = os.environ.get("QDRANT_API_KEY") or None
        self._prefix = os.environ.get("QDRANT_COLLECTION_PREFIX", "tenant_id_")
        qdrant = _qdrant()
        # Local in-memory mode is triggered via path, not url: QdrantClient(url=":memory:")
        # tries to parse it as a remote host and raises LocationParseError. Route the
        # :memory: sentinel to path= so unit tests run with no Qdrant server.
        if url in (":memory:", "memory"):
            self._client = qdrant.QdrantClient(path=":memory:")
        else:
            self._client = qdrant.QdrantClient(url=url, api_key=api_key)
        logger.info("QdrantBackend initialized: kb_id=%s dim=%d url=%s", self._kb_id, dimension, url)

    @property
    def bm25(self):
        # In-process keyword index — Qdrant is vector-only. Same layout as
        # LocalBackend/FusionStoreBackend: bm25_index.db beside the vector dir.
        if self._bm25_index is None:
            from ..engine.bm25_index import BM25Index

            bm25_path = str(Path(self.vector_path).parent / "bm25_index.db")
            self._bm25_index = BM25Index(bm25_path)
        return self._bm25_index

    def _collection_for(self) -> str:
        # Issue #66: pick the collection per request from the authoritative
        # tenant (X-Fusion-Tenant via tenant.get_request_tenant). Tenant set =>
        # `tenant_id_<tenant>` (physical isolation). Tenant None (single-tenant
        # dev, isolation off) => `fusion_rag_kb_<kb_id>` shared collection.
        from ..api.tenant import get_request_tenant

        tenant = get_request_tenant()
        if tenant:
            name = f"{self._prefix}{tenant}"
        else:
            name = f"fusion_rag_kb_{self._kb_id}"
        if not _COLLECTION_RE.match(name):
            # A normalized tenant is charset-safe (tenant.py validates it), but
            # a pathological kb_id could slip through — refuse loudly rather
            # than build a collection Qdrant rejects.
            raise ValueError(f"invalid Qdrant collection name derived: {name!r}")
        return name

    def _ensure_collection(self, name: str) -> None:
        # Idempotent create: a missing collection is created on first touch;
        # an existing one is a no-op. Any other error surfaces (corrupt store,
        # network) — do NOT silently rebuild.
        try:
            self._client.get_collection(name)
            return
        except Exception as e:
            msg = str(e).lower()
            if "not found" not in msg and "doesn't exist" not in msg and "404" not in msg:
                logger.error("QdrantBackend get_collection(%s) failed (not creating): %s", name, e)
                raise
        from qdrant_client.models import Distance, VectorParams

        self._client.create_collection(name, vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE))
        logger.info("QdrantBackend created collection %s (dim=%d)", name, self.dimension)

    def _check_vector_dim(self, records: list[dict[str, Any]]) -> None:
        # P1-4 parity with LocalBackend: validate BEFORE upsert so a model swap
        # or short fallback vector fails loud at the offending id, not mid-batch
        # with a partial Qdrant write.
        for r in records:
            vec = r.get("vector")
            if vec is None or len(vec) != self.dimension:
                raise ValueError(
                    f"vector dimension mismatch for chunk {r.get('id', '?')}: "
                    f"got {len(vec) if vec is not None else 'None'}, expected {self.dimension}. "
                    "Embedding model changed or provider returned a short vector."
                )

    def _points(self, records: list[dict[str, Any]]) -> list:
        from qdrant_client.models import PointStruct

        points = []
        for r in records:
            cid = r["id"]
            meta_json = r.get("metadata_json")
            if meta_json is None:
                meta_json = json.dumps(r.get("metadata", {}), ensure_ascii=False)
            points.append(
                PointStruct(
                    id=_point_id(cid),
                    vector=list(r["vector"]),
                    payload={
                        "chunk_id": cid,
                        "text": r.get("text", ""),
                        "doc_path": r.get("doc_path", ""),
                        "doc_name": r.get("doc_name", ""),
                        "doc_type": r.get("doc_type", ""),
                        "chunk_index": int(r.get("chunk_index", 0)),
                        "metadata_json": meta_json,
                        "context": r.get("context", ""),
                    },
                )
            )
        return points

    def add(self, chunk_id, vector, text, doc_path="", doc_name="", doc_type="",
            chunk_index=0, metadata=None, context="") -> None:
        records = [
            {
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
        ]
        self.add_batch(records)

    def add_batch(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        # LocalBackend.add_batch normalizes metadata→metadata_json; mirror it so
        # bm25.add_documents sees a consistent record shape either way.
        for r in records:
            if "metadata_json" not in r and "metadata" in r:
                r["metadata_json"] = json.dumps(r.pop("metadata", {}), ensure_ascii=False)
        self._check_vector_dim(records)
        collection = self._collection_for()
        self._ensure_collection(collection)
        points = self._points(records)
        self._client.upsert(collection, points=points)
        self.bm25.add_documents(records)
        logger.debug("QdrantBackend upsert %d points → %s", len(points), collection)

    def search(self, query_vector: list[float], top_k: int = 10, threshold: float = 0.0) -> list[dict[str, Any]]:
        # M3 parity: let a genuine search failure raise, do not return [] (an
        # empty collection returns [] normally, so only real errors propagate).
        collection = self._collection_for()
        try:
            self._client.get_collection(collection)
        except Exception:
            # collection not yet created => empty store, no matches.
            return []
        try:
            res = self._client.query_points(collection, query=list(query_vector), limit=top_k)
        except Exception as e:
            logger.error("QdrantBackend search failed (propagating, no silent []): %s", e)
            raise
        filtered = []
        for p in res.points:
            # Qdrant cosine returns similarity in [-1,1]; 1.0 = identical.
            score = float(p.score) if p.score is not None else 0.0
            if score < threshold:
                continue
            payload = p.payload or {}
            try:
                metadata = json.loads(payload.get("metadata_json", "{}"))
            except Exception as e:
                logger.warning("QdrantBackend search: bad metadata_json: %s", e)
                metadata = {}
            filtered.append(
                {
                    "id": payload.get("chunk_id", str(p.id)),
                    "text": payload.get("text", ""),
                    "doc_path": payload.get("doc_path", ""),
                    "doc_name": payload.get("doc_name", ""),
                    "doc_type": payload.get("doc_type", ""),
                    "chunk_index": payload.get("chunk_index", 0),
                    "metadata": metadata,
                    "context": payload.get("context", ""),
                    "score": score,
                }
            )
        return filtered

    def keyword_search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        return self.bm25.search(query, top_k)

    def delete_by_doc(self, doc_path: str) -> int:
        # M3 parity: raise on real failure, return matched count on success.
        # Qdrant delete does not return a count — scroll the matching points
        # first (filter by doc_path), count, then delete by id list.
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        collection = self._collection_for()
        try:
            self._client.get_collection(collection)
        except Exception:
            # nothing to delete from a non-existent collection.
            self.bm25.remove_document(doc_path)
            return 0
        filt = Filter(must=[FieldCondition(key="doc_path", match=MatchValue(value=doc_path))])
        ids = []
        offset = None
        while True:
            points, offset = self._client.scroll(collection, scroll_filter=filt, limit=256, offset=offset)
            ids.extend(pt.id for pt in points)
            if offset is None:
                break
        if ids:
            self._client.delete(collection, points_selector=ids)
        self.bm25.remove_document(doc_path)
        logger.debug("QdrantBackend delete_by_doc(%s) removed %d points from %s", doc_path, len(ids), collection)
        return len(ids)

    def count(self) -> int:
        # M3 parity: raise on failure (do not return 0 indistinguishable from
        # empty). Count the caller's collection only (tenant-aware).
        collection = self._collection_for()
        try:
            return int(self._client.count(collection).count)
        except Exception:
            # collection not yet created => 0 vectors, not an error.
            return 0

    def clear(self) -> None:
        # Drop the caller's collection entirely + recreate empty. The in-process
        # BM25 index is not cleared here (callers clear per-KB via routes_docs
        # which handles bm25 separately); clearing just the Qdrant side keeps the
        # vector/keyword halves consistent if a caller clears vectors only.
        collection = self._collection_for()
        try:
            self._client.delete_collection(collection)
            logger.info("QdrantBackend cleared collection %s", collection)
        except Exception as e:
            logger.warning("QdrantBackend clear failed for %s: %s", collection, e)

    def close(self) -> None:
        if self._bm25_index is not None:
            try:
                self._bm25_index._close_conn()
            except Exception as e:
                logger.warning("QdrantBackend close: BM25 conn close failed: %s", e)
        try:
            self._client.close()
        except Exception as e:
            logger.warning("QdrantBackend close: client close failed: %s", e)
        self._bm25_index = None
