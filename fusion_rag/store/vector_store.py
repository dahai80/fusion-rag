from __future__ import annotations

import logging
from typing import Any

from .local_backend import LocalBackend
from .remote_backend import RemoteBackend
from .store_backend import StoreBackend, StoreBackendFactory

logger = logging.getLogger(__name__)


StoreBackendFactory.register("local", LocalBackend)
StoreBackendFactory.register("remote", RemoteBackend)
try:
    from .fusion_store_backend import FusionStoreBackend

    StoreBackendFactory.register("fusion-store", FusionStoreBackend)
except ImportError:
    logger.debug("FusionStoreBackend unavailable (fusion_store not installed); backend 'fusion-store' disabled")


class VectorStore:
    def __init__(self, vector_path: str, dimension: int = 1024, backend_type: str = "local", **backend_kwargs):
        self.vector_path = vector_path
        self.dimension = dimension
        self.backend_type = backend_type

        # P4-5: route every backend (including local/remote/fusion-store) through
        # StoreBackendFactory.create uniformly. The prior hardcoded 3-branch
        # bypassed the factory for the built-ins, so a 4th registered backend
        # used a different path — inconsistent. Backends now accept **_ to ignore
        # params not meant for them (remote endpoint/api_key vs local vector_path),
        # so a single factory call serves all. Unknown backend_type raises here
        # (P2-10) rather than silently swapping to a different storage engine.
        self._backend: StoreBackend = StoreBackendFactory.create(
            store_type=backend_type,
            vector_path=vector_path,
            dimension=dimension,
            **backend_kwargs,
        )

        logger.info(
            "VectorStore initialized: path=%s backend=%s",
            vector_path,
            backend_type,
        )

    @property
    def backend(self) -> StoreBackend:
        return self._backend

    @property
    def table(self):
        # LanceDB table is a LocalBackend concept; fusion-store has no table.
        if hasattr(self._backend, "table"):
            return self._backend.table
        raise AttributeError("table property only available on LanceDB-backed stores")

    @property
    def bm25(self):
        # Any backend that owns an in-process BM25Index exposes it via .bm25
        # (LocalBackend and FusionStoreBackend both do).
        if hasattr(self._backend, "bm25"):
            return self._backend.bm25
        raise AttributeError("bm25 property not available on this backend")

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
        self._backend.add(
            chunk_id=chunk_id,
            vector=vector,
            text=text,
            doc_path=doc_path,
            doc_name=doc_name,
            doc_type=doc_type,
            chunk_index=chunk_index,
            metadata=metadata,
            context=context,
        )

    def add_batch(self, records: list[dict[str, Any]]) -> None:
        self._backend.add_batch(records)

    def search(self, query_vector: list[float], top_k: int = 10, threshold: float = 0.0) -> list[dict[str, Any]]:
        return self._backend.search(query_vector, top_k=top_k, threshold=threshold)

    def keyword_search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        return self._backend.keyword_search(query, top_k=top_k)

    def count(self) -> int:
        return self._backend.count()

    def delete_by_doc(self, doc_path: str) -> int:
        return self._backend.delete_by_doc(doc_path)

    def clear(self) -> None:
        self._backend.clear()

    def close(self) -> None:
        self._backend.close()
