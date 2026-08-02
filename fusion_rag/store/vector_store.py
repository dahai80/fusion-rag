from __future__ import annotations

import logging
from typing import Any

from .local_backend import LocalBackend
from .remote_backend import RemoteBackend
from .store_backend import StoreBackend, StoreBackendFactory

logger = logging.getLogger(__name__)


StoreBackendFactory.register("local", LocalBackend)
StoreBackendFactory.register("remote", RemoteBackend)


class VectorStore:
    def __init__(self, vector_path: str, dimension: int = 1024, backend_type: str = "local", **backend_kwargs):
        self.vector_path = vector_path
        self.dimension = dimension
        self.backend_type = backend_type

        if backend_type == "local":
            self._backend: StoreBackend = LocalBackend(vector_path=vector_path, dimension=dimension)
        elif backend_type == "remote":
            self._backend = RemoteBackend(**backend_kwargs)
        else:
            self._backend = StoreBackendFactory.create(
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
        if isinstance(self._backend, LocalBackend):
            return self._backend.table
        raise AttributeError("table property only available on LocalBackend")

    @property
    def bm25(self):
        if isinstance(self._backend, LocalBackend):
            return self._backend.bm25
        raise AttributeError("bm25 property only available on LocalBackend")

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
