from __future__ import annotations

import abc
import logging
from typing import Any, ClassVar

logger = logging.getLogger(__name__)


class StoreBackend(abc.ABC):
    @abc.abstractmethod
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
    ) -> None: ...

    @abc.abstractmethod
    def add_batch(self, records: list[dict[str, Any]]) -> None: ...

    @abc.abstractmethod
    def search(self, query_vector: list[float], top_k: int = 10, threshold: float = 0.0) -> list[dict[str, Any]]: ...

    @abc.abstractmethod
    def keyword_search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]: ...

    @abc.abstractmethod
    def delete_by_doc(self, doc_path: str) -> int: ...

    @abc.abstractmethod
    def count(self) -> int: ...

    @abc.abstractmethod
    def clear(self) -> None: ...

    @abc.abstractmethod
    def close(self) -> None: ...


class StoreBackendFactory:
    _registry: ClassVar[dict[str, type[StoreBackend]]] = {}

    @classmethod
    def register(cls, name: str, backend_class: type[StoreBackend]) -> None:
        cls._registry[name] = backend_class
        logger.info("Registered store backend: %s", name)

    @classmethod
    def create(cls, store_type: str = "local", **kwargs) -> StoreBackend:
        if store_type not in cls._registry:
            logger.warning("Unknown store type '%s', falling back to 'local'", store_type)
            store_type = "local"
        backend_cls = cls._registry[store_type]
        return backend_cls(**kwargs)

    @classmethod
    def available_types(cls) -> list[str]:
        return list(cls._registry.keys())
