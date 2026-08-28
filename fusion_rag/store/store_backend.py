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
        # P2-10: a silent overwrite on re-register hid import-time global
        # mutation (a third party re-importing vector_store clobbered "local").
        # Detect a duplicate name and log loudly — the new class wins (late
        # registration is a plugin pattern) but the prior class is surfaced so
        # the clobber is visible, not silent.
        prior = cls._registry.get(name)
        if prior is not None and prior is not backend_class:
            logger.warning(
                "Store backend '%s' re-registered: %s -> %s (late registration wins)", name, prior, backend_class
            )
        cls._registry[name] = backend_class
        logger.info("Registered store backend: %s (available: %s)", name, cls.available_types())

    @classmethod
    def create(cls, store_type: str = "local", *, fallback: bool = False, **kwargs) -> StoreBackend:
        # P2-10: unknown type used to silently fall back to "local" — but "local"
        # is a different storage engine (squared-L2 vs cosine, P0-10). A typo in
        # FUSION_RAG_STORE_BACKEND ("fusion_store" vs "fusion-store") would
        # silently run LanceDB instead of the intended backend with no error, no
        # log beyond a warning, and different retrieval semantics. Fail visibly:
        # raise unless the caller explicitly opts into fallback.
        if store_type not in cls._registry:
            if fallback:
                logger.warning(
                    "Unknown store type '%s' (fallback=True) -> 'local'. Available: %s",
                    store_type,
                    cls.available_types(),
                )
                store_type = "local"
            else:
                logger.error("Unknown store type '%s'. Available: %s", store_type, cls.available_types())
                raise ValueError(
                    f"Unknown store backend type: {store_type!r}. "
                    f"Available: {cls.available_types()}. "
                    "Check FUSION_RAG_STORE_BACKEND spelling, or pass fallback=True for legacy local fallback."
                )
        backend_cls = cls._registry[store_type]
        return backend_cls(**kwargs)

    @classmethod
    def available_types(cls) -> list[str]:
        return list(cls._registry.keys())
