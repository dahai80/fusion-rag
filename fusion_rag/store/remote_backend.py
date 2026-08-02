from __future__ import annotations

import logging
from typing import Any

from .store_backend import StoreBackend

logger = logging.getLogger(__name__)


class RemoteBackend(StoreBackend):
    def __init__(self, endpoint: str = "", api_key: str = "", kb_id: str = "", **kwargs):
        self.endpoint = endpoint
        self.api_key = api_key
        self.kb_id = kb_id
        logger.info("RemoteBackend initialized (stub): endpoint=%s kb_id=%s", endpoint, kb_id)

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
        logger.warning("RemoteBackend.add not yet implemented — data not stored remotely")
        raise NotImplementedError("RemoteBackend.add: not connected to fusion-multi-nodes")

    def add_batch(self, records: list[dict[str, Any]]) -> None:
        logger.warning("RemoteBackend.add_batch not yet implemented")
        raise NotImplementedError("RemoteBackend.add_batch: not connected to fusion-multi-nodes")

    def search(self, query_vector: list[float], top_k: int = 10, threshold: float = 0.0) -> list[dict[str, Any]]:
        logger.warning("RemoteBackend.search not yet implemented")
        raise NotImplementedError("RemoteBackend.search: not connected to fusion-multi-nodes")

    def keyword_search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        logger.warning("RemoteBackend.keyword_search not yet implemented")
        raise NotImplementedError("RemoteBackend.keyword_search: not connected to fusion-multi-nodes")

    def delete_by_doc(self, doc_path: str) -> int:
        logger.warning("RemoteBackend.delete_by_doc not yet implemented")
        raise NotImplementedError("RemoteBackend.delete_by_doc: not connected to fusion-multi-nodes")

    def count(self) -> int:
        return 0

    def clear(self) -> None:
        logger.warning("RemoteBackend.clear not yet implemented")

    def close(self) -> None:
        pass
