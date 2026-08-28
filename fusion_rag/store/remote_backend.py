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
        # P2-11: returning 0 made a misconfigured remote KB look healthy-but-empty
        # (/kb/status, /kb/stats report "0 vectors" with no error) while every
        # write/search already raised. A stub backend must be uniformly
        # unimplemented: count raises too, so a status probe surfaces the
        # misconfiguration instead of hiding it behind a green 0. Also guards
        # P2-10's legacy fallback path routing a typo'd backend to remote.
        logger.warning("RemoteBackend.count not yet implemented")
        raise NotImplementedError("RemoteBackend.count: not connected to fusion-multi-nodes")

    def clear(self) -> None:
        # P2-11: clear returning silently made "reset KB" appear to succeed while
        # doing nothing — an operator believes the store is wiped when it never
        # had data (or the remote is unreachable). Raise so reset is honest.
        logger.warning("RemoteBackend.clear not yet implemented")
        raise NotImplementedError("RemoteBackend.clear: not connected to fusion-multi-nodes")

    def close(self) -> None:
        # Stub holds no resources (no connection opened in __init__). No-op is
        # correct here, unlike count/clear which lie about data.
        pass
