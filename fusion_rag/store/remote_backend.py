from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from .store_backend import StoreBackend

logger = logging.getLogger(__name__)


def _httpx():
    try:
        import httpx

        return httpx
    except ImportError:
        raise ImportError("RemoteBackend requires httpx: pip install httpx")


# Wire-format record keys (matches LocalBackend result shape). The server
# (routes_store) echoes these back on search so the client rebuilds the exact
# same dict LocalBackend.search would have returned.
_RECORD_KEYS = (
    "id",
    "vector",
    "text",
    "doc_path",
    "doc_name",
    "doc_type",
    "chunk_index",
    "metadata",
    "context",
)


class RemoteBackend(StoreBackend):
    """HTTP-backed StoreBackend — delegates to a remote fusion-rag node.

    The matching server half is routes_store.py, mounted at
    /kb/bases/{kb_id}/store/* on the remote node. One fusion-rag instance can
    thus act as the vector store for another (or for any client speaking the
    contract). Metadata stays local (MetadataStore is per-KB sqlite at the
    caller); only the vector + BM25 surface is remote. This is the split-store
    model: the remote owns vectors/keyword search; the local owns document
    metadata, audit, trajectory.

    Config is env-driven (single-embedding-model / single-backend model, same as
    FUSION_RAG_EMBED):
      FUSION_RAG_REMOTE_ENDPOINT  — base URL of the remote node (e.g. http://node-b:11436)
      FUSION_RAG_REMOTE_API_KEY   — api key for the remote node (empty = no auth)
      FUSION_RAG_REMOTE_KB_ID     — the KB id on the remote (defaults to this
                                    node's kb_id, derived from vector_path)
      FUSION_RAG_REMOTE_TIMEOUT   — per-request timeout seconds (default 30)
    """

    def __init__(
        self,
        vector_path: str = "",
        dimension: int = 1024,
        endpoint: str = "",
        api_key: str = "",
        kb_id: str = "",
        timeout: float = 30.0,
        **_,
    ):
        self.vector_path = vector_path
        self.dimension = dimension
        self.endpoint = endpoint or os.environ.get("FUSION_RAG_REMOTE_ENDPOINT", "")
        self.api_key = api_key or os.environ.get("FUSION_RAG_REMOTE_API_KEY", "")
        remote_kb = kb_id or os.environ.get("FUSION_RAG_REMOTE_KB_ID", "")
        if not remote_kb:
            # Default remote kb_id = this node's kb_id (the leaf of vector_path).
            # vector_path is ~/.fusion-rag/stores/{kb_id}/vectors — the kb_id is
            # the parent-of-vectors dir name. Falls back to "" if path is bare.
            leaf = Path(vector_path).parent.name if vector_path else ""
            remote_kb = leaf if leaf and leaf != "vectors" else ""
        self.kb_id = remote_kb
        try:
            self.timeout = float(os.environ.get("FUSION_RAG_REMOTE_TIMEOUT", str(timeout)))
        except ValueError:
            self.timeout = timeout
        if not self.endpoint:
            raise ValueError(
                "RemoteBackend requires FUSION_RAG_REMOTE_ENDPOINT "
                "(the remote fusion-rag node base URL, e.g. http://node-b:11436)"
            )
        if not self.kb_id:
            raise ValueError(
                "RemoteBackend requires a remote kb_id — set FUSION_RAG_REMOTE_KB_ID "
                "or pass kb_id= (could not derive one from vector_path)"
            )
        self._client = None
        logger.info(
            "RemoteBackend initialized: endpoint=%s kb_id=%s timeout=%ss",
            self.endpoint,
            self.kb_id,
            self.timeout,
        )

    @property
    def client(self):
        if self._client is None:
            httpx = _httpx()
            headers = {}
            if self.api_key:
                headers["X-API-Key"] = self.api_key
            self._client = httpx.Client(base_url=self.endpoint, headers=headers, timeout=self.timeout)
        return self._client

    def _url(self, action: str) -> str:
        return f"/kb/bases/{self.kb_id}/store/{action}"

    def _post(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            r = self.client.post(self._url(action), json=payload)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.error("RemoteBackend %s failed: %s", action, e)
            raise

    def _get(self, action: str) -> dict[str, Any]:
        try:
            r = self.client.get(self._url(action))
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.error("RemoteBackend %s failed: %s", action, e)
            raise

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
        self._post("add_batch", {"records": [record]})
        logger.debug("RemoteBackend add: chunk_id=%s", chunk_id)

    def add_batch(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        self._post("add_batch", {"records": records})
        logger.info("RemoteBackend add_batch: %d records", len(records))

    def search(self, query_vector: list[float], top_k: int = 10, threshold: float = 0.0) -> list[dict[str, Any]]:
        resp = self._post("search", {"query_vector": query_vector, "top_k": top_k, "threshold": threshold})
        results = resp.get("results", [])
        logger.info("RemoteBackend search: top_k=%d returned=%d", top_k, len(results))
        return results

    def keyword_search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        resp = self._post("keyword_search", {"query": query, "top_k": top_k})
        results = resp.get("results", [])
        logger.info("RemoteBackend keyword_search: returned=%d", len(results))
        return results

    def delete_by_doc(self, doc_path: str) -> int:
        resp = self._post("delete_by_doc", {"doc_path": doc_path})
        deleted = int(resp.get("deleted", 0))
        logger.info("RemoteBackend delete_by_doc: doc_path=%s deleted=%d", doc_path, deleted)
        return deleted

    def count(self) -> int:
        resp = self._get("count")
        return int(resp.get("count", 0))

    def clear(self) -> None:
        self._post("clear", {})
        logger.info("RemoteBackend clear: kb_id=%s", self.kb_id)

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception as e:
                logger.warning("RemoteBackend close failed: %s", e)
            self._client = None
