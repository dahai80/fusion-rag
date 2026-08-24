from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .store_backend import StoreBackend

logger = logging.getLogger(__name__)


def _lancedb():
    try:
        import lancedb

        return lancedb
    except ImportError:
        raise ImportError("Install lancedb: pip install lancedb")


def _pa():
    try:
        import pyarrow as pa

        return pa
    except ImportError:
        raise ImportError("Install pyarrow: pip install pyarrow")


class LocalBackend(StoreBackend):
    def __init__(self, vector_path: str, dimension: int = 1024):
        self.vector_path = vector_path
        self.dimension = dimension
        self._db = None
        self._table = None
        self._bm25_index = None
        self._connect()

    def _connect(self) -> None:
        Path(self.vector_path).parent.mkdir(parents=True, exist_ok=True)
        ldb = _lancedb()
        self._db = ldb.connect(str(self.vector_path))
        table_name = "chunks"
        pa = _pa()
        schema = pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), self.dimension)),
                pa.field("text", pa.string()),
                pa.field("doc_path", pa.string()),
                pa.field("doc_name", pa.string()),
                pa.field("doc_type", pa.string()),
                pa.field("chunk_index", pa.int32()),
                pa.field("metadata_json", pa.string()),
                pa.field("context", pa.string()),
            ]
        )
        try:
            self._table = self._db.open_table(table_name)
            self._migrate_schema(self._table, schema)
        except Exception as e:
            # F8: only rebuild on a missing-table condition; any other error
            # (corrupt index, schema mismatch, IO) must surface — silently
            # rebuilding an empty table wiped real data on transient faults.
            msg = str(e).lower()
            if "does not exist" in msg or "not found" in msg or isinstance(e, FileNotFoundError):
                logger.info("LocalBackend table '%s' absent, creating new", table_name)
                self._table = self._db.create_table(table_name, schema=schema)
            else:
                logger.error("LocalBackend open_table failed (not auto-rebuilding): %s", e)
                raise

    def _migrate_schema(self, table, target_schema) -> None:
        existing_fields = {f.name for f in table.schema}
        missing = [f for f in target_schema if f.name not in existing_fields]
        if not missing:
            return
        for field in missing:
            logger.info("Migrating schema: adding column '%s'", field.name)
            # F8: a failed migration leaves a half-applied schema — abort loudly
            # rather than swallow + continue on a corrupt table.
            table.add_columns({field.name: "''"})

    @property
    def table(self):
        if self._table is None:
            self._connect()
        return self._table

    @property
    def bm25(self):
        if self._bm25_index is None:
            from ..engine.bm25_index import BM25Index

            bm25_path = str(Path(self.vector_path).parent / "bm25_index.db")
            self._bm25_index = BM25Index(bm25_path)
        return self._bm25_index

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
        data = [
            {
                "id": chunk_id,
                "vector": vector,
                "text": text,
                "doc_path": doc_path,
                "doc_name": doc_name,
                "doc_type": doc_type,
                "chunk_index": chunk_index,
                "metadata_json": json.dumps(metadata or {}, ensure_ascii=False),
                "context": context,
            }
        ]
        self.table.add(data)
        self.bm25.add_documents(data)

    def add_batch(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        for r in records:
            if "metadata_json" not in r and "metadata" in r:
                r["metadata_json"] = json.dumps(r.pop("metadata", {}), ensure_ascii=False)
        self.table.add(records)
        self.bm25.add_documents(records)

    def search(self, query_vector: list[float], top_k: int = 10, threshold: float = 0.0) -> list[dict[str, Any]]:
        # M3: a genuine search failure (corrupt index, IO) used to return [],
        # indistinguishable from a legitimate "no matches". Callers (HybridSearch,
        # routes_search) cannot tell a broken store from an empty one. Let it
        # raise — an empty table returns [] normally (verified), so only real
        # errors propagate. metadata-JSON parse stays a per-row warning (one
        # bad row shouldn't fail the whole search).
        try:
            results = self.table.search(query_vector).metric("cosine").limit(top_k).to_list()
        except Exception as e:
            logger.error("LocalBackend search failed (propagating, no silent []): %s", e)
            raise
        filtered = []
        for r in results:
            score = 1.0 - r.get("_distance", 0.0)
            if score < threshold:
                continue
            r["score"] = score
            r.pop("_distance", None)
            try:
                r["metadata"] = json.loads(r.get("metadata_json", "{}"))
            except Exception as e:
                logger.warning("Failed to parse metadata JSON: %s", e)
                r["metadata"] = {}
            r.pop("metadata_json", None)
            filtered.append(r)
        return filtered

    def keyword_search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        return self.bm25.search(query, top_k)

    def delete_by_doc(self, doc_path: str) -> int:
        # M3: failure used to return 0, indistinguishable from "no matching doc".
        # Callers (routes_docs delete/replace/scan) ignored the return and
        # reported success either way — a broken delete looked like "deleted 0".
        # Let it raise so the route surfaces a real failure. The return value
        # still means "matched rows" on success.
        safe = doc_path.replace("'", "''")
        result = self.table.delete(f"doc_path = '{safe}'")
        if isinstance(result, int):
            count = result
        elif hasattr(result, "__int__"):
            count = int(result)
        elif hasattr(result, "rows_deleted"):
            count = result.rows_deleted
        else:
            logger.debug("delete_by_doc returned unexpected type: %s", type(result))
            count = 0
        self.bm25.remove_document(doc_path)
        return count

    def count(self) -> int:
        # M3: failure used to return 0, indistinguishable from an empty store.
        # Stats endpoints would report "0 vectors" on a corrupt index. Raise.
        return self.table.count_rows()

    def clear(self) -> None:
        try:
            self.table.delete("true")
        except Exception as e:
            logger.warning("LocalBackend clear failed: %s", e)

    def close(self) -> None:
        self._db = None
        self._table = None
        self._bm25_index = None
