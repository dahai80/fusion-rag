"""Vector store — LanceDB-based vector storage and retrieval.

All vector storage uses LanceDB for local, persistent, Apple Silicon-optimized storage.
LanceDB and PyArrow are imported lazily.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


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


class VectorStore:
    """LanceDB-based vector store for knowledge base chunks."""

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
        schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), self.dimension)),
            pa.field("text", pa.string()),
            pa.field("doc_path", pa.string()),
            pa.field("doc_name", pa.string()),
            pa.field("doc_type", pa.string()),
            pa.field("chunk_index", pa.int32()),
            pa.field("metadata_json", pa.string()),
        ])
        try:
            self._table = self._db.open_table(table_name)
        except Exception:
            self._table = self._db.create_table(table_name, schema=schema)

    @property
    def table(self):
        if self._table is None:
            self._connect()
        return self._table

    def add(self, chunk_id: str, vector: list[float], text: str,
            doc_path: str = "", doc_name: str = "", doc_type: str = "",
            chunk_index: int = 0, metadata: dict | None = None) -> None:
        data = [{
            "id": chunk_id, "vector": vector, "text": text,
            "doc_path": doc_path, "doc_name": doc_name, "doc_type": doc_type,
            "chunk_index": chunk_index,
            "metadata_json": json.dumps(metadata or {}, ensure_ascii=False),
        }]
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

    def search(self, query_vector: list[float], top_k: int = 10,
               threshold: float = 0.0) -> list[dict[str, Any]]:
        try:
            results = self.table.search(query_vector).metric("cosine").limit(top_k).to_list()
            filtered = []
            for r in results:
                score = 1.0 - r.get("_distance", 0.0)
                if score < threshold:
                    continue
                r["score"] = score
                r.pop("_distance", None)
                try:
                    r["metadata"] = json.loads(r.get("metadata_json", "{}"))
                except Exception:
                    r["metadata"] = {}
                r.pop("metadata_json", None)
                filtered.append(r)
            return filtered
        except Exception as e:
            logger.error("Vector search failed: %s", e)
            return []

    @property
    def bm25(self):
        if self._bm25_index is None:
            from ..engine.bm25_index import BM25Index
            bm25_path = str(Path(self.vector_path).parent / "bm25_index.db")
            self._bm25_index = BM25Index(bm25_path)
        return self._bm25_index

    def keyword_search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        # BM25-based keyword search (replaces zero-vector + substring counting)
        return self.bm25.search(query, top_k)

    def count(self) -> int:
        try:
            return self.table.count_rows()
        except Exception:
            return 0

    # callers: KnowledgeBase.delete_document(), tests
    # API: delete_by_doc(doc_path: str) -> int
    # schema: LanceDB chunks table + BM25 index, DeleteResult.rows_deleted or int return
    # user instruction: "bug/问题/需求 修改完成，遇到报错的用例，不管和自己的代码是否相关，都要定位修复"
    def delete_by_doc(self, doc_path: str) -> int:
        safe = doc_path.replace("'", "''")
        try:
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
        except Exception:
            return 0

    def clear(self) -> None:
        try:
            self.table.delete("true")
        except Exception:
            pass

    def close(self) -> None:
        self._db = None
        self._table = None
        self._bm25_index = None