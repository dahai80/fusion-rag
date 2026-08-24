"""BM25 index — Okapi BM25 with Chinese/English tokenization.

callers: VectorStore.keyword_search(), HybridSearch.search(), KnowledgeBase intake flow
API: BM25Index.add_documents(), search(), remove_document(), save(), load(), count()
schema: internal _inverted {token: {doc_id: tf}}, _doc_texts {doc_id: text}, persisted to SQLite
user instruction: "按照你的方案和计划落地所有phase阶段的需求"
"""

from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# callers: VectorStore (vector_store.py:115-117), HybridSearch
# API: BM25Index — public class unchanged, only _tokenize() enhanced
# data: tech_dict.txt — jieba user dict format: "词语 词频 词性"
# user instruction: "完成所有待办任务"
_JIEBA_AVAILABLE = False
_JIEBA_DICT_LOADED = False
try:
    import jieba

    _JIEBA_AVAILABLE = True
    _dict_path = Path(__file__).parent.parent / "data" / "tech_dict.txt"
    if _dict_path.exists():
        jieba.load_userdict(str(_dict_path))
        _JIEBA_DICT_LOADED = True
        logger.debug("Loaded tech dict: %s", _dict_path)
except ImportError:
    pass


def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    tokens = []
    chinese_segs = re.findall(r"[一-鿿]+", text)
    if _JIEBA_AVAILABLE:
        for seg in chinese_segs:
            tokens.extend(jieba.lcut(seg))
    else:
        for seg in chinese_segs:
            tokens.extend(seg)
    english_segs = re.findall(r"[a-zA-Z0-9]+", text)
    tokens.extend(w.lower() for w in english_segs)
    return [t for t in tokens if len(t) > 1]


class BM25Index:
    """Okapi BM25 index with SQLite persistence."""

    def __init__(self, store_path: str, k1: float = 1.5, b: float = 0.75):
        self.store_path = Path(store_path)
        self.k1 = k1
        self.b = b
        self._corpus_size = 0
        self._avgdl = 0.0
        self._df: Counter = Counter()
        self._doc_len: dict[str, int] = {}
        self._inverted: dict[str, dict[str, int]] = {}
        self._doc_texts: dict[str, str] = {}
        self._doc_paths: dict[str, str] = {}
        self._db: sqlite3.Connection | None = None
        self._init_db()
        self._load()

    def _init_db(self) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.store_path))
        self._db.execute("CREATE TABLE IF NOT EXISTS bm25_meta (key TEXT PRIMARY KEY, value TEXT)")
        self._db.execute("CREATE TABLE IF NOT EXISTS bm25_docs (doc_id TEXT PRIMARY KEY, text TEXT, doc_len INTEGER)")
        # L5: store doc_path per chunk so remove_document can match exactly
        # instead of substring-scanning every doc text (which deleted chunks
        # whose text merely contained the path string).
        cols = {row[1] for row in self._db.execute("PRAGMA table_info(bm25_docs)").fetchall()}
        if "doc_path" not in cols:
            self._db.execute("ALTER TABLE bm25_docs ADD COLUMN doc_path TEXT NOT NULL DEFAULT ''")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS bm25_inverted "
            "(token TEXT, doc_id TEXT, tf INTEGER, PRIMARY KEY (token, doc_id))"
        )
        self._db.commit()

    def _load(self) -> None:
        # L6: distinguish a recoverable SQLite operational error (corrupt/locked
        # file → starting fresh is safe) from a structural/parse error that
        # signals the on-disk format is wrong. The former warns and recovers;
        # the latter is logged loudly because "starting fresh" here would SILENTLY
        # discard a valid index the user expects to search.
        try:
            row = self._db.execute("SELECT value FROM bm25_meta WHERE key='stats'").fetchone()
            if row:
                stats = json.loads(row[0])
                self._corpus_size = stats.get("corpus_size", 0)
                self._avgdl = stats.get("avgdl", 0.0)
            for doc_id, text, doc_len, doc_path in self._db.execute(
                "SELECT doc_id, text, doc_len, doc_path FROM bm25_docs"
            ):
                self._doc_texts[doc_id] = text
                self._doc_len[doc_id] = doc_len
                self._doc_paths[doc_id] = doc_path or ""
            for token, doc_id, tf in self._db.execute("SELECT token, doc_id, tf FROM bm25_inverted"):
                if token not in self._inverted:
                    self._inverted[token] = {}
                self._inverted[token][doc_id] = tf
            self._df = Counter()
            for token, postings in self._inverted.items():
                self._df[token] = len(postings)
            logger.debug("BM25 index loaded: %d docs, %d tokens", self._corpus_size, len(self._inverted))
        except sqlite3.Error as e:
            logger.warning("BM25 index load failed (DB error, starting fresh): %s", e)
        except Exception as e:
            logger.error("BM25 index load failed (structural error, starting fresh): %s", e)

    def add_documents(self, chunks: list[dict[str, Any]]) -> None:
        if not chunks:
            return
        for chunk in chunks:
            doc_id = chunk.get("id", "")
            if not doc_id:
                continue
            text = chunk.get("text", "")
            context = chunk.get("context", "")
            full_text = (context + " " + text).strip() if context else text
            self._doc_texts[doc_id] = full_text
            self._doc_paths[doc_id] = chunk.get("doc_path", "") or ""
            tokens = _tokenize(full_text)
            self._doc_len[doc_id] = len(tokens)
            tf = Counter(tokens)
            for token, freq in tf.items():
                if token not in self._inverted:
                    self._inverted[token] = {}
                self._inverted[token][doc_id] = freq
            self._corpus_size += 1
        self._df = Counter()
        for token, postings in self._inverted.items():
            self._df[token] = len(postings)
        total_len = sum(self._doc_len.values())
        self._avgdl = total_len / max(self._corpus_size, 1)
        self._save_to_db()
        logger.info("BM25 index updated: %d docs, %d tokens", self._corpus_size, len(self._inverted))

    def remove_document(self, doc_path: str) -> int:
        # L5: exact doc_path match against the stored per-chunk doc_path field,
        # never substring. The old `doc_path in txt or doc_path in str(did)`
        # deleted any chunk whose text happened to contain the path string.
        to_remove = [did for did, dp in self._doc_paths.items() if dp == doc_path]
        if not to_remove:
            return 0
        for did in to_remove:
            del self._doc_texts[did]
            self._doc_len.pop(did, None)
            self._doc_paths.pop(did, None)
            self._corpus_size -= 1
        for token in list(self._inverted.keys()):
            self._inverted[token] = {k: v for k, v in self._inverted[token].items() if k not in set(to_remove)}
            if not self._inverted[token]:
                del self._inverted[token]
        self._df = Counter()
        for token, postings in self._inverted.items():
            self._df[token] = len(postings)
        total_len = sum(self._doc_len.values())
        self._avgdl = total_len / max(self._corpus_size, 1)
        self._save_to_db()
        logger.info("BM25 index: removed %d chunks for doc %s", len(to_remove), doc_path)
        return len(to_remove)

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        if not query or self._corpus_size == 0:
            return []
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []
        scores: dict[str, float] = {}
        for token in query_tokens:
            if token not in self._inverted:
                continue
            df = self._df.get(token, 0)
            idf = math.log((self._corpus_size - df + 0.5) / (df + 0.5) + 1)
            for doc_id, tf in self._inverted[token].items():
                dl = self._doc_len.get(doc_id, 1)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / max(self._avgdl, 1))
                scores[doc_id] = scores.get(doc_id, 0) + idf * numerator / denominator
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [{"id": did, "score": s, "text": self._doc_texts.get(did, "")} for did, s in ranked]

    def count(self) -> int:
        return self._corpus_size

    def _save_to_db(self) -> None:
        # L6: save failure must not be swallowed — the in-memory index would
        # then diverge from disk (this run's additions lost on next restart,
        # or a partial write left on disk). Log AND re-raise so callers
        # (add_documents/remove_document) can decide; the route layer's L7
        # rollback treats a BM25 write failure as an indexing failure.
        try:
            stats = json.dumps(
                {
                    "corpus_size": self._corpus_size,
                    "avgdl": self._avgdl,
                }
            )
            self._db.execute("INSERT OR REPLACE INTO bm25_meta VALUES ('stats', ?)", (stats,))
            for doc_id, text in self._doc_texts.items():
                dl = self._doc_len.get(doc_id, 0)
                dp = self._doc_paths.get(doc_id, "")
                self._db.execute(
                    "INSERT OR REPLACE INTO bm25_docs (doc_id, text, doc_len, doc_path) VALUES (?, ?, ?, ?)",
                    (doc_id, text, dl, dp),
                )
            self._db.execute("DELETE FROM bm25_inverted")
            for token, postings in self._inverted.items():
                for doc_id, tf in postings.items():
                    self._db.execute(
                        "INSERT INTO bm25_inverted VALUES (?, ?, ?)",
                        (token, doc_id, tf),
                    )
            self._db.commit()
        except Exception as e:
            logger.error("BM25 index save failed: %s", e)
            raise
