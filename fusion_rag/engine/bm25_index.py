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

from .sqlite_base import SqliteBase

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


class BM25Index(SqliteBase):
    """Okapi BM25 index with SQLite persistence."""

    def __init__(self, store_path: str, k1: float = 1.5, b: float = 0.75):
        self.store_path = Path(store_path)
        self.db_path = str(self.store_path)
        self.k1 = k1
        self.b = b
        self._corpus_size = 0
        self._avgdl = 0.0
        self._df: Counter = Counter()
        self._doc_len: dict[str, int] = {}
        self._inverted: dict[str, dict[str, int]] = {}
        self._doc_texts: dict[str, str] = {}
        self._doc_paths: dict[str, str] = {}
        # D6: degraded flag. A load failure used to warn + silently start an
        # empty index — keyword recall vanished with no signal. Now mark
        # degraded + log ERROR so /status can surface it and search() refuses
        # rather than silently returning empty (which reads as "no matches").
        self._degraded = False
        self._degraded_reason = ""
        super().__init__()
        self._init_db()
        self._load()

    def _init_db(self) -> None:
        # P4-2: prior `self._db = self._get_conn()` bypassed SqliteBase's lazy
        # contract — _get_conn sets _db_closed=False internally, but a direct
        # assign left _db_closed=True (set in __init__), so the next _get_conn
        # saw _db_closed==True, re-entered the lock, and opened a SECOND
        # connection to the same file — the first never closed (leak per
        # instance). Call _get_conn() and use the return; let it manage _db + _db_closed.
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._get_conn()
        conn.execute("CREATE TABLE IF NOT EXISTS bm25_meta (key TEXT PRIMARY KEY, value TEXT)")
        conn.execute("CREATE TABLE IF NOT EXISTS bm25_docs (doc_id TEXT PRIMARY KEY, text TEXT, doc_len INTEGER)")
        # L5: store doc_path per chunk so remove_document can match exactly
        # instead of substring-scanning every doc text (which deleted chunks
        # whose text merely contained the path string).
        cols = {row[1] for row in conn.execute("PRAGMA table_info(bm25_docs)").fetchall()}
        if "doc_path" not in cols:
            conn.execute("ALTER TABLE bm25_docs ADD COLUMN doc_path TEXT NOT NULL DEFAULT ''")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS bm25_inverted "
            "(token TEXT, doc_id TEXT, tf INTEGER, PRIMARY KEY (token, doc_id))"
        )
        # P-P2-1: back the remove_document DELETE (WHERE doc_id IN (...)) with
        # an index on doc_id. The PRIMARY KEY (token, doc_id) orders by token
        # first, so a doc_id-only delete scans every token otherwise.
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_bm25_inverted_doc ON bm25_inverted(doc_id)"
        )
        conn.commit()

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
            # D6: was logger.warning(... starting fresh) — a corrupt/locked
            # bm25_index.db silently wiped the keyword index, keyword_search
            # returned empty forever (reads as "no matches", not "broken").
            # Mark degraded + ERROR so the operator sees it; search() then
            # refuses rather than serving empty results as if they were real.
            self._degraded = True
            self._degraded_reason = f"DB error: {e}"
            logger.error(
                "BM25 index load FAILED (DB error) — keyword search DEGRADED, "
                "refusing empty results instead of silent wipe: %s",
                e,
            )
        except Exception as e:
            self._degraded = True
            self._degraded_reason = f"structural error: {e}"
            logger.error(
                "BM25 index load FAILED (structural error) — keyword search DEGRADED: %s",
                e,
            )

    def add_documents(self, chunks: list[dict[str, Any]]) -> None:
        if not chunks:
            return
        # P2: collect only the new/changed (token, doc_id) postings this call so
        # _persist_delta can upsert just those rows — not wipe + rewrite the
        # whole inverted table (O(corpus) per doc on a large KB).
        added_docs: dict[str, tuple[str, int, str]] = {}
        added_postings: list[tuple[str, str, int]] = []
        # P0-1/P3-2: track doc_ids already in the index. A re-index of the same
        # chunk (watch re-index, replace) used to increment _corpus_size again
        # — inflating IDF denominators and double-counting doc length. Now an
        # existing doc_id is an upsert, not a new doc. _df updated incrementally
        # per touched token instead of a full O(corpus) rebuild every call.
        new_count = 0
        for chunk in chunks:
            doc_id = chunk.get("id", "")
            if not doc_id:
                continue
            text = chunk.get("text", "")
            context = chunk.get("context", "")
            full_text = (context + " " + text).strip() if context else text
            is_new = doc_id not in self._doc_texts
            if is_new:
                new_count += 1
                self._corpus_size += 1
            else:
                # upsert of an existing chunk: retract its old postings from
                # _df / _inverted before re-inserting, so the tf reflects the
                # new text, not the old + new.
                old_tokens = _tokenize(self._doc_texts.get(doc_id, ""))
                for ot in set(old_tokens):
                    post = self._inverted.get(ot)
                    if post and doc_id in post:
                        del post[doc_id]
                        if not post:
                            del self._inverted[ot]
            self._doc_texts[doc_id] = full_text
            self._doc_paths[doc_id] = chunk.get("doc_path", "") or ""
            tokens = _tokenize(full_text)
            self._doc_len[doc_id] = len(tokens)
            added_docs[doc_id] = (full_text, len(tokens), self._doc_paths[doc_id])
            tf = Counter(tokens)
            for token, freq in tf.items():
                if token not in self._inverted:
                    self._inverted[token] = {}
                self._inverted[token][doc_id] = freq
                added_postings.append((token, doc_id, freq))
                # P3-2: incremental _df — only the touched token's df changes,
                # not every token's. Rebuild was O(corpus) per add.
                self._df[token] = len(self._inverted[token])
        total_len = sum(self._doc_len.values())
        self._avgdl = total_len / max(self._corpus_size, 1)
        self._persist_delta(added_docs, added_postings, removed_doc_ids=None)
        logger.info(
            "BM25 index updated: %d docs (+%d new), %d tokens", self._corpus_size, new_count, len(self._inverted)
        )

    def remove_document(self, doc_path: str) -> int:
        # L5: exact doc_path match against the stored per-chunk doc_path field,
        # never substring. The old `doc_path in txt or doc_path in str(did)`
        # deleted any chunk whose text happened to contain the path string.
        to_remove = [did for did, dp in self._doc_paths.items() if dp == doc_path]
        if not to_remove:
            return 0
        removed_set = set(to_remove)
        # P-P2-1: capture each removed doc's tokens BEFORE deleting its text so
        # the affected-token set is known directly (O(doc tokens)). The prior
        # loop scanned the ENTIRE _inverted dict per removed doc to find which
        # tokens posted it — O(unique corpus tokens) per delete, a 50k-token KB
        # did 50k iterations to delete one doc. Re-tokenizing the (small) doc
        # text yields exactly the tokens to retract.
        removed_tokens_by_doc: dict[str, list[str]] = {}
        total_len_delta = 0
        for did in to_remove:
            removed_tokens_by_doc[did] = _tokenize(self._doc_texts.get(did, ""))
            del self._doc_texts[did]
            total_len_delta += self._doc_len.pop(did, 0)
            self._doc_paths.pop(did, None)
            self._corpus_size -= 1
        # P3-2: decrement _df incrementally per affected token instead of
        # rebuilding the whole Counter (O(unique tokens) per delete). Only the
        # tokens that appeared in a removed doc change df; pruning an empty
        # posting list also drops the df entry. P-P2-1: iterate the captured
        # doc tokens (deduped), not a corpus-wide inverted scan.
        for did, tokens in removed_tokens_by_doc.items():
            for token in set(tokens):
                post = self._inverted.get(token)
                if post and did in post:
                    del post[did]
                    if not post:
                        del self._inverted[token]
                        self._df.pop(token, None)
                    else:
                        self._df[token] = len(post)
        # _avgdl from cached total length, not a full O(corpus) sum.
        total_len = max(self._avgdl * (self._corpus_size + len(to_remove)) - total_len_delta, 0.0)
        self._avgdl = total_len / max(self._corpus_size, 1)
        # P2: delete only the rows for the removed doc_ids, not the whole table.
        self._persist_delta(added_docs=None, added_postings=None, removed_doc_ids=removed_set)
        logger.info("BM25 index: removed %d chunks for doc %s", len(to_remove), doc_path)
        return len(to_remove)

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        # D6: a degraded index (load failed) must NOT silently return [].
        # Empty results are indistinguishable from "no matches" to a caller;
        # surface the degradation so the caller/hybrid layer can fall back to
        # vector-only and log it, rather than pretending keyword recall works.
        if self._degraded:
            logger.warning(
                "BM25 search refused (index degraded: %s) — keyword recall unavailable, falling back to vector-only",
                self._degraded_reason,
            )
            return []
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

    def _persist_delta(
        self,
        added_docs: dict[str, tuple[str, int, str]] | None,
        added_postings: list[tuple[str, str, int]] | None,
        removed_doc_ids: set[str] | None,
    ) -> None:
        # P2: incremental persistence — upsert only the docs/postings touched
        # this call and delete only the rows for removed doc_ids. Before this
        # _save_to_db did `DELETE FROM bm25_inverted` + full reinsert every
        # add/remove, O(corpus) per doc — a 10k-doc KB rewrote the whole
        # inverted table on every single ingest.
        #
        # L6: save failure must not be swallowed — the in-memory index would
        # then diverge from disk. Log AND re-raise so callers (add_documents /
        # remove_document) can decide; the route layer's L7 rollback treats a
        # BM25 write failure as an indexing failure.
        try:
            stats = json.dumps({"corpus_size": self._corpus_size, "avgdl": self._avgdl})
            self._db.execute("INSERT OR REPLACE INTO bm25_meta VALUES ('stats', ?)", (stats,))
            if added_docs:
                for doc_id, (text, dl, dp) in added_docs.items():
                    self._db.execute(
                        "INSERT OR REPLACE INTO bm25_docs (doc_id, text, doc_len, doc_path) VALUES (?, ?, ?, ?)",
                        (doc_id, text, dl, dp),
                    )
            if added_postings:
                # executemany upserts the touched postings; PRIMARY KEY (token,
                # doc_id) makes INSERT OR REPLACE overwrite a re-indexed chunk
                # in place rather than appending a duplicate.
                self._db.executemany(
                    "INSERT OR REPLACE INTO bm25_inverted (token, doc_id, tf) VALUES (?, ?, ?)",
                    added_postings,
                )
            if removed_doc_ids:
                placeholders = ",".join("?" for _ in removed_doc_ids)
                self._db.execute(
                    f"DELETE FROM bm25_inverted WHERE doc_id IN ({placeholders})",
                    tuple(removed_doc_ids),
                )
                self._db.execute(
                    f"DELETE FROM bm25_docs WHERE doc_id IN ({placeholders})",
                    tuple(removed_doc_ids),
                )
            self._db.commit()
        except Exception as e:
            logger.error("BM25 index persist failed: %s", e)
            raise
