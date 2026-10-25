# Issue #79: real recall benchmark — deterministic ≥500-chunk corpus (markdown
# + code mix through the REAL chunker), ≥50 queries with ground truth,
# recall@k + MRR, and the ablation the report asked for: BM25-only vs
# vector-only vs RRF fusion.
#
# Two embed modes:
#   - mode=mock  : deterministic hashed-bag-of-tokens vectors. Wiring /
#     no-regression evidence ONLY — never quality evidence. Every output
#     carries the explicit "mode=mock" label.
#   - mode=real  : live fusion-mlx BGE-M3 via EmbeddingClient. The numbers
#     that belong in docs/benchmarks/.
#
# Callers: scripts/recall_benchmark.py (CLI), tests/test_recall_benchmark.py (CI gate).
# API: generate_corpus, build_chunk_records, run_benchmark, MockEmbed.

from __future__ import annotations

import hashlib
import logging
import random
import re
from typing import Any

from .chunker import Chunker
from .document import DocumentType, ParseResult

logger = logging.getLogger(__name__)

DIM = 1024
SEED = 20261025
DEFAULT_TOPICS = 55
DEFAULT_DISTRACTORS = 35
K_VALUES = (5, 10)

# Distinctive two-token vocabulary: every topic gets a unique pair that
# appears NOWHERE else in the corpus (high BM25 IDF, clean ground truth)
# plus a real-ish tech keyword for readability.
_TERM_POOL = [
    ("zorbek", "flume"),
    ("quantlyx", "drift"),
    ("verdan", "helio"),
    ("murex", "pyre"),
    ("caldera", "span"),
    ("obsidian", "reach"),
    ("fenwick", "vault"),
    ("solace", "grid"),
    ("tessera", "loom"),
    ("pyrrhic", "forge"),
    ("nimbus", "crag"),
    ("eldrin", "wash"),
    ("vantage", "spool"),
    ("cobalt", "sheen"),
    ("myriad", "plinth"),
    ("oracle", "tide"),
    ("brindle", "cadence"),
    ("lacuna", "spire"),
    ("thorn", "aperture"),
    ("vesper", "cascade"),
    ("ardent", "mill"),
    ("gossamer", "warp"),
    ("invective", "tarn"),
    ("paladin", "ridge"),
    ("quietus", "bloom"),
    ("sable", "kern"),
    ("tidal", "gambit"),
    ("umbra", "lintel"),
    ("verdant", "plait"),
    ("wraith", "cadence"),
    ("xanthic", "veil"),
    ("yarrow", "plume"),
    ("zephyr", "strut"),
    ("abacus", "gleam"),
    ("brevity", "solder"),
    ("cistern", "plunge"),
    ("dovetail", "murmur"),
    ("effigy", "ratchet"),
    ("fulcrum", "baste"),
    ("galleon", "twill"),
    ("hallow", "spindle"),
    ("ingot", "vessel"),
    ("jasper", "quiet"),
    ("keystone", "fray"),
    ("lattice", "harrow"),
    ("mantle", "grist"),
    ("nocturne", "shale"),
    ("onyx", "trellis"),
    ("parable", "winch"),
    ("quorum", "slate"),
    ("ripple", "gauge"),
    ("sonata", "bracket"),
    ("tundra", "clasp"),
    ("ultra", "moraine"),
    ("vellum", "crank"),
]

_TOPIC_WORDS = [
    "pipeline",
    "scheduler",
    "replicator",
    "aggregator",
    "validator",
    "serializer",
    "dispatcher",
    "collector",
    "resolver",
    "allocator",
    "sequencer",
    "compressor",
]

_GENERIC_SENTENCES = [
    "The system processes incoming data records and stores the result in the backend service layer.",
    "A background model reads the queue and updates internal state for each processed item.",
    "Configuration is loaded at startup and applied to every component in the service chain.",
    "The service exposes an HTTP endpoint that returns the current status of the data pipeline.",
    "Logging captures each request with a correlation id for later analysis by operators.",
    "Metrics are emitted periodically so the dashboard can chart system throughput over time.",
    "Errors are retried with exponential backoff until the deadline expires and the item is parked.",
    "The component validates its input schema before any storage write is attempted.",
    "A periodic job compacts the storage layer and removes entries older than the retention window.",
    "Handlers are registered at init and dispatched by name when a matching event arrives.",
]

_MD_SECTIONS = ["Overview", "Architecture", "Quick Start", "Configuration", "Monitoring", "Troubleshooting"]

_CODE_HEADER = "import logging\n\nlogger = logging.getLogger(__name__)\n\n\n"


def _tokens(text: str) -> list[str]:
    return [w.lower() for w in re.findall(r"[a-zA-Z0-9]+", text) if len(w) > 1]


def _hashed_vec(text: str) -> list[float]:
    vec = [0.0] * DIM
    for tok in _tokens(text):
        h = int(hashlib.blake2b(tok.encode(), digest_size=4).hexdigest(), 16)
        vec[h % DIM] += 1.0
    return vec


class MockEmbed:
    # Deterministic hashed-bag stand-in for EmbeddingClient. Wiring-regression
    # evidence only — NOT quality evidence (module docstring).
    base_url = "http://127.0.0.1:11432/v1"
    api_key = ""

    async def embed(self, text: str) -> list[float]:
        return _hashed_vec(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [_hashed_vec(t) for t in texts]

    async def health(self) -> bool:
        return True


def _md_doc(rng: random.Random, doc_id: str, title: str, terms: tuple[str, str, str]) -> str:
    t1, t2, word = terms
    lines = [f"# {title}", ""]
    for section in _MD_SECTIONS:
        body = [
            f"The {t1} {word} coordinates {t2} events across every stage of this {section.lower()} path.",
            f"Operators tune the {t2} ratio per tenant and the {t1} module reports the {word} result back.",
            f"Each {t1} stage emits a {t2} checkpoint so the {word} flow can resume after a restart.",
            f"The {section.lower()} contract for {t1} lists every {t2} field the {word} layer requires.",
        ]
        lines.append(f"## {section}")
        lines.append(" ".join(body))
        lines.append("")
    return "\n".join(lines)


def _code_doc(rng: random.Random, doc_id: str, title: str, terms: tuple[str, str, str]) -> str:
    t1, t2, word = terms
    parts = [f"# {title} — {t1} {t2} {word} module", _CODE_HEADER]
    funcs = [
        (f"process_{t1}", f"Route one {t2} {word} event through the {t1} handler chain and return the outcome."),
        (f"validate_{t2}", f"Check that a {t1} {word} payload carries the required {t2} fields before storage."),
        (f"snapshot_{word}_state", f"Persist the current {t1} {t2} counters so a restart can resume the {word} loop."),
        (f"resolve_{t1}_backlog", f"Drain queued {t2} items older than the deadline using the {t1} retry policy."),
    ]
    for fname, desc in funcs:
        filler = rng.choice(_GENERIC_SENTENCES)
        parts.append(
            f"def {fname}(payload):\n"
            f"    # {desc}\n"
            f"    # {filler}\n"
            f"    if not payload:\n"
            f"        return None\n"
            f'    logger.info("{fname} handling %s", payload.get("id"))\n'
            f'    return {{"ok": True, "module": "{t1}"}}\n\n\n'
        )
    return "\n".join(parts)


def _distractor_doc(rng: random.Random, doc_id: str, idx: int) -> str:
    lines = [f"# Internal Service Note {idx}", ""]
    for section in _MD_SECTIONS[:5]:
        body = [rng.choice(_GENERIC_SENTENCES) for _ in range(4)]
        lines.append(f"## {section}")
        lines.append(" ".join(body))
        lines.append("")
    return "\n".join(lines)


def generate_corpus(
    seed: int = SEED, topics: int = DEFAULT_TOPICS, distractors: int = DEFAULT_DISTRACTORS
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    # Returns (docs, queries): docs = [{doc_id, text, doc_type}], queries =
    # [(query_text, relevant_doc_id)]. Deterministic for a given seed.
    rng = random.Random(seed)
    docs: list[dict[str, Any]] = []
    queries: list[tuple[str, str]] = []
    n_code = topics // 4
    for i in range(topics):
        t1, t2 = _TERM_POOL[i % len(_TERM_POOL)]
        word = _TOPIC_WORDS[i % len(_TOPIC_WORDS)]
        # unique qualifier so no two topics share a term pair
        t1q = f"{t1}{i:02d}" if i >= len(_TERM_POOL) else t1
        terms = (t1q, t2, word)
        doc_id = f"topic_{i:03d}"
        title = f"{t1q.title()} {t2.title()} {word.title()}"
        if i < n_code:
            docs.append({"doc_id": doc_id, "text": _code_doc(rng, doc_id, title, terms), "doc_type": "py"})
        else:
            docs.append({"doc_id": doc_id, "text": _md_doc(rng, doc_id, title, terms), "doc_type": "md"})
        queries.append((f"{t1q} {t2} {word}", doc_id))
    for j in range(distractors):
        docs.append({"doc_id": f"distract_{j:03d}", "text": _distractor_doc(rng, f"distract_{j}", j), "doc_type": "md"})
    logger.info(
        "generated corpus: %d docs (%d code topics), %d queries, seed=%d", len(docs), n_code, len(queries), seed
    )
    return docs, queries


async def build_chunk_records(docs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    # Chunk every doc through the REAL chunker (semantic for markdown, code
    # auto-detect for python) so the benchmark exercises the production
    # structure-aware path. Returns (records, doc_to_chunks).
    records: list[dict[str, Any]] = []
    doc_to_chunks: dict[str, list[str]] = {}
    for d in docs:
        if d["doc_type"] == "py":
            dtype = DocumentType.CODE_PYTHON
            chunker = Chunker(strategy="semantic", chunk_size=512)
        else:
            dtype = DocumentType.MARKDOWN
            chunker = Chunker(strategy="semantic", chunk_size=512)
        result = ParseResult(
            file_path=f"/bench/{d['doc_id']}.{d['doc_type']}",
            file_name=f"{d['doc_id']}.{d['doc_type']}",
            doc_type=dtype,
            content=d["text"],
        )
        chunks = await chunker.chunk(result)
        ids = []
        for c in chunks:
            cid = f"{d['doc_id']}_{c.index}"
            ids.append(cid)
            records.append(
                {
                    "id": cid,
                    "vector": None,  # filled by the caller's embed mode
                    "text": c.text,
                    "doc_path": f"/bench/{d['doc_id']}.{d['doc_type']}",
                    "doc_name": f"{d['doc_id']}.{d['doc_type']}",
                    "doc_type": d["doc_type"],
                    "chunk_index": c.index,
                    "metadata": {"doc_id": d["doc_id"], "chunk_index": c.index},
                    "context": "",
                }
            )
        doc_to_chunks[d["doc_id"]] = ids
    logger.info("chunked %d docs into %d chunks", len(docs), len(records))
    return records, doc_to_chunks


def _recall_at_k(retrieved: list[str], truth: set[str], k: int) -> float:
    return sum(1 for rid in retrieved[:k] if rid in truth) / max(len(truth), 1)


def _mrr_at_k(retrieved: list[str], truth: set[str], k: int) -> float:
    for rank, rid in enumerate(retrieved[:k], start=1):
        if rid in truth:
            return 1.0 / rank
    return 0.0


def _empty_metrics(k_values: tuple[int, ...]) -> dict[str, Any]:
    return {f"recall@{k}": [] for k in k_values} | {"mrr@10": []}


def _finalize(mode_name: str, per_query: dict[str, list[float]]) -> dict[str, Any]:
    out: dict[str, Any] = {"mode": mode_name}
    for metric, vals in per_query.items():
        out[metric] = round(sum(vals) / len(vals), 4) if vals else 0.0
    return out


async def run_benchmark(
    embed: Any,
    docs: list[dict[str, Any]],
    queries: list[tuple[str, str]],
    *,
    workdir: str,
    k_values: tuple[int, ...] = K_VALUES,
) -> dict[str, Any]:
    # Runs BM25-only, vector-only, and RRF fusion over the same corpus and
    # returns per-mode recall@k + MRR plus the ablation verdict
    # (rrf >= max(bm25, vector) on recall@10).
    import os

    from ..store.vector_store import VectorStore
    from .bm25_index import BM25Index
    from .reranker import HybridSearch

    records, doc_to_chunks = await build_chunk_records(docs)
    vectors = await embed.embed_batch([r["text"] for r in records])
    for r, v in zip(records, vectors):
        r["vector"] = v

    vpath = os.path.join(workdir, "vectors")
    vs = VectorStore(vector_path=vpath, dimension=len(vectors[0]), backend_type="local")
    vs.add_batch(records)
    bm25 = BM25Index(store_path=os.path.join(workdir, "bm25.db"))
    bm25.add_documents(records)
    hs = HybridSearch(vs, method="rrf")

    bm25_m = _empty_metrics(k_values)
    vec_m = _empty_metrics(k_values)
    rrf_m = _empty_metrics(k_values)
    for query, doc_id in queries:
        truth = set(doc_to_chunks.get(doc_id, []))
        if not truth:
            continue
        bm25_rows = bm25.search(query, top_k=10)
        bm25_ids = [r["id"] for r in bm25_rows]
        qv = await embed.embed(query)
        vec_rows = vs.search(qv, top_k=10, threshold=0.0)
        vec_ids = [r.get("id", "") for r in vec_rows]
        rrf_rows = await hs.search(qv, query, top_k=10, threshold=0.0)
        rrf_ids = [r.get("id", "") for r in rrf_rows]
        for k in k_values:
            bm25_m[f"recall@{k}"].append(_recall_at_k(bm25_ids, truth, k))
            vec_m[f"recall@{k}"].append(_recall_at_k(vec_ids, truth, k))
            rrf_m[f"recall@{k}"].append(_recall_at_k(rrf_ids, truth, k))
        bm25_m["mrr@10"].append(_mrr_at_k(bm25_ids, truth, 10))
        vec_m["mrr@10"].append(_mrr_at_k(vec_ids, truth, 10))
        rrf_m["mrr@10"].append(_mrr_at_k(rrf_ids, truth, 10))

    vs.close()

    metrics = {
        "bm25_only": _finalize("bm25_only", bm25_m),
        "vector_only": _finalize("vector_only", vec_m),
        "rrf_fusion": _finalize("rrf_fusion", rrf_m),
    }
    best_single = max(metrics["bm25_only"]["recall@10"], metrics["vector_only"]["recall@10"])
    metrics["ablation"] = {
        "best_single_mode_recall@10": round(best_single, 4),
        "rrf_vs_best_single": round(metrics["rrf_fusion"]["recall@10"] - best_single, 4),
        "rrf_geq_max_single": metrics["rrf_fusion"]["recall@10"] >= best_single,
    }
    metrics["corpus"] = {"chunks": len(records), "docs": len(docs), "queries": len(queries)}
    logger.info(
        "benchmark done: chunks=%d queries=%d | bm25@10=%.3f vector@10=%.3f rrf@10=%.3f | rrf>=max(single): %s",
        len(records),
        len(queries),
        metrics["bm25_only"]["recall@10"],
        metrics["vector_only"]["recall@10"],
        metrics["rrf_fusion"]["recall@10"],
        metrics["ablation"]["rrf_geq_max_single"],
    )
    return metrics
