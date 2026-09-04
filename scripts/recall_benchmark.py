# Issue #70 acceptance 3 — Recall@k delta: vector-only vs hybrid vs
# hybrid+cross-encoder-rerank.
#
# Runs OFFLINE: embed is a deterministic hashed-bag-of-tokens vector (no live
# fusion-mlx), and the cross-encoder is a mocked token-overlap scorer (stands
# in for bge-reranker-v2-m3).
#
# What this script VALIDATES (and why):
#   - The three retrieval stages run end-to-end offline (vector → hybrid →
#     hybrid+rerank) through the real VectorStore / HybridSearch /
#     CrossEncoderReranker code paths.
#   - No recall REGRESSION across stages: rerank >= hybrid >= vector. Rerank
#     re-scores a wider candidate pool then truncates to top_k, so it must
#     never drop recall below hybrid; hybrid adds a BM25 keyword channel, so
#     it must never drop below vector-only. This is the contract the acceptance
#     cares about for a regression guard.
#
# What it CANNOT show (and says so in output):
#   - The MAGNITUDE of the recall lift. A real cross-encoder (bge-reranker-v2-m3)
#     scores query-document RELEVANCE via cross-attention — it demotes docs
#     that share surface terms but are not about the query. A hashed-bag mock
#     has no semantics, and a Jaccard-overlap rerank mock carries the same
#     signal as BM25, so it cannot model that precision lift. A real BGE-M3
#     dense vector also has a semantic gap (weak on exact keywords / code
#     identifiers) that the BM25 channel fills — the keyword-based mock dense
#     vector does not reproduce that gap. Fabricating a fake lift here would
#     be a test that passes for the wrong reason (Rule 9). For the absolute
#     recall delta, run this against live fusion-mlx with BGE-M3 +
#     bge-reranker-v2-m3 loaded (see README §Rerank).
#
# Cleans up its temp store dir after the run (global rule: keep only output +
# logs).
#
# Callers: manual `python scripts/recall_benchmark.py`, CI.
# API: VectorStore, BM25Index, HybridSearch, CrossEncoderReranker.

import asyncio
import hashlib
import logging
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger("recall_bench")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

DIM = 1024


def _tokens(text):
    import re

    return [w.lower() for w in re.findall(r"[a-zA-Z0-9]+", text) if len(w) > 1]


def _hashed_vec(text):
    # Deterministic 1024-d bag-of-tokens vector. Each token hashes to a bucket
    # and adds 1.0; semantically-overlapping texts share buckets → cosine
    # proximity. Imperfect by design (hash collisions, no semantics) so dense
    # recall leaves headroom for hybrid + rerank to show a lift.
    vec = [0.0] * DIM
    for tok in _tokens(text):
        h = int(hashlib.blake2b(tok.encode(), digest_size=4).hexdigest(), 16)
        vec[h % DIM] += 1.0
    return vec


class _MockEmbed:
    # Drop-in for EmbeddingClient: embed(text) and embed_batch(texts) return
    # the hashed-bag vector. base_url/api_key kept for _do_rerank compat.
    base_url = "http://127.0.0.1:11432/v1"
    api_key = ""

    async def embed(self, text):
        return _hashed_vec(text)

    async def embed_batch(self, texts):
        return [_hashed_vec(t) for t in texts]

    async def health(self):
        return True


# Synthetic corpus: code + prose + keyword-heavy. Relevant docs carry the
# query's DISTINCTIVE terms; distractors are full of GENERIC high-frequency
# terms (system, model, data, process, service, result) that share many
# hashed-vector buckets with any query, so dense cosine ranks them high
# (false friends), but BM25 IDF downweights generic terms and the cross-encoder
# Jaccard over distinctive terms ranks them low. That leaves headroom: dense
# retrieval alone misses some relevant docs buried under distractors; hybrid
# + rerank lifts them.
CORPUS = [
    # topic: hybrid retrieval
    ("doc-vector-bm25",
     "Hybrid retrieval combines BM25 keyword matching with dense vector similarity for better recall"),
    ("doc-vector-rrf",
     "Reciprocal rank fusion RRF merges ranked lists from multiple retrieval channels into one score"),
    ("doc-bm25-okapi",
     "Okapi BM25 scores documents by term frequency inverse document frequency with length normalization"),
    # topic: rerank
    ("doc-rerank-cross",
     "Cross-encoder reranker bge-reranker-v2-m3 re-scores top candidates by query-document relevance"),
    ("doc-rerank-llm",
     "LLM prompt reranker asks the model to rate each document relevance on a scale then sorts"),
    # topic: tenant isolation
    ("doc-qdrant-tenant",
     "Qdrant vector backend isolates collections per tenant so cross-tenant queries see no foreign vectors"),
    ("doc-identity-jwt",
     "fusion-identity verifies the JWT and returns the authoritative tenant id tid claim for scoping"),
    # topic: code
    ("doc-py-ast",
     "Python AST chunker splits source code on function and class boundaries for code-aware indexing"),
    ("doc-py-lint",
     "Ruff linter checks Python code style unused imports and complexity with fast static analysis"),
    # topic: auth
    ("doc-auth-login",
     "def login(user, password): verify credentials and issue a session token for authentication"),
    ("doc-auth-token",
     "Token refresh endpoint: rotate the JWT without re-entering the password on the auth service"),
    # topic: embed
    ("doc-embed-bge",
     "BGE-M3 embedding model produces 1024 dimensional dense vectors served by fusion-mlx over HTTP"),
    # ── distractors: generic boilerplate, high bucket overlap, low IDF ──
    ("dist-system-1",
     "The system processes data using a model and returns the result to the service for storage"),
    ("dist-system-2",
     "A service model handles data processing and returns results stored by the system backend"),
    ("dist-system-3",
     "System backend stores the result returned by the data processing model service component"),
    ("dist-generic-1",
     "This component uses a model to process input data and produce an output result for the caller"),
    ("dist-generic-2",
     "The model reads data from the service and writes a result back to the system storage layer"),
    ("dist-generic-3",
     "Processing data with a model returns a result that the system service stores for later use"),
    ("dist-filler-1",
     "Data model service system result process return store component backend output input caller"),
    ("dist-filler-2",
     "Service system model data result process return store component backend layer handle request"),
]

# Ground truth: query -> set of relevant doc ids (the docs whose topic matches).
# Queries use the DISTINCTIVE topic terms so BM25 + cross-encoder surface them
# over the generic distractors.
QUERIES = [
    ("hybrid BM25 vector retrieval RRF fusion", {"doc-vector-bm25", "doc-vector-rrf", "doc-bm25-okapi"}),
    ("cross-encoder rerank bge-reranker-v2-m3 relevance", {"doc-rerank-cross", "doc-rerank-llm"}),
    ("qdrant per-tenant collection isolation JWT tid", {"doc-qdrant-tenant", "doc-identity-jwt"}),
    ("python AST chunker function class boundaries ruff lint", {"doc-py-ast", "doc-py-lint"}),
    ("authentication login JWT token refresh credentials", {"doc-auth-login", "doc-auth-token", "doc-identity-jwt"}),
]


def _build_store(tmpdir):
    from fusion_rag.store.vector_store import VectorStore

    vpath = os.path.join(tmpdir, "vectors")
    vs = VectorStore(vector_path=vpath, dimension=DIM, backend_type="local")
    records = []
    for did, text in CORPUS:
        records.append({
            "id": did,
            "vector": _hashed_vec(text),
            "text": text,
            "doc_path": f"/corp/{did}.txt",
            "doc_name": f"{did}.txt",
            "doc_type": "txt",
            "chunk_index": 0,
            "metadata": {},
            "context": "",
        })
    vs.add_batch(records)
    logger.info("indexed %d docs into VectorStore (local backend)", len(records))
    return vs


def _recall_at_k(retrieved_ids, truth, k):
    top = retrieved_ids[:k]
    hits = sum(1 for rid in top if rid in truth)
    return hits / max(len(truth), 1)


async def _bench_vector_only(vs, embed, k=5):
    scores = []
    for query, truth in QUERIES:
        qv = await embed.embed(query)
        results = vs.search(qv, top_k=k, threshold=0.0)
        scores.append(_recall_at_k([r.get("id", "") for r in results], truth, k))
    return sum(scores) / len(scores)


async def _bench_hybrid(vs, embed, k=5):
    from fusion_rag.engine.reranker import HybridSearch

    hs = HybridSearch(vs, method="rrf")
    scores = []
    for query, truth in QUERIES:
        qv = await embed.embed(query)
        results = await hs.search(qv, query, top_k=k, threshold=0.0)
        scores.append(_recall_at_k([r.get("id", "") for r in results], truth, k))
    return sum(scores) / len(scores)


async def _bench_hybrid_rerank(vs, embed, k=5, pool_n=20):
    # Mocked cross-encoder: scores each candidate by token overlap with the
    # query (stand-in for bge-reranker-v2-m3's relevance scoring). Demonstrates
    # the rerank stage re-ordering a wider pool so the final top_k lifts recall
    # past what hybrid ranking alone produced.
    from fusion_rag.engine.cross_encoder_reranker import CrossEncoderReranker
    from fusion_rag.engine.reranker import HybridSearch

    hs = HybridSearch(vs, method="rrf")

    async def fake_call(payload, headers):
        qtoks = set(_tokens(payload["query"]))
        scored = []
        for idx, doc_text in enumerate(payload["documents"]):
            dtoks = set(_tokens(doc_text))
            overlap = len(qtoks & dtoks) / max(len(qtoks | dtoks), 1)
            scored.append({"index": idx, "relevance_score": overlap})
        scored.sort(key=lambda r: r["relevance_score"], reverse=True)
        return scored

    scores = []
    for query, truth in QUERIES:
        qv = await embed.embed(query)
        pool = await hs.search(qv, query, top_k=pool_n, threshold=0.0)
        reranker = CrossEncoderReranker(mlx_base_url="http://x/v1", model="mock")
        reranker._call_rerank = fake_call
        reranked = await reranker.rerank(query, pool, top_k=k)
        scores.append(_recall_at_k([r.get("id", "") for r in reranked], truth, k))
    return sum(scores) / len(scores)


async def run():
    logger.info("Fusion-RAG Recall Benchmark (issue #70 acceptance 3)")
    logger.info("=====================================================")
    logger.info("offline / mocked embed + mocked cross-encoder — NO-REGRESSION guard")
    logger.info("corpus=%d docs, queries=%d, k=5", len(CORPUS), len(QUERIES))
    logger.info("NOTE: measures the no-regression contract (rerank>=hybrid>=vector),")
    logger.info("      NOT the recall-lift magnitude — see module docstring + README.")
    tmpdir = tempfile.mkdtemp(prefix="recall_bench_")
    try:
        vs = _build_store(tmpdir)
        embed = _MockEmbed()
        vec_recall = await _bench_vector_only(vs, embed)
        hyb_recall = await _bench_hybrid(vs, embed)
        rr_recall = await _bench_hybrid_rerank(vs, embed)
        vs.close()

        logger.info("")
        logger.info("=== Recall@5 ===")
        logger.info("  vector-only            : %.3f", vec_recall)
        logger.info("  hybrid (RRF)           : %.3f  (delta vs vector: %+.3f)", hyb_recall, hyb_recall - vec_recall)
        logger.info("  hybrid + cross-rerank  : %.3f  (delta vs vector: %+.3f, vs hybrid: %+.3f)",
                    rr_recall, rr_recall - vec_recall, rr_recall - hyb_recall)

        # No-regression contract: each stage never drops recall below the prior.
        # A real cross-encoder LIFTS recall above hybrid (demoting same-surface
        # but off-topic docs); the mock cannot model that lift, so we assert the
        # weaker, honest invariant: the rerank stage does not LOSE recall vs
        # hybrid, and hybrid does not lose vs vector-only. A live-model run
        # (README) documents the positive lift.
        no_reg_hyb = hyb_recall >= vec_recall
        no_reg_rr = rr_recall >= hyb_recall
        logger.info("")
        logger.info("hybrid >= vector-only (no regression): %s", "PASS" if no_reg_hyb else "FAIL")
        logger.info("hybrid+rerank >= hybrid (no regression): %s", "PASS" if no_reg_rr else "FAIL")
        return {
            "vector_only": vec_recall,
            "hybrid": hyb_recall,
            "hybrid_rerank": rr_recall,
            "pass": no_reg_hyb and no_reg_rr,
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        logger.info("cleaned up temp store dir: %s", tmpdir)


if __name__ == "__main__":
    res = asyncio.run(run())
    sys.exit(0 if res["pass"] else 1)
