"""Issue #132 (R10): BGE-M3 native sparse channel ablation — REAL mode.

Arms over the deterministic recall_bench corpus (55 queries, live fusion-mlx
BGE-M3 with the sparse head, PR fusion-mlx#1627):
  bm25_only | vector_only | rrf2 (dense+bm25, current default) |
  rrf3 (dense+bm25+sparse — query sparse weights dot-producted against
  ingest-time chunk sparse weights over the candidate pool, RRF third channel)

Run: FUSION_MLX_URL=... python scripts/sparse_ablation.py
Output: docs/benchmarks/sparse-ablation.json
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("FUSION_RAG_SPARSE_CHANNEL", "1")

from fusion_rag.embed.client import EmbeddingClient
from fusion_rag.engine.bm25_index import BM25Index
from fusion_rag.engine.recall_bench import (
    _finalize,
    _map_at_k,
    _mrr_at_k,
    _ndcg_at_k,
    _recall_at_k,
    build_chunk_records,
    generate_corpus,
)
from fusion_rag.engine.reranker import HybridSearch
from fusion_rag.engine.runtime_config import get_runtime_config
from fusion_rag.store.vector_store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("sparse_ablation")

_WORK = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_WORK)
WORKDIR = os.path.join(_WORK, ".sparse_ablation_work")
OUT_PATH = os.path.join(_ROOT, "docs", "benchmarks", "sparse-ablation.json")
K_VALUES = (5, 10)


async def main() -> None:
    assert get_runtime_config().sparse_channel, "FUSION_RAG_SPARSE_CHANNEL=1 required"
    # FUSION_MLX_URL honored here (benchmarks bypass the gateway per the
    # gateway-cache gotcha); EmbeddingClient's default param is a literal.
    base_url = os.environ.get("FUSION_MLX_URL", "").strip() or "http://127.0.0.1:11434/v1"
    embed = EmbeddingClient(base_url=base_url)
    docs, queries = generate_corpus()
    records, doc_to_chunks = await build_chunk_records(docs)

    texts = [r["text"] for r in records]
    t0 = time.monotonic()
    vectors, sparse_list = await embed.embed_batch_sparse(texts)
    embed_ms = (time.monotonic() - t0) * 1000
    n_sparse = sum(1 for s in sparse_list if s)
    logger.info(
        "embedded %d chunks (dense+sparse in one pass): %.0f ms, %d with sparse weights",
        len(records), embed_ms, n_sparse,
    )
    if n_sparse == 0:
        logger.error("no sparse weights returned — model lacks the sparse head; ablation invalid")
        return
    for r, v in zip(records, vectors):
        r["vector"] = v

    vs = VectorStore(vector_path=os.path.join(WORKDIR, "vectors"), dimension=len(vectors[0]), backend_type="local")
    vs.add_batch(records)
    bm25 = BM25Index(store_path=os.path.join(WORKDIR, "bm25.db"))
    bm25.add_documents(records)
    hs = HybridSearch(vs, method="rrf")

    doc_weights = {r["id"]: s for r, s in zip(records, sparse_list) if s}

    def sparse_lookup(chunk_ids: list[str]) -> dict[str, dict]:
        return {c: doc_weights[c] for c in chunk_ids if c in doc_weights}

    arms = {}
    for name in ("bm25_only", "vector_only", "rrf2", "rrf3"):
        per: dict = {}
        for k in K_VALUES:
            per[f"recall@{k}"] = []
            per[f"ndcg@{k}"] = []
        per["mrr@10"] = []
        per["map@10"] = []
        arms[name] = per

    t0 = time.monotonic()
    for query, doc_id in queries:
        truth = set(doc_to_chunks.get(doc_id, []))
        if not truth:
            continue
        bm25_ids = [r["id"] for r in bm25.search(query, top_k=10)]
        qv = await embed.embed(query)
        vec_ids = [r.get("id", "") for r in vs.search(qv, top_k=10, threshold=0.0)]
        _, q_sparse = await embed.embed_batch_sparse([query])
        qs = q_sparse[0] if q_sparse else None
        rrf2_ids = [r.get("id", "") for r in await hs.search(qv, query, top_k=10, threshold=0.0)]
        rrf3_rows = await hs.search(
            qv, query, top_k=10, threshold=0.0, query_sparse=qs, sparse_lookup=sparse_lookup
        )
        rrf3_ids = [r.get("id", "") for r in rrf3_rows]
        for k in K_VALUES:
            arms["bm25_only"][f"recall@{k}"].append(_recall_at_k(bm25_ids, truth, k))
            arms["vector_only"][f"recall@{k}"].append(_recall_at_k(vec_ids, truth, k))
            arms["rrf2"][f"recall@{k}"].append(_recall_at_k(rrf2_ids, truth, k))
            arms["rrf3"][f"recall@{k}"].append(_recall_at_k(rrf3_ids, truth, k))
            for name, ids in (
                ("bm25_only", bm25_ids), ("vector_only", vec_ids),
                ("rrf2", rrf2_ids), ("rrf3", rrf3_ids),
            ):
                arms[name][f"ndcg@{k}"].append(_ndcg_at_k(ids, truth, k))
        for name, ids in (
            ("bm25_only", bm25_ids), ("vector_only", vec_ids),
            ("rrf2", rrf2_ids), ("rrf3", rrf3_ids),
        ):
            arms[name]["mrr@10"].append(_mrr_at_k(ids, truth, 10))
            arms[name]["map@10"].append(_map_at_k(ids, truth, 10))
    query_ms = (time.monotonic() - t0) * 1000 / max(len(queries), 1)

    vs.close()
    metrics = {name: _finalize(name, per) for name, per in arms.items()}
    metrics["ablation"] = {
        "rrf3_vs_rrf2_recall@10": round(metrics["rrf3"]["recall@10"] - metrics["rrf2"]["recall@10"], 4),
        "rrf3_geq_rrf2": metrics["rrf3"]["recall@10"] >= metrics["rrf2"]["recall@10"],
        "rrf3_vs_rrf2_ndcg@10": round(metrics["rrf3"]["ndcg@10"] - metrics["rrf2"]["ndcg@10"], 4),
        "sparse_rows": n_sparse,
        "chunks": len(records),
        "queries": len(queries),
        "avg_query_ms_all_arms": round(query_ms, 1),
        "note": (
            "single combined dense+sparse embed pass at ingest; "
            "query adds one sparse call when the flag is on"
        ),
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    r2, r3 = metrics["rrf2"]["recall@10"], metrics["rrf3"]["recall@10"]
    logger.info("rrf2 recall@10=%.4f rrf3 recall@10=%.4f -> %s", r2, r3, OUT_PATH)
    for name in ("bm25_only", "vector_only", "rrf2", "rrf3"):
        m = metrics[name]
        logger.info(
            "%-12s recall@5=%.4f recall@10=%.4f ndcg@10=%.4f mrr@10=%.4f",
            name, m["recall@5"], m["recall@10"], m["ndcg@10"], m["mrr@10"],
        )


if __name__ == "__main__":
    asyncio.run(main())
