#!/usr/bin/env python3
# R7 (issue #129): PPR graph-layer ablation — vector-only vs vector+BFS-expand
# vs vector+PPR on (a) simple-fact queries and (b) a two-topic multi-hop
# subset, with the explicit HippoRAG-2 regression check: PPR must NOT drop
# below vector-only on the simple-fact subset. Graph is built deterministically
# from the corpus's own term vocabulary (entities = topic terms, relations =
# within-doc co-occurrence) so the ablation isolates the ranking layer, not
# extraction quality. Requires live BGE-M3 embeddings. Feature stays OFF by
# default in the service; this script is the gate for enabling it.

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fusion_rag.engine.graph_rag import GraphRAG
from fusion_rag.engine.recall_bench import (
    _TERM_POOL,
    _TOPIC_WORDS,
    DEFAULT_TOPICS,
    SEED,
    _recall_at_k,
    build_chunk_records,
    generate_corpus,
)
from fusion_rag.store.vector_store import VectorStore

RRF_K = 60


def _topic_terms(i: int) -> tuple[str, str, str]:
    t1, t2 = _TERM_POOL[i % len(_TERM_POOL)]
    t1q = f"{t1}{i:02d}" if i >= len(_TERM_POOL) else t1
    return t1q, t2, _TOPIC_WORDS[i % len(_TOPIC_WORDS)]


def _build_graph(docs_terms: dict[str, tuple[str, str, str]], chunks: list[dict], graph: GraphRAG) -> None:
    conn = graph._get_conn()
    for d in chunks:
        doc_id = d["doc_path"].rsplit("/", 1)[-1].rsplit(".", 1)[0]
        terms = docs_terms.get(doc_id)
        if not terms:
            continue
        for t in terms:
            conn.execute(
                "INSERT OR IGNORE INTO entities (name, type, chunk_id) VALUES (?, 'CONCEPT', ?)", (t, d["id"])
            )
        conn.executemany(
            "INSERT OR IGNORE INTO relations (source, target) VALUES (?, ?)",
            [(terms[0], terms[1]), (terms[1], terms[2])],
        )
    conn.commit()
    conn.close()


def _rrf_fuse(vector_rows: list[dict], graph_rows: list[dict], top_k: int = 10) -> list[str]:
    scores: dict[str, float] = {}
    for rank, r in enumerate(vector_rows):
        scores[r["id"]] = scores.get(r["id"], 0.0) + 1.0 / (RRF_K + rank + 1)
    for rank, g in enumerate(graph_rows):
        cid = g["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
    return [cid for cid, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]]


def _graph_hits(graph: GraphRAG, tokens: list[str], mode: str) -> list[dict]:
    seen: dict[str, dict] = {}
    for t in tokens:
        rows = graph.search(t) if mode == "bfs" else graph.search_ppr(t)
        for r in rows:
            if r["chunk_id"] not in seen:
                seen[r["chunk_id"]] = r
    return list(seen.values())


async def _main(out_path: str) -> int:
    from fusion_rag.embed.client import EmbeddingClient

    topics = DEFAULT_TOPICS
    docs, _queries = generate_corpus(seed=SEED, topics=topics)
    records, doc_to_chunks = await build_chunk_records(docs)
    docs_terms = {f"topic_{i:03d}": _topic_terms(i) for i in range(topics)}

    embed = EmbeddingClient()
    vectors = await embed.embed_batch([r["text"] for r in records])
    for r, v in zip(records, vectors):
        r["vector"] = v

    with tempfile.TemporaryDirectory(prefix="fusion-rag-ppr-ablation-") as workdir:
        vs = VectorStore(vector_path=os.path.join(workdir, "vectors"), dimension=len(vectors[0]), backend_type="local")
        vs.add_batch(records)
        graph = GraphRAG(db_path=os.path.join(workdir, "graph.db"))
        _build_graph(docs_terms, records, graph)

        # simple-fact subset: single-topic queries (all topics)
        simple = [(f"{t1} {t2} {w}", doc_id) for doc_id, (t1, t2, w) in list(docs_terms.items())]
        # multi-hop subset: query mixes terms from two topics; truth = both docs
        multihop = []
        for i in range(0, topics - 1, 4):
            j = i + 1
            t1i, t2i, _ = docs_terms[f"topic_{i:03d}"]
            _, _, wj = docs_terms[f"topic_{j:03d}"]
            multihop.append((f"{t1i} {t2i} {wj}", [f"topic_{i:03d}", f"topic_{j:03d}"]))

        results: dict[str, dict[str, dict[str, float]]] = {}
        for subset, qset, truth_of in (
            ("simple_fact", simple, lambda d: set(doc_to_chunks.get(d, []))),
            ("multihop", multihop, lambda d: set().union(*(doc_to_chunks.get(x, []) for x in d))),
        ):
            per_mode: dict[str, list[float]] = {"vector_only": [], "vector_bfs": [], "vector_ppr": []}
            for q, doc_key in qset:
                truth = truth_of(doc_key)
                if not truth:
                    continue
                qv = await embed.embed(q)
                vector_rows = vs.search(qv, top_k=10, threshold=0.0)
                per_mode["vector_only"].append(_recall_at_k([r["id"] for r in vector_rows], truth, 10))
                tokens = q.lower().split()
                for mode in ("bfs", "ppr"):
                    fused = _rrf_fuse(vector_rows, _graph_hits(graph, tokens, mode))
                    per_mode[f"vector_{mode}"].append(_recall_at_k(fused, truth, 10))
            results[subset] = {
                m: round(sum(v) / len(v), 4) if v else 0.0 for m, v in per_mode.items()
            }
        vs.close()

    summary = {
        "generated_at": os.environ.get("BENCH_DATE", "2026-11-14"),
        "queries": {"simple_fact": len(simple), "multihop": len(multihop)},
        "recall@10": results,
        "hipporag2_regression_check": {
            "subset": "simple_fact",
            "requirement": "vector_ppr >= vector_only",
            "pass": results["simple_fact"]["vector_ppr"] >= results["simple_fact"]["vector_only"],
        },
        "note": (
            "graph built deterministically from the corpus term vocabulary "
            "(entities = topic terms, relations = within-doc co-occurrence); "
            "the synthetic corpus has no cross-doc entity chains, so PPR's "
            "multi-hop walk advantage cannot express itself here — treat as a "
            "regression gate, not a quality measurement. Real multi-hop corpus "
            "required before enabling."
        ),
    }
    print(json.dumps(summary, indent=2))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(f"\nwritten to {out_path}", file=sys.stderr)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="PPR graph-layer ablation (issue #129)")
    parser.add_argument(
        "--out",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "docs",
            "benchmarks",
            "ppr-ablation.json",
        ),
    )
    args = parser.parse_args()
    return asyncio.run(_main(args.out))


if __name__ == "__main__":
    sys.exit(main())
