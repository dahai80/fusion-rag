#!/usr/bin/env python3
# R8 (issue #130): chunk-size ablation over the recall bench — semantic chunker
# at sizes 256/384/512/768/1024 (overlap = size/8) x BM25 + vector (BGE-M3)
# recall@10. Output feeds the CHUNK_PROFILES defaults so they come from bench
# data, not constants. Requires live BGE-M3.

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fusion_rag.engine.bm25_index import BM25Index
from fusion_rag.engine.chunker import Chunker
from fusion_rag.engine.document import DocumentType, ParseResult
from fusion_rag.engine.recall_bench import (
    DEFAULT_TOPICS,
    SEED,
    _map_at_k,
    _mrr_at_k,
    _recall_at_k,
    generate_corpus,
)
from fusion_rag.store.vector_store import VectorStore


async def _run_size(size: int, docs: list[dict], queries: list[tuple[str, str]], embed, workdir: str) -> dict:
    from fusion_rag.engine.recall_bench import build_chunk_records

    _records, doc_to_chunks = await build_chunk_records(docs)
    # Re-chunk with the ablation size/overlap (build_chunk_records hardcodes 512).
    chunked: list[dict] = []
    for d in docs:
        dtype = DocumentType.CODE_PYTHON if d["doc_type"] == "py" else DocumentType.MARKDOWN
        chunker = Chunker(strategy="semantic", chunk_size=size, chunk_overlap=size // 8)
        result = ParseResult(
            file_path=f"/bench/{d['doc_id']}.{d['doc_type']}",
            file_name=f"{d['doc_id']}.{d['doc_type']}",
            doc_type=dtype,
            content=d["text"],
        )
        chunks = await chunker.chunk(result)
        for c in chunks:
            chunked.append(
                {
                    "id": f"{d['doc_id']}_{c.index}",
                    "vector": None,
                    "text": c.text,
                    "doc_path": f"/bench/{d['doc_id']}.{d['doc_type']}",
                    "doc_name": f"{d['doc_id']}.{d['doc_type']}",
                    "doc_type": d["doc_type"],
                    "chunk_index": c.index,
                    "metadata": {"doc_id": d["doc_id"], "chunk_index": c.index},
                    "context": "",
                }
            )

    vectors = await embed.embed_batch([r["text"] for r in chunked])
    for r, v in zip(chunked, vectors):
        r["vector"] = v

    vdir = os.path.join(workdir, f"vectors-{size}")
    vs = VectorStore(vector_path=vdir, dimension=len(vectors[0]), backend_type="local")
    vs.add_batch(chunked)
    bm25 = BM25Index(store_path=os.path.join(workdir, f"bm25-{size}.db"))
    bm25.add_documents(chunked)

    metrics = {"n_chunks": len(chunked)}
    for name, k in (("vector", 10), ("bm25", 10)):
        rows = {"recall@10": [], "mrr@10": [], "map@10": []}
        for q, doc_id in queries:
            truth = set(doc_to_chunks.get(doc_id, []))
            if not truth:
                continue
            if name == "vector":
                qv = await embed.embed(q)
                hits = [r["id"] for r in vs.search(qv, top_k=10, threshold=0.0)]
            else:
                hits = [r["id"] for r in bm25.search(q, top_k=10)]
            rows["recall@10"].append(_recall_at_k(hits, truth, 10))
            rows["mrr@10"].append(_mrr_at_k(hits, truth, 10))
            rows["map@10"].append(_map_at_k(hits, truth, 10))
        metrics[name] = {k2: round(sum(v) / len(v), 4) if v else 0.0 for k2, v in rows.items()}
    bm25._close_conn()
    vs.close()
    return metrics


async def _main(out_path: str) -> int:
    from fusion_rag.embed.client import EmbeddingClient

    docs, queries = generate_corpus(seed=SEED, topics=DEFAULT_TOPICS)
    embed = EmbeddingClient()
    out = {}
    with tempfile.TemporaryDirectory(prefix="fusion-rag-chunk-ablation-") as workdir:
        for size in (256, 384, 512, 768, 1024):
            out[str(size)] = await _run_size(size, docs, queries, embed, workdir)
            print(f"size {size}: {json.dumps(out[str(size)])}", file=sys.stderr)
    payload = {"generated_at": os.environ.get("BENCH_DATE", "2026-11-14"), "overlap_ratio": 0.125, "sizes": out}
    print(json.dumps(payload, indent=2))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"\nwritten to {out_path}", file=sys.stderr)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="chunk-size ablation (issue #130)")
    parser.add_argument(
        "--out",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "docs",
            "benchmarks",
            "chunk-size-ablation.json",
        ),
    )
    args = parser.parse_args()
    import asyncio

    return asyncio.run(_main(args.out))


if __name__ == "__main__":
    sys.exit(main())
