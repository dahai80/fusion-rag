#!/usr/bin/env python3
# Issue #131 (R9): table-PDF ablation — legacy PyMuPDF extraction vs Docling
# layout-aware parsing. Generates deterministic table PDFs with cells inserted
# COLUMN-MAJOR (the layout a naive text extractor reads out of row order), then
# measures BM25 + vector recall@10 on row-level questions for both backends.
# Truth = the chunk containing both the item name and the expected value.
# Requires live BGE-M3 + docling (models auto-download on first run).

from __future__ import annotations

import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fitz

from fusion_rag.engine.bm25_index import BM25Index
from fusion_rag.engine.chunker import Chunker
from fusion_rag.engine.document import DocumentType, ParseResult
from fusion_rag.engine.recall_bench import _recall_at_k

ITEMS = ["ledger", "beacon", "canyon", "harbor", "meadow", "quartz", "summit", "tundra"]
REGIONS = ["north", "south", "east", "west"]
QUARTERS = ["q1", "q2", "q3", "q4"]
SEED = 131


def _build_pdf(path: str, doc_idx: int, rows: list[tuple[str, str, str, int]]) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 60), f"Operations report {doc_idx:03d}")
    # COLUMN-MAJOR insertion: per-cell text blocks stored column by column, so
    # naive text extraction reads entire columns back-to-back and loses row
    # alignment. Docling's layout model rebuilds the table rows.
    x_cols = [72, 200, 300, 400]
    y0 = 90
    row_h = 16
    # Ruled grid: without visible table borders TableFormer does not segment
    # the region as a table at all (probed — both backends then scramble).
    table_h = len(rows) * row_h
    x_end = 480
    y_end = y0 + table_h
    for x in x_cols + [x_end]:
        page.draw_line((x, y0 - 4), (x, y_end))
    for ri in range(len(rows) + 1):
        page.draw_line((72, y0 - 4 + ri * row_h), (x_end, y0 - 4 + ri * row_h))
    for ci in range(4):
        for ri, (item, region, quarter, value) in enumerate(rows):
            cell = [item, region, quarter, str(value)][ci]
            page.insert_text((x_cols[ci], y0 + ri * 16), cell)
    doc.save(path)
    doc.close()


def _gen_corpus(dirpath: str, n_docs: int, rows_per_doc: int):
    import random

    rng = random.Random(SEED)
    docs = []
    queries = []
    for d in range(n_docs):
        rows = []
        for _ in range(rows_per_doc):
            item = rng.choice(ITEMS)
            region = rng.choice(REGIONS)
            quarter = rng.choice(QUARTERS)
            value = rng.randint(100, 999)
            rows.append((item, region, quarter, value))
        path = os.path.join(dirpath, f"report-{d:03d}.pdf")
        _build_pdf(path, d, rows)
        docs.append({"doc_id": f"report-{d:03d}", "path": path, "rows": rows})
        for item, region, quarter, value in rows[:2]:
            queries.append(
                (
                    f"{item} {region} {quarter} revenue {value}",
                    f"report-{d:03d}",
                    f"{item} {value}",
                )
            )
    return docs, queries


def _contains(truth_pair: tuple[str, str], text: str) -> bool:
    return truth_pair[0] in text and truth_pair[1] in text


async def _run_backend(backend: str, docs, queries, embed, workdir: str) -> dict:
    from fusion_rag.engine.docling_parser import DoclingParser

    chunked = []
    for d in docs:
        if backend == "docling":
            parser = DoclingParser()
            content, _pages, spans = await parser.parse_pdf(d["path"])
        else:
            parser = DocumentParserLegacy()
            content, _pages, spans = parser(d["path"])
        # FIXED strategy: the semantic chunker only splits at markdown
        # headings, so a borderless table exports as ONE section and never
        # splits — fixed-stride chunks give both backends the same row-level
        # discrimination budget.
        chunker = Chunker(strategy="fixed", chunk_size=256, chunk_overlap=32)
        result = ParseResult(
            file_path=d["path"],
            file_name=os.path.basename(d["path"]),
            doc_type=DocumentType.PDF,
            content=content,
            page_spans=spans,
        )
        chunks = await chunker.chunk(result)
        for c in chunks:
            chunked.append(
                {
                    "id": f"{d['doc_id']}_{c.index}",
                    "vector": None,
                    "text": c.text,
                    "doc_path": d["path"],
                    "doc_name": os.path.basename(d["path"]),
                    "doc_type": "pdf",
                    "chunk_index": c.index,
                    "metadata": {"doc_id": d["doc_id"], "chunk_index": c.index, "page": c.page},
                }
            )

    vectors = await embed.embed_batch([r["text"] for r in chunked])
    for r, v in zip(chunked, vectors):
        r["vector"] = v

    from fusion_rag.store.vector_store import VectorStore

    vdir = os.path.join(workdir, f"vectors-{backend}")
    # snapshot before add_batch: LocalBackend pops "metadata" into "metadata_json".
    doc_of = {r["id"]: r["metadata"]["doc_id"] for r in chunked}
    text_of = {r["id"]: r["text"] for r in chunked}
    vs = VectorStore(vector_path=vdir, dimension=len(vectors[0]), backend_type="local")
    vs.add_batch(chunked)
    bm25 = BM25Index(store_path=os.path.join(workdir, f"bm25-{backend}.db"))
    bm25.add_documents(chunked)

    recall = {"bm25": [], "vector": []}
    lat = {"bm25": [], "vector": []}
    n_rows_intact = 0
    n_evaluable = 0
    for q, doc_id, _pair in queries:
        t0 = time.perf_counter()
        bm25_hits = [r["id"] for r in bm25.search(q, top_k=10)]
        lat["bm25"].append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        qv = await embed.embed(q)
        vec_hits = [r["id"] for r in vs.search(qv, top_k=10, threshold=0.0)]
        lat["vector"].append((time.perf_counter() - t0) * 1000)
        # truth at chunk level: any chunk of the right doc containing the
        # item name AND the expected value (row-level discrimination).
        item, value = _pair.split(" ", 1)
        truth = {
            cid
            for cid, txt in text_of.items()
            if doc_of[cid] == doc_id and item in txt and value in txt
        }
        if truth:
            n_rows_intact += 1
            n_evaluable += 1
            recall["bm25"].append(_recall_at_k(bm25_hits, truth, 10))
            recall["vector"].append(_recall_at_k(vec_hits, truth, 10))
    bm25._close_conn()
    vs.close()
    return {
        "n_chunks": len(chunked),
        "rows_intact_ratio": round(n_rows_intact / len(queries), 4),
        "n_evaluable": n_evaluable,
        "recall@10_bm25": round(sum(recall["bm25"]) / len(recall["bm25"]), 4) if recall["bm25"] else 0.0,
        "recall@10_vector": round(sum(recall["vector"]) / len(recall["vector"]), 4) if recall["vector"] else 0.0,
        "latency_bm25_p50_ms": round(sorted(lat["bm25"])[len(lat["bm25"]) // 2], 1),
        "latency_vector_p50_ms": round(sorted(lat["vector"])[len(lat["vector"]) // 2], 1),
    }


def DocumentParserLegacy():
    def _parse(path: str):
        doc = fitz.open(path)
        content = ""
        spans = []
        pages = len(doc)
        for i, page in enumerate(doc):
            text = page.get_text()
            if not text:
                continue
            if content:
                content += "\n\n"
            start = len(content)
            content += text
            spans.append((start, len(content), i + 1))
        doc.close()
        return content, pages, spans

    return _parse


async def _main(n_docs: int, rows_per_doc: int, out_path: str) -> int:
    import time

    from fusion_rag.embed.client import EmbeddingClient

    embed = EmbeddingClient()
    out = {}
    with tempfile.TemporaryDirectory(prefix="fusion-rag-docling-") as workdir:
        docs, queries = _gen_corpus(workdir, n_docs, rows_per_doc)
        for backend in ("legacy", "docling"):
            t0 = time.perf_counter()
            out[backend] = await _run_backend(backend, docs, queries, embed, workdir)
            out[backend]["parse_ingest_seconds"] = round(time.perf_counter() - t0, 1)
            print(f"{backend}: {json.dumps(out[backend])}", file=sys.stderr)
    payload = {
        "generated_at": os.environ.get("BENCH_DATE", "2026-11-14"),
        "n_docs": n_docs,
        "rows_per_doc": rows_per_doc,
        "n_queries": len(queries),
        "chunk": "fixed 256/32",
        "backends": out,
    }
    print(json.dumps(payload, indent=2))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"\nwritten to {out_path}", file=sys.stderr)
    return 0


def main() -> int:
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description="docling vs legacy table-PDF ablation (issue #131)")
    parser.add_argument("--docs", type=int, default=20)
    parser.add_argument("--rows", type=int, default=60)
    parser.add_argument(
        "--out",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "docs",
            "benchmarks",
            "docling-ablation.json",
        ),
    )
    args = parser.parse_args()
    return asyncio.run(_main(args.docs, args.rows, args.out))


if __name__ == "__main__":
    sys.exit(main())
