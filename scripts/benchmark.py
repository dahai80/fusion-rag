# Callers: manual benchmark, CI. API: BM25Index, EmbeddingCache, _rrf_fusion. schema: result dict. user instruction: "完成所有待办任务"

import time
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger("benchmark")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

PRD_TARGETS = {
    "bm25_search_10k": {"max_ms": 100, "desc": "BM25 search <100ms @10K docs"},
    "embedding_cache_hit": {"min_pct": 90, "desc": "Embedding cache hit >90%"},
    "rrf_fusion": {"desc": "RRF fusion produces diverse results"},
}


def bench_bm25():
    logger.info("=== BM25 Search Benchmark ===")
    from fusion_rag.engine.bm25_index import BM25Index

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        idx = BM25Index(store_path=os.path.join(tmpdir, "bm25.db"))
        docs = []
        for i in range(10000):
            text = f"文档编号{i} 包含了关于机器学习和深度学习的内容 编号{i}还涉及自然语言处理技术"
            docs.append({"id": f"doc-{i}", "text": text, "metadata": {"source": f"file_{i}.txt"}})
        t0 = time.perf_counter()
        idx.add_documents(docs)
        build_ms = (time.perf_counter() - t0) * 1000
        logger.info("BM25 index build: %.1f ms for %d docs", build_ms, len(docs))

        t0 = time.perf_counter()
        results = idx.search("机器学习 深度学习", top_k=10)
        search_ms = (time.perf_counter() - t0) * 1000
        logger.info("BM25 search: %.1f ms, %d results", search_ms, len(results))

    passed = search_ms < PRD_TARGETS["bm25_search_10k"]["max_ms"]
    logger.info("BM25 target: <%.0fms, actual: %.1fms — %s",
                PRD_TARGETS["bm25_search_10k"]["max_ms"], search_ms,
                "PASS" if passed else "FAIL")
    return {"name": "bm25_search_10k", "value_ms": search_ms, "passed": passed}


def bench_embedding_cache():
    logger.info("=== Embedding Cache Benchmark ===")
    from fusion_rag.engine.embedding_cache import EmbeddingCache

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = EmbeddingCache(db_path=os.path.join(tmpdir, "cache.db"), ttl=3600)
        texts = [f"测试文本内容片段编号{i}用于缓存命中率测试" for i in range(100)]
        for text in texts:
            cache.set(text, [0.1] * 128)

        hits = 0
        misses = 0
        for text in texts:
            if cache.get(text) is not None:
                hits += 1
            else:
                misses += 1

        for i in range(10):
            new_text = f"新文本不在缓存中编号{i}"
            if cache.get(new_text) is None:
                misses += 1
            else:
                hits += 1

        hit_rate = hits / (hits + misses) * 100
        logger.info("Cache hits: %d, misses: %d, hit rate: %.1f%%", hits, misses, hit_rate)

    passed = hit_rate >= PRD_TARGETS["embedding_cache_hit"]["min_pct"]
    logger.info("Cache hit target: >%.0f%%, actual: %.1f%% — %s",
                PRD_TARGETS["embedding_cache_hit"]["min_pct"], hit_rate,
                "PASS" if passed else "FAIL")
    return {"name": "embedding_cache_hit", "value_pct": hit_rate, "passed": passed}


def bench_rrf_fusion():
    logger.info("=== RRF Fusion Benchmark ===")
    from fusion_rag.engine.reranker import HybridSearch

    vector_results = [
        {"id": f"vec-{i}", "score": 0.9 - i * 0.05, "text": f"Vector result {i}"}
        for i in range(20)
    ]
    keyword_results = [
        {"id": f"kw-{i}", "score": 10.0 - i, "text": f"Keyword result {i}"}
        for i in range(20)
    ]

    t0 = time.perf_counter()
    for _ in range(100):
        vs = type("VS", (), {"search": lambda self, q, k: vector_results})()
        hs = HybridSearch(vs, method="rrf")
        fused = hs._rrf_fusion(vector_results, keyword_results, top_k=10, k=60, threshold=0.0)
    fuse_ms = (time.perf_counter() - t0) / 100 * 1000
    logger.info("RRF fusion: %.3f ms avg, %d results", fuse_ms, len(fused))

    has_diversity = len(set(r["id"] for r in fused)) == len(fused)
    logger.info("RRF fusion produces diverse results: %s", has_diversity)

    return {"name": "rrf_fusion", "value_ms": fuse_ms, "passed": has_diversity and fuse_ms < 100}


def run_all():
    results = []
    logger.info("Fusion-RAG Performance Benchmark")
    logger.info("================================")

    results.append(bench_bm25())
    results.append(bench_embedding_cache())
    results.append(bench_rrf_fusion())

    logger.info("")
    logger.info("=== Summary ===")
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        logger.info("  %s: %s", r["name"], status)
    logger.info("Total: %d/%d passed", passed, total)

    return results


if __name__ == "__main__":
    run_all()
