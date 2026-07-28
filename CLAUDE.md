# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fusion-RAG is an Apple Silicon native offline vector knowledge base backend for the Fusion-MLX ecosystem. All model inference (embeddings, chat) goes through fusion-mlx's HTTP API — never imports MLX, mlx-lm, or torch directly.

## Build & Run Commands

```bash
source .venv/bin/activate          # Always activate venv first
pip install -e ".[test]"           # Install with test deps

# Tests
python -m pytest tests/                              # All tests
python -m pytest tests/test_core.py                  # Single file
python -m pytest "tests/test_core.py::TestKnowledgeBaseConfig::test_defaults"  # Single case
python -m pytest tests/ --cov=fusion_rag --cov-report=term-missing  # With coverage

# Server
./start.sh start      # Start on http://127.0.0.1:11436
./start.sh stop
./start.sh restart
./start.sh status
```

Known test failures: All 164 tests passing (LanceDB DeleteResult bug fixed).

## Architecture

```
fusion_rag/
├── api/
│   ├── server.py        # FastAPI app factory + uvicorn runner
│   ├── routes.py        # /kb/* endpoints (CRUD, search, RAG, status)
│   ├── auth.py          # API key authentication (AuthConfig + verify_api_key)
│   └── mcp_server.py    # MCP JSON-RPC server (Claude/Cursor integration)
├── engine/
│   ├── knowledge_base.py  # KnowledgeBase + KnowledgeBaseManager (CRUD, persistence via kb_meta.json)
│   ├── document.py        # DocumentParser (PDF/DOCX/MD/TXT/HTML/code)
│   ├── chunker.py         # Chunker (semantic/fixed/code) + RecursiveChunker
│   ├── preprocessor.py    # DocumentPreprocessor (clean/normalize/dedup) + RecursiveChunker
│   ├── bm25_index.py      # BM25Index — Okapi BM25 with jieba Chinese tokenization, SQLite persistence
│   ├── contextualizer.py  # Contextualizer — Anthropic Contextual Retrieval for chunk context
│   ├── reranker.py        # Reranker (batch LLM scoring) + HybridSearch (alpha-weighted + RRF fusion)
│   ├── query_rewriter.py  # QueryRewriter — HyDE, query expansion, condensation
│   ├── rag_chain.py       # MultiTurnRAG (token-budget history) + DocumentChain (stuff/refine/map_reduce)
│   ├── graph_rag.py       # GraphRAG — entity extraction + relationship-aware retrieval
│   ├── evaluator.py       # RAGEvaluator — faithfulness/relevance/context_recall scoring
│   ├── embedding_cache.py # EmbeddingCache — SQLite-backed vector cache with TTL
│   ├── retrievers.py      # MMRRetriever, ContextCompressionRetriever, FusionRetriever
│   └── streaming.py       # SSEStreamer, MetadataExtractor, ResultCache
├── connectors/
│   └── __init__.py        # DatabaseConnector (SQLite/PostgreSQL) + WebLoader
├── embed/
│   └── client.py          # EmbeddingClient — calls fusion-mlx /v1/embeddings via httpx (with cache)
└── store/
    ├── vector_store.py    # VectorStore — LanceDB + BM25 hybrid search
    └── metadata_store.py  # MetadataStore — SQLite for document/chunk metadata
```

### Data Flow

1. **Ingest**: `DocumentParser.parse()` → `Chunker.chunk()` → `Contextualizer.contextualize()` → `EmbeddingClient.embed_batch()` (cached) → `VectorStore.add_batch()` + `BM25Index.add_documents()` + `MetadataStore`
2. **Search**: `QueryRewriter.rewrite()` (optional) → `EmbeddingClient.embed(query)` → `VectorStore.search()` + `BM25Index.search()` → `HybridSearch` (alpha or RRF fusion) → optional `Reranker.rerank()`
3. **RAG**: Search results → context assembly → fusion-mlx `/v1/chat/completions` → answer with source citations + token tracking

### Key Design Constraints

- **No direct MLX imports**: All inference via fusion-mlx HTTP API (`http://localhost:11434/v1`)
- **LanceDB lazy import**: `lancedb` and `pyarrow` imported via `_lancedb()` / `_pa()` helpers in `vector_store.py`
- **Per-KB isolation**: Each KB gets its own `vectors/` (LanceDB) + `metadata.db` (SQLite) under `~/.fusion-rag/stores/{kb_id}/`
- **Server wiring**: `server.py` creates `KnowledgeBaseManager` + `EmbeddingClient`, injects into `routes.py` via `set_kb_context()` globals. MCP router mounted at `/mcp`. Auth via `Depends(verify_api_key)` on write endpoints.

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `FUSION_RAG_PORT` | 11436 | Server port |
| `FUSION_RAG_HOST` | 127.0.0.1 | Listen address |
| `FUSION_MLX_URL` | http://localhost:11434/v1 | fusion-mlx base URL |
| `FUSION_RAG_EMBED` | BGE-M3 | Default embedding model |
| `FUSION_RAG_API_KEY` | (empty) | API key auth — empty = auth disabled |

### Dependencies

Runtime: httpx, fastapi, uvicorn, pydantic, lancedb, pyarrow, PyMuPDF, python-docx, markdownify, aiofiles, jieba, rank_bm25
Test: pytest, pytest-asyncio, pytest-cov, pytest-mock

Requires Python 3.12+ and macOS Apple Silicon. fusion-mlx must be running for embedding/RAG endpoints.
