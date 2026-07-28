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

Known test failures: 3 tests in `test_coverage_edge.py` and `test_coverage_final.py` fail due to LanceDB `delete()` returning `DeleteResult` instead of `int` — these need fixing.

## Architecture

```
fusion_rag/
├── api/
│   ├── server.py        # FastAPI app factory + uvicorn runner
│   └── routes.py        # /kb/* endpoints (CRUD, search, RAG, status)
├── engine/
│   ├── knowledge_base.py  # KnowledgeBase + KnowledgeBaseManager (CRUD, persistence via kb_meta.json)
│   ├── document.py        # DocumentParser (PDF/DOCX/MD/TXT/HTML/code)
│   ├── chunker.py         # Chunker (semantic/fixed/code) + RecursiveChunker
│   ├── preprocessor.py    # DocumentPreprocessor (clean/normalize/dedup) + RecursiveChunker
│   ├── reranker.py        # Reranker (LLM-scored) + HybridSearch (alpha-weighted vector+keyword fusion)
│   ├── retrievers.py      # MMRRetriever, ContextCompressionRetriever, FusionRetriever
│   ├── rag_chain.py       # MultiTurnRAG (history tracking) + DocumentChain (stuff/refine/map_reduce)
│   └── streaming.py       # SSEStreamer, MetadataExtractor, ResultCache
├── connectors/
│   └── __init__.py        # DatabaseConnector (SQLite/PostgreSQL) + WebLoader
├── embed/
│   └── client.py          # EmbeddingClient — calls fusion-mlx /v1/embeddings via httpx
└── store/
    ├── vector_store.py    # VectorStore — LanceDB with lazy import (cosine search + keyword search)
    └── metadata_store.py  # MetadataStore — SQLite for document/chunk metadata
```

### Data Flow

1. **Ingest**: `DocumentParser.parse()` → `Chunker.chunk()` → `EmbeddingClient.embed_batch()` → `VectorStore.add_batch()` + `MetadataStore`
2. **Search**: `EmbeddingClient.embed(query)` → `VectorStore.search()` or `HybridSearch` → optional `Reranker.rerank()`
3. **RAG**: Search results → context assembly → fusion-mlx `/v1/chat/completions` → answer with source citations

### Key Design Constraints

- **No direct MLX imports**: All inference via fusion-mlx HTTP API (`http://localhost:11434/v1`)
- **LanceDB lazy import**: `lancedb` and `pyarrow` imported via `_lancedb()` / `_pa()` helpers in `vector_store.py`
- **Per-KB isolation**: Each KB gets its own `vectors/` (LanceDB) + `metadata.db` (SQLite) under `~/.fusion-rag/stores/{kb_id}/`
- **Server wiring**: `server.py` creates `KnowledgeBaseManager` + `EmbeddingClient`, injects into `routes.py` via `set_kb_context()` globals

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `FUSION_RAG_PORT` | 11436 | Server port |
| `FUSION_RAG_HOST` | 127.0.0.1 | Listen address |
| `FUSION_MLX_URL` | http://localhost:11434/v1 | fusion-mlx base URL |
| `FUSION_RAG_EMBED` | BGE-M3 | Default embedding model |

### Dependencies

Runtime: httpx, fastapi, uvicorn, pydantic, lancedb, pyarrow, PyMuPDF, python-docx, markdownify, aiofiles
Test: pytest, pytest-asyncio, pytest-cov, pytest-mock

Requires Python 3.12+ and macOS Apple Silicon. fusion-mlx must be running for embedding/RAG endpoints.
