# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fusion-RAG is an Apple Silicon native offline vector knowledge base backend for the Fusion-MLX ecosystem. All model inference (embeddings, chat) goes through fusion-mlx's HTTP API — never imports MLX, mlx-lm, or torch directly.

## Build & Run Commands

```bash
source /Users/dahai/fusion/.venv/bin/activate   # Monorepo venv (Python 3.14); fusion_core already installed there
pip install -e ".[test]"                        # Install with test deps

# Tests
python -m pytest tests/                              # All tests
python -m pytest tests/test_core.py                  # Single file
python -m pytest "tests/test_core.py::TestKnowledgeBaseConfig::test_defaults"  # Single case
python -m pytest tests/ --cov=fusion_rag --cov-report=term-missing  # With coverage

# Lint (ruff config in pyproject.toml: E/F/W/I/S/SIM/C4/RUF/UP, line-length 120, py312)
ruff check .

# Server
./start.sh start      # Start on http://127.0.0.1:11436
./start.sh stop
./start.sh restart
./start.sh status
```

Known test failures: None — all tests green. Run via the monorepo venv (`source /Users/dahai/fusion/.venv/bin/activate`); `fusion_core` is an in-tree dependency, already installed there. Outside the venv (system Python) `import fusion_core` fails and 4 test modules error at collection.

### Benchmark

```bash
source .venv/bin/activate
python scripts/benchmark.py   # PRD metrics: BM25 <100ms, cache >90%, RRF fusion
```

### Known Limitations

- **No MLX embedding model**: fusion-mlx currently has no MLX-format embedding model (BGE-M3 uses pytorch_model.bin, not safetensors). Server runs but embedding/RAG endpoints return errors until an MLX embedding model is available upstream.

## Architecture

```
fusion_rag/
├── api/
│   ├── server.py          # FastAPI app factory + uvicorn runner
│   ├── routes.py          # /kb router: shared helpers, /status, /stats, sub-router mount hub
│   ├── routes_kb.py       # KB CRUD endpoints (/kb/bases/*)
│   ├── routes_docs.py     # Document ingest/list/delete/replace/scan/watch endpoints
│   ├── routes_search.py   # /search + /ask endpoints (hybrid, rerank, templates, rewrite)
│   ├── routes_admin.py    # Versions, templates, permissions, audit, sync, bench endpoints
│   ├── routes_project.py  # Project-KB mapping endpoints (/kb/projects/*)
│   ├── routes_auth.py     # Auth token/login endpoints
│   ├── auth.py            # API key authentication (AuthConfig + verify_api_key)
│   └── mcp_server.py      # MCP JSON-RPC server (Claude/Cursor integration)
├── engine/
│   ├── knowledge_base.py    # KnowledgeBase + KnowledgeBaseManager (CRUD, persistence via kb_meta.json)
│   ├── document.py          # DocumentParser (PDF/DOCX/MD/TXT/HTML/code)
│   ├── chunker.py           # Chunker (semantic/fixed/code) + RecursiveChunker
│   ├── ast_chunker.py       # ASTChunker — Python AST-aware code chunking (auto-detected)
│   ├── preprocessor.py      # DocumentPreprocessor (clean/normalize/dedup) + RecursiveChunker
│   ├── bm25_index.py        # BM25Index — Okapi BM25 with jieba Chinese tokenization, SQLite persistence
│   ├── contextualizer.py    # Contextualizer — Anthropic Contextual Retrieval for chunk context
│   ├── reranker.py          # Reranker (batch LLM scoring) + HybridSearch (alpha-weighted + RRF fusion)
│   ├── query_rewriter.py    # QueryRewriter — HyDE, query expansion, condensation
│   ├── rag_chain.py         # MultiTurnRAG (token-budget history) + DocumentChain (stuff/refine/map_reduce)
│   ├── graph_rag.py         # GraphRAG — entity extraction + relationship-aware retrieval
│   ├── evaluator.py         # RAGEvaluator — faithfulness/relevance/context_recall scoring
│   ├── embedding_cache.py   # EmbeddingCache — SQLite-backed vector cache with TTL
│   ├── retrievers.py        # MMRRetriever, ContextCompressionRetriever, FusionRetriever
│   ├── version_manager.py   # KB snapshot/rollback via hard-link copies
│   ├── incremental_sync.py  # MD5 + mtime change detection for directory sync
│   ├── search_template.py   # Preset (general/code/design) + custom search templates
│   ├── audit_logger.py      # Search audit trail with JSON/CSV export
│   ├── trajectory_writer.py # D1 retrieval trajectory sink (JSONL → ~/.fusion/trajectories/rag/)
│   ├── bench.py             # Search latency benchmark runner + SQLite results
│   ├── runtime_config.py    # RuntimeConfig — env-driven operator knobs (scan cap, cache TTL, token budget) + reset for tests
│   ├── metrics.py           # R5 RED metrics registry + middleware → /metrics (Prometheus text format)
│   └── streaming.py         # SSEStreamer, MetadataExtractor
├── parse/
│   ├── __init__.py          # DatabaseConnector (SQLite/PostgreSQL) + WebLoader
│   └── git_loader.py        # Git repo clone + .gitignore-aware file indexing
├── permissions/
│   └── acl.py               # Role-based ACL with path-prefix inheritance
├── embed/
│   ├── client.py            # EmbeddingClient — fusion-mlx /v1/embeddings (own retry, cloud fallback)
│   └── local.py             # Local embedding fallback
└── store/
    ├── store_backend.py     # StoreBackend ABC + StoreBackendFactory
    ├── local_backend.py     # LocalBackend — LanceDB + BM25 implementation
    ├── remote_backend.py    # RemoteBackend — remote storage stub (extensible)
    ├── vector_store.py      # VectorStore — StoreBackend wrapper, hybrid search
    ├── fusion_store_backend.py  # FusionStoreBackend — fusion-store HNSW (PyO3) + in-process BM25
    └── metadata_store.py    # MetadataStore — SQLite document/chunk metadata
```

### Data Flow

1. **Ingest**: `DocumentParser.parse()` → `Chunker.chunk()` → `Contextualizer.contextualize()` → `EmbeddingClient.embed_batch()` (cached) → `VectorStore.add_batch()` + `BM25Index.add_documents()` + `MetadataStore`
2. **Search**: `QueryRewriter.rewrite()` (optional) → `EmbeddingClient.embed(query)` → `VectorStore.search()` + `BM25Index.search()` → `HybridSearch` (alpha or RRF fusion) → optional `Reranker.rerank()`
3. **RAG**: Search results → context assembly → fusion-mlx `/v1/chat/completions` → answer with source citations + token tracking

### Key Design Constraints

- **No direct MLX imports**: All inference via fusion-mlx HTTP API (`http://127.0.0.1:11432/v1`)
- **fusion_core in-tree dependency**: LLM HTTP calls (reranker, contextualizer, query_rewriter, rag_chain, graph_rag, evaluator, streaming, routes._generate_answer) use `fusion_core.http_client.get_async_client` (shared connection pool, LRU-keyed by loop+base_url) + `with_retry` (auto-retry on 429/5xx + transient errors). Auth headers passed per-request, not baked into pooled client. `fusion_core` lives at `../fusion-core` (in-tree), already in the monorepo venv; CI installs it via `pip install git+https://github.com/dahai80/fusion-core.git`. Non-LLM httpx (`embed/client.py`, `connectors`) and SSE streaming (`streaming.SSEStreamer`) keep raw httpx (own retry / `httpx.stream`).
- **LanceDB lazy import**: `lancedb` and `pyarrow` imported via `_lancedb()` / `_pa()` helpers in `vector_store.py`
- **Per-KB isolation**: Each KB gets its own `vectors/` (LanceDB) + `metadata.db` (SQLite) under `~/.fusion-rag/stores/{kb_id}/`
- **Server wiring**: `server.py` creates `KnowledgeBaseManager` + `EmbeddingClient`, injects into `routes.py` via `set_kb_context()`. `routes.py` is the hub router (`/kb` prefix) that mounts 5 sub-routers (kb/docs/search/admin/project); it also holds `/status`, `/stats`, and shared helpers (`_get_base`, `_do_rerank`, `_generate_answer`). MCP router mounted at `/mcp`, auth router at top level. Auth via `Depends(verify_api_key)` on write endpoints. `/metrics` (Prometheus, no auth) + `/health` mounted at top level.
- **Single-process only**: directory watches (`routes_docs._watch_loop`) and the watch registry live in process memory — no cross-process coordination. Scale horizontally behind a stateless load balancer, NOT by running multiple fusion-rag processes against the same `FUSION_RAG_STORES_DIR`. A multi-process deployment would double-watch and corrupt the registry (R3/H3).
- **Single embedding model**: the service builds ONE `EmbeddingClient` from `FUSION_RAG_EMBED` at startup; all KBs share it. A KB created with a different `embedding_model` config is rejected at ingest (400) rather than silently persisting cross-model vectors (D7). To change the model, re-create the KB or restart the service with a new `FUSION_RAG_EMBED`.
- **Runtime config (D4)**: operator knobs (scan cap, embed cache TTL/size, RAG token budget, fetch_k multiplier) are env-driven via `RuntimeConfig` — see `runtime_config.py`. `get_runtime_config()` lazy-loads once, `reset_runtime_config()` for tests. Read-only view at `GET /kb/config`.

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `FUSION_RAG_PORT` | 11436 | Server port |
| `FUSION_RAG_HOST` | 127.0.0.1 | Listen address |
| `FUSION_MLX_URL` | http://127.0.0.1:11432/v1 | fusion-mlx base URL |
| `FUSION_MLX_API_KEY` | (auto-detected) | MLX API key; auto-read from `~/.fusion-mlx/settings.json` if unset |
| `FUSION_RAG_EMBED` | BGE-M3 | Default embedding model |
| `FUSION_RAG_API_KEY` | (empty) | API key auth — empty = auth disabled |
| `FUSION_RAG_AUTH_BACKEND` | apikey | Auth backend: `apikey` or `none` |
| `FUSION_RAG_SYSTEM_PROMPT` | (built-in) | Custom system prompt for RAG generation |
| `FUSION_RAG_FALLBACK_URL` | (empty) | Cloud embedding fallback URL (used when local embed fails) |
| `FUSION_RAG_FALLBACK_API_KEY` | (empty) | Cloud fallback API key |
| `FUSION_RAG_STORE_BACKEND` | local | Vector store backend: `local` (LanceDB) or `fusion-store` (HNSW via in-tree fusion-store PyO3 binding; install with `pip install -e ../fusion-store`, not on PyPI) |
| `FUSION_TRAJECTORY_DIR` | ~/.fusion/trajectories/rag | D1 retrieval trajectory output dir |
| `FUSION_RAG_LOG_LEVEL` | INFO | Server log level |
| `FUSION_RAG_SCAN_MAX_FILES` | 1000 | Max files a single scan_directory ingests (D4) |
| `FUSION_RAG_EMBED_CACHE_TTL` | 604800 | Embedding cache TTL seconds (D4, default 7d) |
| `FUSION_RAG_EMBED_CACHE_MAX_ENTRIES` | 100000 | Max embedding cache rows before LRU eviction (D4) |
| `FUSION_RAG_TOKEN_BUDGET` | 8192 | Multi-turn RAG token budget (D4) |
| `FUSION_RAG_MAX_HISTORY_TURNS` | 10 | Max RAG history turns kept (D4) |
| `FUSION_RAG_FETCH_K_MULTIPLIER` | 4 | Over-fetch factor for filtered search (D4) |
| `FUSION_RAG_WATCH_CAP` | 16 | Max concurrent directory watches per process (R3) |
| `FUSION_RAG_TRAJECTORY_MAX_MB` | 100 | Max trajectory file MB before rotation (R6) |
| `FUSION_RAG_TRAJECTORY_KEEP` | 5 | Rotated trajectory files kept (R6) |
| `FUSION_RAG_AUDIT_RETENTION_DAYS` | 30 | Audit log retention days, 0 = forever (R6) |
| `FUSION_RAG_STORES_DIR` | ~/.fusion-rag/stores | Per-KB store root (R3 watch registry) |

### Dependencies

Runtime: fusion-core (in-tree), httpx, fastapi, uvicorn, pydantic, lancedb, pyarrow, PyMuPDF, python-docx, markdownify, aiofiles, jieba, rank_bm25
Test: pytest, pytest-asyncio, pytest-cov, pytest-mock

Requires Python 3.12+ (CI matrix: 3.12, 3.13; local venv runs 3.14) and macOS Apple Silicon. fusion-mlx must be running for embedding/RAG endpoints.
