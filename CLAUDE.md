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

- **Embedding model lazy-loads on first call**: fusion-mlx serves BGE-M3 embeddings (MLX-format `model.safetensors`, resolved via fusion-mlx issue #248). The model is NOT loaded at startup — the first `/v1/embeddings` request after a cold start returns 502 while fusion-mlx loads the weights, then succeeds. fusion-rag's `EmbeddingClient` retries transparently, so this is a one-time startup latency, not a functional gap. Tests mock `embed_batch` (`patch.object(EmbeddingClient, "embed_batch", _fake_embed_batch)` → `[[0.01]*1024]`) so they never depend on a live fusion-mlx.

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
│   ├── routes_store.py    # /kb/bases/{kb_id}/store/* — M2M vector store surface (RemoteBackend server half)
│   ├── auth.py            # API key authentication (AuthConfig + verify_api_key)
│   ├── app_state.py       # Per-app state on app.state (contextvar-bound) + resource pools
│   ├── logging_setup.py   # O-P1-2/O-P2-2: RotatingFileHandler + JSON formatter + request-id
│   ├── access.py          # KB action ACL (require_kb_action / require_admin over permissions/acl)
│   ├── tenant.py          # Issue #61: gateway-origin enforcement + X-Fusion-Tenant scoping; #68: authoritative JWT tenant resolution (identity mode)
│   ├── identity.py        # Issue #68: IdentityClient — fusion-identity /verify HTTP client (authoritative tenant resolution)
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
│   ├── cross_encoder_reranker.py # CrossEncoderReranker — fusion-mlx /v1/rerank (real bge-reranker-v2-m3), #70
│   ├── query_rewriter.py    # QueryRewriter — HyDE, query expansion, condensation
│   ├── rag_chain.py         # MultiTurnRAG (token-budget history) + DocumentChain (stuff/refine/map_reduce)
│   ├── graph_rag.py         # GraphRAG — entity extraction + relationship-aware retrieval (internal library, NOT HTTP-wired)
│   ├── evaluator.py         # RAGEvaluator — faithfulness/relevance/context_recall scoring (internal library, NOT HTTP-wired)
│   ├── embedding_cache.py   # EmbeddingCache — SQLite-backed vector cache with TTL
│   ├── retrievers.py        # MMRRetriever, ContextCompressionRetriever, FusionRetriever (internal library, NOT HTTP-wired)
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
    ├── remote_backend.py    # RemoteBackend — HTTP client to a remote fusion-rag node's /store/* surface
    ├── qdrant_backend.py    # QdrantBackend — Qdrant vector DB + per-tenant collection isolation (#66)
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
- **Runtime config (D4)**: operator knobs (scan cap, embed cache TTL/size, RAG token budget, fetch_k multiplier, rerank model/backend/top_n) are env-driven via `RuntimeConfig` — see `runtime_config.py`. `get_runtime_config()` lazy-loads once, `reset_runtime_config()` for tests. Read-only view at `GET /kb/config`.
- **Cross-encoder rerank (#70)**: `CrossEncoderReranker` (`engine/cross_encoder_reranker.py`) is a plain async HTTP client to fusion-mlx's Cohere/Jina-compatible `POST /v1/rerank` (e.g. `bge-reranker-v2-m3`) — no MLX import (respects "no direct MLX imports" + "only modify own project"). Selected via `FUSION_RAG_RERANK_BACKEND=cross_encoder` + `FUSION_RAG_RERANK_MODEL=<model>`. When `FUSION_RAG_RERANK_MODEL` is set, `/search` and `/ask` default `rerank=true` (default-ON when a model is available). `rerank_top_n` (default 20) widens the candidate pool the cross-encoder re-scores before truncating to `top_k` (recall lift). `_do_rerank` (`routes.py`) runs a fallback chain: cross-encoder → legacy LLM-prompt `Reranker` (`engine/reranker.py`) → original retrieval order (logged, never crashes — backward compat). The legacy LLM-prompt `Reranker` is KEPT as the fallback tier (not replaced) so a missing/unreachable rerank model degrades, never fails. Hybrid retrieval (BM25 + vector + RRF) was already present (`hybrid=true`). Per-request params `rerank_backend`/`rerank_model`/`rerank_top_n` override the env defaults. `scripts/recall_benchmark.py` guards the no-regression contract offline (rerank ≥ hybrid ≥ vector); absolute lift needs a live cross-encoder run.
- **Multi-tenant isolation (#61)**: opt-in via `FUSION_RAG_REQUIRE_GATEWAY=1`. When on, `/kb/*` requests must carry `X-Fusion-Route: gateway-decision` (the gateway origin signal) or are rejected 403, and KB list/get are scoped to the `X-Fusion-Tenant` header (the authoritative, gateway-derived tenant). A `tenant` field on `KnowledgeBase` stamps ownership at create time; existing KBs have `tenant=None` and are invisible to tenant-scoped callers. The per-KB ACL (`access.py`) is the second defense (sub-tenant path rules); tenant scoping is the first (list/get hide other tenants' KBs). Default OFF — single-tenant local-first dev sees zero behavior change. `/health`, `/ready`, `/metrics`, `/mcp`, and auth routes are exempt from the gateway-origin gate so the service stays observable when the gateway is down.
- **Per-tenant vector collection isolation (#66)**: when `FUSION_RAG_STORE_BACKEND=qdrant`, `QdrantBackend` picks the Qdrant collection PER REQUEST from `tenant.get_request_tenant()` (the `X-Fusion-Tenant` contextvar). Tenant set → `tenant_id_{tenant}`; tenant None → shared `fusion_rag_kb_{kb_id}`. This is the physical data-layer isolation tier (cross-tenant vectors invisible at the collection level); the #61 KB `tenant` field is the logical tier. The backend reads the tenant lazily per call — one pooled backend serves all tenants, collection chosen per request. `chunk_id` (str) → Qdrant point id (int) via deterministic blake2b, so re-ingest overwrites. Keyword search stays in-process BM25 (Qdrant is vector-only). Default `local` backend unaffected.
- **Authoritative tenant resolution via fusion-identity (#68)**: opt-in via `FUSION_RAG_REQUIRE_IDENTITY=1`. When on, `/kb/*` requests require `Authorization: Bearer <JWT>`; `IdentityClient` (`api/identity.py`) verifies it via fusion-identity `POST /api/v1/auth/verify` (service-token-gated by `FUSION_IDENTITY_SERVICE_TOKEN`), and the JWT `tid` becomes the authoritative tenant, set into the SAME `_request_tenant` contextvar that #61/#66 scoping read — so KB scoping and per-tenant collection isolation work unchanged. `X-Fusion-Tenant` is demoted to defense-in-depth (must equal the JWT `tid` if present, else 401), retiring the blind-trust path. No/invalid/revoked JWT → 401; unreachable identity → fail-closed deny (NOT fallback to header). fusion-identity is an HTTP service dependency (port 11470, `FUSION_IDENTITY_URL`), NOT a Python import — fusion-rag talks to it over plain HTTP via `fusion_core.http_client`'s pooled client. Identity mode supersedes gateway mode on `/kb/*` when both are on (identity is the primary authority; the #61 gateway-origin gate is the defense-in-depth tier beneath it). Default OFF — single-tenant local-first dev sees zero behavior change. `/health`, `/ready`, `/metrics`, `/mcp`, `/v1`, `/auth`, and the dynamic `/store/*` M2M surface are exempt. A short-TTL in-process cache (15s, keyed by token hash) avoids hitting identity on every request in a burst; revoked tokens are never cached.

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
| `FUSION_RAG_STORE_BACKEND` | local | Vector store backend: `local` (LanceDB), `fusion-store` (HNSW via in-tree fusion-store PyO3 binding; install with `pip install -e ../fusion-store`, not on PyPI), `qdrant` (Qdrant vector DB with per-tenant collection isolation, #66; install with `pip install 'fusion-rag[qdrant]'`, see `QDRANT_*`), or `remote` (HTTP client to another fusion-rag node's `/kb/bases/{kb_id}/store/*` surface — see `FUSION_RAG_REMOTE_*`) |
| `FUSION_RAG_REMOTE_ENDPOINT` | (empty) | Base URL of the remote node for the `remote` backend (e.g. `http://node-b:11436`). Required when `FUSION_RAG_STORE_BACKEND=remote`. |
| `FUSION_RAG_REMOTE_API_KEY` | (empty) | API key sent as `X-API-Key` to the remote node. Empty = no auth. |
| `FUSION_RAG_REMOTE_KB_ID` | (derived) | KB id on the remote node. Defaults to this node's kb_id (leaf of `vector_path`). |
| `FUSION_RAG_REMOTE_TIMEOUT` | 30 | Per-request timeout (seconds) for the `remote` backend. |
| `QDRANT_URL` | :memory: | Qdrant endpoint for the `qdrant` backend. `:memory:` = in-process local mode (no server, dev/test); otherwise remote Qdrant URL. |
| `QDRANT_API_KEY` | (empty) | API key for the remote Qdrant server (ignored in `:memory:` mode). |
| `QDRANT_COLLECTION_PREFIX` | tenant_id_ | Prefix for per-tenant collection names. Tenant `T` → `{prefix}{T}`; no tenant → shared `fusion_rag_kb_{kb_id}`. |
| `FUSION_TRAJECTORY_DIR` | ~/.fusion/trajectories/rag | D1 retrieval trajectory output dir |
| `FUSION_RAG_LOG_LEVEL` | INFO | Server log level |
| `FUSION_RAG_SCAN_MAX_FILES` | 1000 | Max files a single scan_directory ingests (D4) |
| `FUSION_RAG_EMBED_CACHE_TTL` | 604800 | Embedding cache TTL seconds (D4, default 7d) |
| `FUSION_RAG_EMBED_CACHE_MAX_ENTRIES` | 100000 | Max embedding cache rows before LRU eviction (D4) |
| `FUSION_RAG_TOKEN_BUDGET` | 8192 | Multi-turn RAG token budget (D4) |
| `FUSION_RAG_MAX_HISTORY_TURNS` | 10 | Max RAG history turns kept (D4) |
| `FUSION_RAG_FETCH_K_MULTIPLIER` | 4 | Over-fetch factor for filtered search (D4) |
| `FUSION_RAG_RERANK_BACKEND` | llm | Rerank stage: `llm` (legacy LLM prompt-scoring `Reranker`) or `cross_encoder` (real cross-encoder via fusion-mlx `POST /v1/rerank`, e.g. `bge-reranker-v2-m3`). Issue #70. |
| `FUSION_RAG_RERANK_MODEL` | (empty) | Cross-encoder model name for the `cross_encoder` backend. When set, `/search`+`/ask` default `rerank=true` (rerank ON when a model is available). Empty = off unless caller passes `rerank=true`. Issue #70. |
| `FUSION_RAG_RERANK_TOP_N` | 20 | Candidate pool size fed to the reranker before truncating to `top_k` (recall lift, issue #70). |
| `FUSION_RAG_WATCH_CAP` | 16 | Max concurrent directory watches per process (R3) |
| `FUSION_RAG_TRAJECTORY_MAX_MB` | 100 | Max trajectory file MB before rotation (R6) |
| `FUSION_RAG_TRAJECTORY_KEEP` | 5 | Rotated trajectory files kept (R6) |
| `FUSION_RAG_AUDIT_RETENTION_DAYS` | 30 | Audit log retention days, 0 = forever (R6) |
| `FUSION_RAG_STORES_DIR` | ~/.fusion-rag/stores | Per-KB store root (R3 watch registry) |
| `FUSION_RAG_LOG_DIR` | `./logs` | Log file dir — RotatingFileHandler 10MB x 5 (O-P1-2) |
| `FUSION_RAG_LOG_FORMAT` | text | Log format: `text` or `json` (one JSON obj/line for aggregators) (O-P2-2) |
| `FUSION_RAG_SKIP_STARTUP_PROBE` | (empty) | `1` = skip boot-time fusion-mlx reachability probe (O-P2-3) |
| `FUSION_RAG_REQUIRE_GATEWAY` | (empty) | `1` = enforce multi-tenant isolation (issue #61): reject `/kb/*` requests missing `X-Fusion-Route: gateway-decision` (403), and scope KB list/get to the `X-Fusion-Tenant` header. Default OFF — single-tenant local-first dev unaffected. |
| `FUSION_RAG_REQUIRE_IDENTITY` | (empty) | `1` = authoritative tenant resolution via fusion-identity (issue #68): `/kb/*` requires `Authorization: Bearer <JWT>` verified by identity `/verify`; tenant = JWT `tid`; `X-Fusion-Tenant` demoted to defense-in-depth. Supersedes gateway mode on `/kb/*` when both on. Default OFF. |
| `FUSION_IDENTITY_URL` | http://127.0.0.1:11470 | fusion-identity base URL (HTTP service, port 11470). Used only when `FUSION_RAG_REQUIRE_IDENTITY=1`. |
| `FUSION_IDENTITY_SERVICE_TOKEN` | (empty) | Service token sent as `Authorization: Bearer` to identity `/api/v1/auth/verify`. Required when `FUSION_RAG_REQUIRE_IDENTITY=1` (fail-closed if unset). |
| `X-Fusion-Tenant` | — | Request header. Authoritative tenant when `FUSION_RAG_REQUIRE_GATEWAY=1` (#61); defense-in-depth (must match JWT `tid`) when `FUSION_RAG_REQUIRE_IDENTITY=1` (#68). Ignored as a scoping key otherwise. |
| `X-Fusion-Route` | — | Request header. `gateway-decision` = the request transited fusion-gateway; required on `/kb/*` when gateway isolation is on. `X-Space-Id` is a non-authoritative passthrough, ignored for scoping. |
| `Authorization` | — | Request header. `Bearer <JWT>` — the user JWT; required on `/kb/*` when `FUSION_RAG_REQUIRE_IDENTITY=1`. Verified via fusion-identity `/verify`; the `tid` claim is the authoritative tenant. |

### Deployment (O-P1-7)

Single-process service (H3): do NOT run `uvicorn --workers N` or multiple instances against one shared `stores` dir. Deploy artifacts:
- Root `Dockerfile` (issue #55) + `deploy/docker-compose.yml` — container deploy; `docker build -t fusion-rag .` from repo root. fusion-mlx stays on the host (Apple Silicon metal), reached via `FUSION_MLX_URL=http://host.docker.internal:11432/v1`. Stores on a named volume. No fusion-memory dependency (no UDS socket mount needed).
- `fusion-rag.service` — systemd unit; `TimeoutStopSec=40` drains in-flight requests before SIGKILL.
- `start.sh` — dev/single-user; nohup, logs to `logs/stdout.log` (bootstrap) + `logs/fusion-rag.log` (rotated).
Probes: `/health` = liveness (process up), `/ready` = readiness (deps reachable, 503 when down). Snapshot a KB's stores dir only after `POST /kb/bases/{kb_id}/checkpoint` (O-P2-1).

### Dependencies

Runtime: fusion-core (in-tree), httpx, fastapi, uvicorn, pydantic, lancedb, pyarrow, PyMuPDF, python-docx, markdownify, aiofiles, jieba, rank_bm25
Test: pytest, pytest-asyncio, pytest-cov, pytest-mock

Requires Python 3.12+ (CI matrix: 3.12, 3.13; local venv runs 3.14) and macOS Apple Silicon. fusion-mlx must be running for embedding/RAG endpoints.
