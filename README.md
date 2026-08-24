<div align="center">

# Fusion-RAG

**Apple Silicon Native Offline Vector Knowledge Base Backend**

Local vector knowledge base service for the Fusion-MLX ecosystem — 100% offline, no data leaves your device.

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-164-success.svg)](tests/)
[![Version](https://img.shields.io/badge/Version-0.6.6-blue.svg)]()

[Quick Start](#quick-start) · [API Reference](#api-reference) · [Architecture](#architecture) · [Documentation](docs/)

</div>

---

## Why Fusion-RAG?

| Feature | Fusion-RAG | Dify RAG | LangChain RAG | Claude RAG |
|---------|-----------|----------|---------------|------------|
| **MLX native** | ✅ fusion-mlx API | ❌ Ollama/Cloud | ❌ Cloud API | ❌ Cloud |
| **Apple Silicon optimized** | ✅ LanceDB + MLX | ❌ | ❌ | ❌ |
| **Multi-KB isolation** | ✅ | ✅ | ❌ | ❌ |
| **BM25 + Vector hybrid** | ✅ Okapi BM25 + RRF | ❌ | ❌ | ❌ |
| **Contextual Retrieval** | ✅ Anthropic-style | ❌ | ❌ | ✅ |
| **MCP Server** | ✅ Claude/Cursor native | ❌ | ❌ | ❌ |
| **GraphRAG** | ✅ Entity-relation graph | ❌ | ❌ | ❌ |
| **Code AST chunking** | ✅ Python AST-aware | ❌ | ❌ | ❌ |
| **Version snapshots** | ✅ Hard-link snapshots | ❌ | ❌ | ❌ |
| **Incremental sync** | ✅ MD5 + mtime detection | ❌ | ❌ | ❌ |
| **Search templates** | ✅ Preset + custom | ❌ | ❌ | ❌ |
| **Audit logging** | ✅ Full search audit trail | ❌ | ❌ | ❌ |
| **Fine-grained permissions** | ✅ ACL with role inheritance | ❌ | ❌ | ❌ |
| **Cloud embed fallback** | ✅ Auto-fallback on local fail | ❌ | ✅ | ✅ |
| **Git repo indexing** | ✅ Clone + .gitignore aware | ❌ | ❌ | ❌ |
| **Bench API** | ✅ Latency benchmarks | ❌ | ❌ | ❌ |
| **Distributed storage** | ✅ StoreBackend abstraction | ❌ | ❌ | ❌ |
| **Local offline** | ✅ 100% | ⚠️ Partial | ❌ | ❌ |
| **Zero API cost** | ✅ | ❌ | ❌ | ❌ |

**One sentence:** Fusion-RAG is a unified local vector knowledge base backend for the Fusion-MLX ecosystem — all Embedding goes through fusion-mlx HTTP API, no direct MLX imports.

---

## Quick Start

### Prerequisites

- macOS with Apple Silicon (M1–M5)
- Python 3.12+
- [fusion-mlx](https://github.com/dahai80/fusion-mlx) running on `localhost:11432`

### Install

```bash
git clone https://github.com/dahai80/fusion-rag.git
cd fusion-rag
pip install -e ".[test]"
```

### Start the Server

```bash
./start.sh start
# Fusion-RAG running on http://127.0.0.1:11436
```

### Minimal Example

```python
import asyncio
from fusion_rag import KnowledgeBase, DocumentParser, Chunker
from fusion_rag.embed.client import EmbeddingClient
from fusion_rag.store.vector_store import VectorStore

async def main():
    # 1. Parse a document
    parser = DocumentParser()
    result = await parser.parse("README.md")
    print(f"Parsed: {result.file_name} ({result.chars} chars)")

    # 2. Chunk the text
    chunker = Chunker(strategy="semantic")
    chunks = await chunker.chunk(result)
    print(f"Chunks: {len(chunks)}")

    # 3. Embed via fusion-mlx
    embed = EmbeddingClient(model="BGE-M3")
    vectors = await embed.embed_batch([c.text for c in chunks])
    print(f"Vectors: {len(vectors)}")

asyncio.run(main())
```

---

## API Reference

Fusion-RAG provides a REST API at `/kb/*` for knowledge base operations.

### Knowledge Base Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/kb/bases` | List all knowledge bases |
| POST | `/kb/bases` | Create a knowledge base |
| GET | `/kb/bases/{id}` | Get knowledge base details |
| DELETE | `/kb/bases/{id}` | Delete a knowledge base |
| GET | `/kb/bases/{id}/stats` | Get KB statistics |

### Document Operations

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/kb/bases/{id}/documents` | Upload and index a document |
| POST | `/kb/bases/{id}/documents/batch` | Batch upload multiple documents |
| POST | `/kb/bases/{id}/documents/ingest` | Ingest inline content |
| GET | `/kb/bases/{id}/documents` | List all documents in a KB |
| DELETE | `/kb/bases/{id}/documents/{doc_id}` | Delete a document and its chunks |
| PUT | `/kb/bases/{id}/documents/{doc_id}` | Replace a document (re-index) |
| GET | `/kb/bases/{id}/documents/{doc_id}/status` | Get document indexing status |
| POST | `/kb/bases/{id}/scan` | Scan directory and index all files |
| POST | `/kb/bases/{id}/watch` | Watch files for changes |
| POST | `/kb/bases/{id}/unwatch` | Stop watching files |
| GET | `/kb/bases/{id}/watch/status` | Get watch status |

### Search & RAG

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/kb/bases/{id}/search` | Semantic search (hybrid, rerank, templates) |
| POST | `/kb/bases/{id}/ask` | RAG Q&A (model, hybrid, rerank, folder_prefix) |

#### Search Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `query` | (required) | Search query text |
| `top_k` | 10 | Number of results |
| `threshold` | 0.0 | Minimum similarity score |
| `hybrid` | false | Enable BM25+Vector hybrid search |
| `hybrid_alpha` | 0.7 | Vector weight (alpha fusion) |
| `hybrid_method` | "rrf" | Fusion method: "alpha" or "rrf" |
| `rerank` | false | Enable LLM reranking of results |
| `folder_prefix` | (none) | Filter results by folder path prefix |
| `template` | (none) | Search template name (general/code/design) |
| `rewrite_mode` | (none) | Query rewrite: hyde, expand, condense |

#### Ask Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `question` | (required) | Question text |
| `model` | "qwen3.5-9b" | LLM model for answer generation |
| `max_tokens` | 4096 | Max output tokens |
| `temperature` | 0.3 | Sampling temperature |
| `hybrid` | false | Enable hybrid search |
| `rerank` | false | Enable LLM reranking |
| `folder_prefix` | (none) | Filter by folder path |

### Version Snapshots

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/kb/bases/{id}/versions` | Create a snapshot |
| GET | `/kb/bases/{id}/versions` | List snapshots |
| GET | `/kb/bases/{id}/versions/{vid}` | Get snapshot details |
| POST | `/kb/bases/{id}/versions/{vid}/rollback` | Rollback to snapshot |
| DELETE | `/kb/bases/{id}/versions/{vid}` | Delete a snapshot |

### Search Templates

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/kb/bases/{id}/templates` | List search templates |
| POST | `/kb/bases/{id}/templates` | Create a custom template |
| DELETE | `/kb/bases/{id}/templates/{name}` | Delete a custom template |

### Permissions

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/kb/bases/{id}/permissions` | List permission rules |
| POST | `/kb/bases/{id}/permissions` | Add a permission rule |
| DELETE | `/kb/bases/{id}/permissions/{rule_id}` | Delete a permission rule |
| POST | `/kb/bases/{id}/permissions/check` | Check if action is allowed |

### Audit & Admin

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/kb/bases/{id}/audit` | Query audit logs |
| GET | `/kb/bases/{id}/audit/{log_id}` | Get a specific log entry |
| GET | `/kb/bases/{id}/audit/export` | Export audit logs (json/csv) |
| POST | `/kb/bases/{id}/sync` | Incremental directory sync |
| POST | `/kb/bases/{id}/bench` | Run search benchmark |
| GET | `/kb/bases/{id}/bench/results` | List benchmark results |

### Project-KB Mapping

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/kb/projects/{project_id}/kb` | Map project to KB |
| GET | `/kb/projects/{project_id}/kb` | Get project KB mapping |
| DELETE | `/kb/projects/{project_id}/kb` | Remove project KB mapping |

### MCP Protocol

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/mcp` | MCP JSON-RPC handler (tools/list, tools/call) |

### System

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/kb/status` | Service status |
| GET | `/health` | Health check |

---

## Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                   Fusion-RAG HTTP API (FastAPI)                   │
│  routes_kb.py  routes_docs.py  routes_search.py  routes_admin.py │
└───────────────────────────┬───────────────────────────────────────┘
                            │
┌───────────────────────────▼───────────────────────────────────────┐
│                      RAG Core Engine                              │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────────┐   │
│  │ DocumentParser │  │    Chunker     │  │  KnowledgeBase    │   │
│  │ PDF/DOCX/MD/   │  │ semantic/fixed │  │  Manager (CRUD)   │   │
│  │ TXT/HTML/Code  │  │ /code/AST      │  │  + isolation      │   │
│  └────────┬───────┘  └───────┬────────┘  └────────┬─────────┘   │
│  ┌────────┴───────┐  ┌───────┴────────┐  ┌────────┴─────────┐   │
│  │ VersionManager │  │ SearchTemplate │  │  AuditLogger     │   │
│  │ Snapshots/Roll │  │  Manager       │  │  PermissionMgr   │   │
│  └────────────────┘  └────────────────┘  └──────────────────┘   │
└───────────┼──────────────────┼────────────────────┼──────────────┘
            │                  │                    │
┌───────────▼──────────────────▼────────────────────▼──────────────┐
│                     Storage Layer                                 │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │   StoreBackend (ABC)                                       │  │
│  │   ├── LocalBackend (LanceDB + BM25)                        │  │
│  │   └── RemoteBackend (stub)                                 │  │
│  └────────────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────┐  ┌────────────────────────────┐  │
│  │   MetadataStore (SQLite)   │  │  EmbeddingCache (SQLite)   │  │
│  └────────────────────────────┘  └────────────────────────────┘  │
└───────────────────────────┬──────────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────────┐
│                    Embedding Client                               │
│  Primary: fusion-mlx /v1/embeddings → Fallback: Cloud API       │
└───────────────────────────┬──────────────────────────────────────┘
                            │ HTTP
┌───────────────────────────▼──────────────────────────────────────┐
│  fusion-mlx (/v1/embeddings, /v1/chat/completions)              │
│  Apple Silicon MLX Runtime (Metal GPU)                           │
└──────────────────────────────────────────────────────────────────┘
```

### LLM Client Infrastructure

All LLM chat calls (`/v1/chat/completions`) go through **fusion-core**'s HTTP client (`fusion_core.http_client`):

- `get_async_client(base_url, timeout)` — shared connection pool (LRU, keyed by loop + base_url), reused across modules
- `with_retry(fn, retries=2)` — automatic retry on 429/5xx and transient connection errors (backoff + jitter)
- Auth headers passed **per request**, not baked into the pooled client (avoids header leak between instances sharing a base_url)
- **D-H3 guard**: success path checks for empty/whitespace content and degrades explicitly (logged error/fallback) rather than returning empty content as a valid result

Modules migrated to fusion-core: `contextualizer`, `query_rewriter`, `reranker`, `rag_chain` (MultiTurnRAG + DocumentChain), `graph_rag`, `streaming` (MetadataExtractor; SSEStreamer keeps raw `httpx.stream`), `evaluator`, `api/routes._generate_answer`.

### Key Modules

| Module | File | Description |
|--------|------|-------------|
| **Knowledge Base** | `engine/knowledge_base.py` | KB CRUD, config, persistence |
| **Document Parser** | `engine/document.py` | PDF, DOCX, MD, TXT, HTML, code |
| **Chunker** | `engine/chunker.py` | Semantic, fixed, code, AST chunking |
| **AST Chunker** | `engine/ast_chunker.py` | Python AST-aware code chunking |
| **BM25 Index** | `engine/bm25_index.py` | Okapi BM25 with jieba Chinese tokenization |
| **Contextualizer** | `engine/contextualizer.py` | Anthropic Contextual Retrieval |
| **Reranker** | `engine/reranker.py` | Batch LLM reranking + HybridSearch (alpha/RRF) |
| **Query Rewriter** | `engine/query_rewriter.py` | HyDE, query expansion, condensation |
| **GraphRAG** | `engine/graph_rag.py` | Entity extraction and graph-based retrieval |
| **Evaluator** | `engine/evaluator.py` | RAG quality evaluation (faithfulness/relevance) |
| **Embedding Cache** | `engine/embedding_cache.py` | SQLite-backed embedding vector cache |
| **Version Manager** | `engine/version_manager.py` | KB snapshot and rollback |
| **Incremental Sync** | `engine/incremental_sync.py` | MD5+mtime change detection |
| **Search Templates** | `engine/search_template.py` | Preset + custom search strategies |
| **Audit Logger** | `engine/audit_logger.py` | Search audit trail with export |
| **Trajectory Writer** | `engine/trajectory_writer.py` | D1 retrieval trajectory sink (JSONL to `~/.fusion/trajectories/rag/`) |
| **Bench Runner** | `engine/bench.py` | Search latency benchmarking |
| **Permissions** | `permissions/acl.py` | Role-based ACL with path inheritance |
| **Git Loader** | `connectors/git_loader.py` | Git repo clone + index |
| **Store Backend** | `store/store_backend.py` | Abstract storage backend |
| **Local Backend** | `store/local_backend.py` | LanceDB + BM25 implementation |
| **Remote Backend** | `store/remote_backend.py` | Remote storage stub |
| **Auth** | `api/auth.py` | API key authentication |
| **MCP Server** | `api/mcp_server.py` | Model Context Protocol for Claude/Cursor |
| **Embedding** | `embed/client.py` | MLX Embedding + cloud fallback |
| **Vector Store** | `store/vector_store.py` | StoreBackend wrapper |
| **Metadata Store** | `store/metadata_store.py` | SQLite document/chunk tracking |
| **API Routes** | `api/routes.py` | Shared helpers + sub-router mount |
| **Server** | `api/server.py` | FastAPI server |

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FUSION_RAG_PORT` | 11436 | Server port |
| `FUSION_RAG_HOST` | 127.0.0.1 | Listen address |
| `FUSION_MLX_URL` | http://127.0.0.1:11432/v1 | fusion-mlx URL |
| `FUSION_MLX_API_KEY` | (empty) | MLX gateway API key (auto-detected from `~/.fusion-mlx/settings.json` if unset) |
| `FUSION_RAG_EMBED` | BGE-M3 | Embedding model |
| `FUSION_RAG_API_KEY` | (empty) | API key auth (disabled if empty) |
| `FUSION_RAG_AUTH_BACKEND` | apikey | Auth backend: `apikey` or `none` |
| `FUSION_RAG_SYSTEM_PROMPT` | (built-in) | Custom system prompt for RAG answer generation |
| `FUSION_RAG_FALLBACK_URL` | (empty) | Cloud embedding fallback URL |
| `FUSION_RAG_FALLBACK_API_KEY` | (empty) | Cloud fallback API key |
| `FUSION_TRAJECTORY_DIR` | ~/.fusion/trajectories/rag | RAG retrieval trajectory output dir (D1 轨迹飞轮) |

### Using start.sh

```bash
./start.sh start      # Start the server
./start.sh stop       # Stop the server
./start.sh restart    # Restart the server
./start.sh status     # Check server status
```

---

## Development

```bash
# Install dev dependencies
pip install -e ".[test]"

# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=fusion_rag --cov-report=term-missing

# Lint
ruff check fusion_rag/
```

---

## What's New in v0.6.6

- **RRF Hybrid Search Fix** — `_rrf_fusion` no longer applies a cosine-scaled `similarity_threshold` (0.3) to rank-fusion scores (max ~0.066); hybrid RRF search now returns matches instead of silently empty results. Regression test added.

## What's New in v0.6.0

- **Version Snapshots** — Hard-link based KB snapshots with rollback
- **Incremental Sync** — MD5 hash + mtime change detection for directories
- **AST Chunking** — Python AST-aware code chunking (auto-detect)
- **Search Templates** — 3 presets (general/code/design) + custom templates
- **Audit Logging** — Full search audit trail with JSON/CSV export
- **Fine-grained Permissions** — Role-based ACL with path-prefix inheritance
- **StoreBackend Abstraction** — LocalBackend (LanceDB) + RemoteBackend (extensible)
- **Bench API** — Search latency benchmarking with SQLite results
- **Cloud Embed Fallback** — Auto-fallback to cloud API when local fails
- **Git Repo Indexing** — Clone + .gitignore-aware file indexing
- **Routes Split** — Monolithic routes.py → 4 focused sub-modules

---

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.

## Acknowledgments

- [fusion-mlx](https://github.com/dahai80/fusion-mlx) — Apple Silicon model serving
- [LanceDB](https://github.com/lancedb/lancedb) — Vector database
- [LightRAG](https://github.com/HKUDS/LightRAG) — Reference architecture
