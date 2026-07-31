<div align="center">

# Fusion-RAG

**Apple Silicon Native Offline Vector Knowledge Base Backend**

Local vector knowledge base service for the Fusion-MLX ecosystem — 100% offline, no data leaves your device.

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-164-success.svg)](tests/)
<!-- callers: GitHub README page; API: documentation; user instruction: "按照你的方案和计划落地所有phase阶段的需求" -->

[Quick Start](#quick-start) · [API Reference](#api-reference) · [Architecture](#architecture) · [Documentation](docs/)

</div>

---

## Why Fusion-RAG?

| Feature | Fusion-RAG | Dify RAG | LangChain RAG |
|---------|-----------|----------|---------------|
| **MLX native** | ✅ fusion-mlx API | ❌ Ollama/Cloud | ❌ Cloud API |
| **Apple Silicon optimized** | ✅ LanceDB + MLX | ❌ | ❌ |
| **Multi-KB isolation** | ✅ | ✅ | ❌ |
| **BM25 + Vector hybrid** | ✅ Okapi BM25 + RRF | ❌ | ❌ |
| **Contextual Retrieval** | ✅ Anthropic-style | ❌ | ❌ |
| **MCP Server** | ✅ Claude/Cursor native | ❌ | ❌ |
| **GraphRAG** | ✅ Entity-relation graph | ❌ | ❌ |
| **Code-specific chunking** | ✅ | ❌ | ❌ |
| **Local offline** | ✅ 100% | ⚠️ Partial | ❌ |
| **Zero API cost** | ✅ | ❌ | ❌ |

**One sentence:** Fusion-RAG is a unified local vector knowledge base backend for the Fusion-MLX ecosystem — all Embedding goes through fusion-mlx HTTP API, no direct MLX imports.

---

## Quick Start

### Prerequisites

- macOS with Apple Silicon (M1–M5)
- Python 3.12+
- [fusion-mlx](https://github.com/dahai80/fusion-mlx) running on `localhost:11434`

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
| GET | `/kb/bases/{id}/documents` | List all documents in a KB |
| DELETE | `/kb/bases/{id}/documents/{doc_id}` | Delete a document and its chunks |
| PUT | `/kb/bases/{id}/documents/{doc_id}` | Replace a document (re-index) |
| GET | `/kb/bases/{id}/documents/{doc_id}/status` | Get document indexing status |
| POST | `/kb/bases/{id}/scan` | Scan directory and index all files |

### Search & RAG

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/kb/bases/{id}/search` | Semantic search (supports hybrid, rerank, folder_prefix) |
| POST | `/kb/bases/{id}/ask` | RAG Q&A (supports model, hybrid, rerank, folder_prefix) |

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
┌─────────────────────────────────────────────────────────────────┐
│                    Fusion-RAG HTTP API (FastAPI)                   │
│  /kb/bases  /kb/bases/{id}/documents  /kb/bases/{id}/search     │
│  /kb/bases/{id}/ask  /kb/status  /health                        │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                      RAG Core Engine                             │
│                                                                  │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────────┐  │
│  │ DocumentParser │  │    Chunker     │  │  KnowledgeBase    │  │
│  │ PDF/DOCX/MD/   │  │ semantic/fixed │  │  Manager (CRUD)   │  │
│  │ TXT/HTML/Code  │  │ /code-specific │  │  + isolation      │  │
│  └────────┬───────┘  └───────┬────────┘  └────────┬─────────┘  │
└───────────┼──────────────────┼────────────────────┼────────────┘
            │                  │                    │
┌───────────▼──────────────────▼────────────────────▼────────────┐
│                     Storage Layer                                │
│                                                                  │
│  ┌────────────────────────────┐  ┌────────────────────────────┐ │
│  │   VectorStore (LanceDB)    │  │   MetadataStore (SQLite)   │ │
│  │   Vector storage & search  │  │   Document/chunk metadata │ │
│  └────────────┬───────────────┘  └──────────────┬─────────────┘ │
└───────────────┼──────────────────────────────────┼───────────────┘
                │                                  │
┌───────────────▼──────────────────────────────────▼───────────────┐
│                    Embedding Client                              │
│  Calls fusion-mlx /v1/embeddings via HTTP API                    │
│  Never imports MLX, mlx-lm, or torch directly                    │
└───────────────────────────────┬─────────────────────────────────┘
                                │ HTTP
┌───────────────────────────────▼─────────────────────────────────┐
│  fusion-mlx (/v1/embeddings, /v1/chat/completions)               │
│  Apple Silicon MLX Runtime (Metal GPU)                          │
└─────────────────────────────────────────────────────────────────┘
```

### Key Modules

| Module | File | Description |
|--------|------|-------------|
| **Knowledge Base** | `engine/knowledge_base.py` | KB CRUD, config, persistence |
| **Document Parser** | `engine/document.py` | PDF, DOCX, MD, TXT, HTML, code |
| **Chunker** | `engine/chunker.py` | Semantic, fixed, code-specific chunking |
| **BM25 Index** | `engine/bm25_index.py` | Okapi BM25 with jieba Chinese tokenization |
| **Contextualizer** | `engine/contextualizer.py` | Anthropic Contextual Retrieval |
| **Reranker** | `engine/reranker.py` | Batch LLM reranking + HybridSearch (alpha/RRF) |
| **Query Rewriter** | `engine/query_rewriter.py` | HyDE, query expansion, condensation |
| **GraphRAG** | `engine/graph_rag.py` | Entity extraction and graph-based retrieval |
| **Evaluator** | `engine/evaluator.py` | RAG quality evaluation (faithfulness/relevance) |
| **Embedding Cache** | `engine/embedding_cache.py` | SQLite-backed embedding vector cache |
| **Auth** | `api/auth.py` | API key authentication |
| **MCP Server** | `api/mcp_server.py` | Model Context Protocol for Claude/Cursor |
| **Embedding** | `embed/client.py` | MLX Embedding via fusion-mlx HTTP API |
| **Vector Store** | `store/vector_store.py` | LanceDB storage (lazy import) + BM25 |
| **Metadata Store** | `store/metadata_store.py` | SQLite document/chunk tracking |
| **API Routes** | `api/routes.py` | FastAPI endpoints |
| **Server** | `api/server.py` | FastAPI server |

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FUSION_RAG_PORT` | 11436 | Server port |
| `FUSION_RAG_HOST` | 127.0.0.1 | Listen address |
| `FUSION_MLX_URL` | http://localhost:11434/v1 | fusion-mlx URL |
| `FUSION_RAG_EMBED` | BGE-M3 | Embedding model |
| `FUSION_RAG_API_KEY` | (empty) | API key auth (disabled if empty) |

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
```

### Test Stats
- **164 tests**, 0 failures
- **84%+ statement coverage**
- **Python 3.12+** compatible

---

## Comparison with Alternatives

| Dimension | LightRAG | PrivateGPT | **Fusion-RAG** |
|-----------|----------|-----------|--------------|
| **MLX native** | ❌ | ❌ | ✅ fusion-mlx API |
| **Apple Silicon optimized** | ❌ | ❌ | ✅ LanceDB + MLX |
| **BM25 + Vector hybrid** | ❌ | ❌ | ✅ RRF fusion |
| **Contextual Retrieval** | ❌ | ❌ | ✅ Anthropic-style |
| **MCP Server** | ❌ | ❌ | ✅ Claude/Cursor |
| **GraphRAG** | ❌ | ❌ | ✅ Entity graph |
| **Multi-KB isolation** | ❌ | ❌ | ✅ |
| **Code chunking** | ❌ | ❌ | ✅ |
| **Local offline** | ✅ | ✅ | ✅ 100% |
| **Zero API cost** | ✅ | ✅ | ✅ |

---

## License

MIT

## Acknowledgments

- [fusion-mlx](https://github.com/dahai80/fusion-mlx) — Apple Silicon model serving
- [LanceDB](https://github.com/lancedb/lancedb) — Vector database
- [LightRAG](https://github.com/HKUDS/LightRAG) — Reference architecture