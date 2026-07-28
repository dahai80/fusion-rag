# Fusion-RAG API Reference

> Module-level documentation for `fusion_rag` packages.

---

## `fusion_rag.engine.knowledge_base` — Knowledge Base Management

```python
from fusion_rag.engine.knowledge_base import KnowledgeBaseManager, KnowledgeBase, KnowledgeBaseConfig
```

### KnowledgeBaseManager

Manages multiple knowledge bases with CRUD operations.

| Method | Returns | Description |
|--------|---------|-------------|
| `create(name, description, chunk_strategy, embedding_model)` | `KnowledgeBase` | Create a new KB |
| `get(kb_id)` | `KnowledgeBase` | Get KB by ID |
| `list()` | `list[dict]` | List all KBs |
| `delete(kb_id)` | `bool` | Delete a KB |
| `update(kb_id, **kwargs)` | `KnowledgeBase` | Update KB config |

### KnowledgeBaseConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | — | KB name |
| `description` | `str` | `""` | KB description |
| `chunk_strategy` | `str` | `"semantic"` | `semantic`, `code`, `fixed` |
| `chunk_size` | `int` | `512` | Chunk size |
| `chunk_overlap` | `int` | `64` | Chunk overlap |
| `embedding_model` | `str` | `"BGE-M3"` | Embedding model name |
| `max_results` | `int` | `10` | Max search results |
| `similarity_threshold` | `float` | `0.6` | Similarity threshold |

---

## `fusion_rag.engine.document` — Document Parser

```python
from fusion_rag.engine.document import DocumentParser, DocumentType, ParseResult
```

### DocumentParser

Parses documents of various formats into plain text.

| Method | Returns | Description |
|--------|---------|-------------|
| `parse(file_path)` | `ParseResult` | Parse a single file |
| `parse_directory(dir_path, recursive, max_files)` | `list[ParseResult]` | Parse all files in directory |
| `detect_type(file_path)` | `DocumentType` | Detect file type |
| `is_code_file(doc_type)` | `bool` | Check if code file |

### Supported Formats

| Type | Extension | Parser |
|------|-----------|--------|
| PDF | `.pdf` | PyMuPDF |
| DOCX | `.docx` | python-docx |
| Markdown | `.md`, `.markdown` | Direct read |
| TXT | `.txt` | Direct read |
| HTML | `.html`, `.htm` | markdownify |
| Code | `.py`, `.swift`, `.js`, `.ts`, `.c`, `.cpp`, `.rs`, `.go`, `.sh`, etc. | Direct read |

---

## `fusion_rag.engine.chunker` — Chunker

```python
from fusion_rag.engine.chunker import Chunker, Chunk
```

Splits parsed text into chunks for embedding.

| Method | Returns | Description |
|--------|---------|-------------|
| `chunk(result)` | `list[Chunk]` | Split a ParseResult into chunks |

### Strategies

| Strategy | Description |
|----------|-------------|
| `semantic` | Split by markdown headings or paragraph boundaries |
| `fixed` | Split by fixed character count with overlap |
| `code` | Split by function/class definitions |

---

## `fusion_rag.embed.client` — Embedding Client

```python
from fusion_rag.embed.client import EmbeddingClient
```

Generates text embeddings via fusion-mlx HTTP API. Never imports MLX directly.

**Constructor:**
```python
EmbeddingClient(base_url="http://localhost:11434/v1", model="BGE-M3", api_key="local")
```

| Method | Returns | Description |
|--------|---------|-------------|
| `embed(text)` | `list[float]` | Embed a single text string |
| `embed_batch(texts)` | `list[list[float]]` | Embed a batch of texts |
| `health()` | `bool` | Check if fusion-mlx is available |

---

## `fusion_rag.store.vector_store` — Vector Store

```python
from fusion_rag.store.vector_store import VectorStore
```

LanceDB-based vector storage. LanceDB is imported lazily.

**Constructor:**
```python
VectorStore(vector_path, dimension=1024)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `add(chunk_id, vector, text, ...)` | `None` | Add a single chunk |
| `add_batch(records)` | `None` | Add multiple chunks |
| `search(query_vector, top_k, threshold)` | `list[dict]` | Vector similarity search |
| `keyword_search(query, top_k)` | `list[dict]` | Full-text keyword search |
| `count()` | `int` | Total chunk count |
| `delete_by_doc(doc_path)` | `int` | Delete by document path |
| `clear()` | `None` | Clear all vectors |

---

## `fusion_rag.store.metadata_store` — Metadata Store

```python
from fusion_rag.store.metadata_store import MetadataStore
```

SQLite-backed metadata tracking for documents and chunks.

**Constructor:**
```python
MetadataStore(db_path)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `add_document(doc_id, file_path, ...)` | `None` | Register a document |
| `delete_document(doc_id)` | `None` | Delete a document |
| `get_document(doc_id)` | `dict \| None` | Get document metadata |
| `list_documents()` | `list[dict]` | List all documents |
| `add_chunk(chunk_id, doc_id, ...)` | `None` | Register a chunk |
| `doc_count()` | `int` | Total document count |
| `chunk_count()` | `int` | Total chunk count |

---

## `fusion_rag.api.routes` — HTTP API Routes

FastAPI routes mounted at `/kb/*`.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/kb/bases` | GET | List knowledge bases |
| `/kb/bases` | POST | Create knowledge base |
| `/kb/bases/{id}` | GET | Get KB details |
| `/kb/bases/{id}` | DELETE | Delete KB |
| `/kb/bases/{id}/documents` | POST | Upload document |
| `/kb/bases/{id}/scan` | POST | Scan directory |
| `/kb/bases/{id}/search` | POST | Semantic search |
| `/kb/bases/{id}/ask` | POST | RAG Q&A |
| `/kb/status` | GET | Service status |
| `/kb/bases/{id}/stats` | GET | KB statistics |
| `/health` | GET | Health check |