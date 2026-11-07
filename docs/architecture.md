---
doc_id: fusion-rag/architecture
product: fusion-rag
class: platform
entry: service
doc_type: architecture
title: {zh: "Fusion-RAG 架构", en: "Fusion-RAG Architecture"}
audience: engineer
status: draft
version: 0.8.21
updated: 2026-11-07
owner: @dahai
source_of_truth:
  - README.md
  - CLAUDE.md
  - fusion_rag/api/server.py
  - fusion_rag/store/store_backend.py
---

# Fusion-RAG 架构 / Fusion-RAG Architecture

## 总览 / Overview

单进程 FastAPI 服务，四层：HTTP API -> RAG 核心引擎 -> 存储层 -> embedding/LLM 客户端（全部走 fusion-mlx HTTP）。无内嵌模型、无直连 MLX import；进程标题 `fusion-rag serve :11436`（setproctitle，软依赖）。

A single-process FastAPI service in four layers: HTTP API -> RAG core engine -> storage -> embedding/LLM clients (all via the fusion-mlx HTTP API). No embedded models, no direct MLX imports; the process title is `fusion-rag serve :11436` (setproctitle, soft dependency).

```
HTTP API (FastAPI, :11436)
  routes_kb / routes_docs / routes_search / routes_admin / routes_auth
  routes_bnup / routes_graph / routes_store / routes_project / mcp_server
        |
RAG core engine
  DocumentParser | Chunker/ASTChunker | KnowledgeBaseManager
  VersionManager | SearchTemplate | AuditLogger | PermissionMgr
  BM25Index | Contextualizer | Reranker | QueryRewriter | GraphRAG | Evaluator
        |
Storage layer
  StoreBackend ABC -> LocalBackend (LanceDB + BM25)
                   -> FusionStoreBackend (HNSW) / QdrantBackend / RemoteBackend
  MetadataStore (SQLite) | EmbeddingCache (SQLite) | auth.db (SQLite)
        |
EmbeddingClient (fusion-mlx /v1/embeddings; local fallback; optional cloud)
LLM calls via fusion-core http_client (shared pool + retry)
        |
fusion-mlx :11432 (gateway) -> :11434 (Metal GPU)
```

## 关键决策与约束 / Key decisions and constraints

### H3 — 单进程 / single process

同一 stores 目录上不允许 `--workers N` 或多实例：LanceDB 目录、SQLite（metadata/audit/auth/cache）与 watch 注册表都假设单写者。横向扩容的正确路径是：共享 `FUSION_RAG_API_KEY`（config-plane key，见下）+ 每节点独立数据目录，或用 `remote` 后端把向量面集中到一个节点。

Running `--workers N` or multiple instances against one stores dir is unsupported: the LanceDB directories, all SQLite files and the watch registry assume a single writer. Scale out with the shared `FUSION_RAG_API_KEY` plus per-node data dirs, or centralize the vector plane with the `remote` backend.

### Embedding 单通道 / single embedding channel

所有 embedding 走 fusion-mlx `/v1/embeddings`。降级路径两条：本地 sentence-transformers 兜底（默认开；并发冷加载有双检锁保护，`FUSION_RAG_EMBED_LOCAL_FALLBACK=0` 可关闭改为快速失败 503），以及云端兜底（`FUSION_RAG_FALLBACK_URL`，默认空）。约束：换 embedding 模型 = 换向量空间，存量 KB 需重建索引；服务假设单 embedding 模型（见 DEPLOYMENT.md 硬约束）。

All embeddings flow through fusion-mlx. Two degradation paths: the local sentence-transformers fallback (on by default, double-checked-locking on the cold load; `FUSION_RAG_EMBED_LOCAL_FALLBACK=0` switches to fail-fast 503) and the cloud fallback (`FUSION_RAG_FALLBACK_URL`, empty by default). Constraint: changing the embedding model changes the vector space — existing KBs need reindexing; the service assumes a single embedding model per deployment.

### LLM 客户端经 fusion-core / LLM calls via fusion-core

`contextualizer`、`query_rewriter`、`reranker`、`rag_chain`、`graph_rag`、`evaluator`、`routes._generate_answer` 全部使用 `fusion_core.http_client`：共享连接池（按 loop+base_url LRU）、429/5xx 自动重试、逐请求鉴权头、空内容显式降级（D-H3 guard）。已知边界（A7）：池 key 不含 timeout，首个调用方的超时对同 loop+base_url 生效——fusion-rag 统一 60s 规避，根因在上游 fusion-core#16。

All chat-call modules use `fusion_core.http_client`: a shared pool (LRU keyed by loop+base_url), retry on 429/5xx, per-request auth headers, and explicit degradation on empty content (D-H3 guard). Known edge (A7): the pool key excludes timeout, so the first caller's timeout wins — fusion-rag unifies LLM timeouts at 60s; root cause tracked upstream in fusion-core#16.

### 租户模型 / tenancy model

请求租户解析顺序：identity JWT `tid`（`FUSION_RAG_REQUIRE_IDENTITY=1` 时权威）> `X-Fusion-Tenant` 头（`FUSION_RAG_REQUIRE_GATEWAY=1` 时要求，但可被直连方伪造）> 无租户（单租户默认）。KB 创建时打租户戳，list/get 按租户过滤；qdrant 后端按 `{prefix}{tenant_id}` 隔离集合。identity 服务不可达 = fail-closed（401），不回退到请求头。

Tenant resolution order: identity JWT `tid` (authoritative when `FUSION_RAG_REQUIRE_IDENTITY=1`) > `X-Fusion-Tenant` header (required when `FUSION_RAG_REQUIRE_GATEWAY=1`, forgeable by direct callers) > no tenant (single-tenant default). KBs are stamped at creation and scoped on list/get; the qdrant backend isolates collections per `{prefix}{tenant_id}`. An unreachable identity service fails closed (401) with no header fallback.

### 存储后端抽象 / store backend abstraction

`store/store_backend.py` 定义 ABC；四个实现共享同一 chunk-id 语义（含 `delete_by_chunk_ids` 精确删除，避免同内容文档共享物理 chunk 的误删，issue #94）。LocalBackend 暴力扫描换精确性；fusion-store 提供 HNSW；qdrant 提供租户集合；remote 把向量面代理到另一节点。新后端 = 实现 ABC + 接入 `FUSION_RAG_STORE_BACKEND`。

The StoreBackend ABC has four implementations sharing chunk-id semantics (including exact `delete_by_chunk_ids`, which prevents deleting physically-shared chunks of identical-content docs, issue #94). LocalBackend trades an index for exactness; fusion-store adds HNSW; qdrant adds tenant collections; remote proxies the vector plane to another node. A new backend means implementing the ABC and registering it under `FUSION_RAG_STORE_BACKEND`.

## 数据流 / Data flows

### 摄取 / ingest

上传或 scan -> DocumentParser（PDF/DOCX/EPUB/MD/TXT/HTML/code；扫描件走 OCR 服务）-> Chunker（semantic/fixed/code/AST）-> Contextualizer（可选）-> EmbeddingClient 批量向量 -> StoreBackend.add_batch + MetadataStore 落库。增量同步用 MD5+mtime 判重；删除走精确 chunk-id。

Upload or scan -> DocumentParser -> Chunker -> optional Contextualizer -> batch embeddings -> StoreBackend.add_batch + MetadataStore. Incremental sync dedupes on MD5+mtime; deletes use exact chunk ids.

### 检索 / search

query -> QueryRewriter（可选 hyde/expand/condense）-> 并行向量检索 + BM25（jieba 分词）-> RRF/alpha 融合 -> 过取 `fetch_k_multiplier` -> 可选 rerank（候选池 rerank_top_n -> cross_encoder 或 llm）-> 截断 top_k -> AuditLogger 记录 -> 响应（含分数与 low_confidence 标记）。

query -> optional rewrite -> parallel vector + BM25 (jieba) -> RRF/alpha fusion -> over-fetch -> optional rerank (pool rerank_top_n) -> truncate to top_k -> audit log -> response with scores and low_confidence flags.

### 问答 / ask

/search 的结果集 -> 组装上下文（token 预算 8192、历史 10 轮）-> fusion-core HTTP 客户端调 fusion-mlx `/v1/chat/completions` -> 带 citations 的答案。上游 401/403 时返回 502 "LLM credential rejected"，`/health` 的 `llm_auth` 位翻转，成功生成后自动清除。

/search results -> context assembly (8192-token budget, 10 history turns) -> fusion-mlx `/v1/chat/completions` via the fusion-core client -> answer with citations. Upstream 401/403 surfaces as 502 "LLM credential rejected" and flips the `llm_auth` field in `/health`, auto-cleared on the next successful generation.

## 模块地图 / Module map

顶层职责速查（完整表见 README Architecture 一节）：

- `api/` — FastAPI 路由分域 + auth/tenant/identity + mcp_server + logging_setup
- `engine/` — 解析、分块、检索、rerank、图、评测、版本、审计、轨迹
- `store/` — StoreBackend 四实现 + vector_store/metadata_store
- `embed/` — MLX 客户端 + 本地兜底（`embed/local.py`）
- `permissions/` — ACL 与角色继承
- `connectors/` — git_loader（clone + .gitignore 感知索引）
- `console/` — 内置控制台静态资源（`web/` 构建产物，package data）

Quick map (full table in the README Architecture section): `api/` routes by domain plus auth/tenant/mcp; `engine/` parse/chunk/retrieve/rerank/graph/eval/version/audit/trajectory; `store/` the four backends plus SQLite stores; `embed/` the MLX client and local fallback; `permissions/` ACL; `connectors/` the git loader; `console/` the built-in console assets built from `web/`.

## 部署形态 / Deploy shapes

三种支持形态：直接 `./start.sh`（开发/单机）、Docker（根 Dockerfile + `deploy/docker-compose.yml`）、systemd（`deploy/fusion-rag.service`）。硬约束与冷启动 runbook 见 [../DEPLOYMENT.md](../DEPLOYMENT.md)。

Three supported shapes: bare `./start.sh`, Docker (root Dockerfile + deploy/docker-compose.yml), systemd (deploy/fusion-rag.service). Hard constraints and the cold-start runbook: DEPLOYMENT.md.
