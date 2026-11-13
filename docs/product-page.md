---
doc_id: fusion-rag/product-page
product: fusion-rag
class: platform
entry: service
doc_type: product-page
title: {zh: "Fusion-RAG 产品页", en: "Fusion-RAG Product Page"}
audience: engineer
status: draft
version: 0.8.21
updated: 2026-11-07
owner: "@dahai"
source_of_truth:
  - README.md
  - pyproject.toml
  - fusion_rag/api/server.py
---

# Fusion-RAG 产品页 / Fusion-RAG Product Page

Fusion-RAG 是 fusion-mlx 生态的本地离线向量知识库后端：一个单进程 FastAPI 服务（默认 `127.0.0.1:11436`），提供知识库 CRUD、文档解析/分块/索引、混合检索（BM25 + 向量 + RRF）、RAG 问答、GraphRAG、版本快照、ACL 权限与审计日志。所有 Embedding 与 LLM 调用走 fusion-mlx 的 HTTP API，本仓库不直接 import MLX。默认离线——数据不离开本机，除非显式配置云端 embedding 兜底（`FUSION_RAG_FALLBACK_URL`，默认为空）。

Fusion-RAG is the local offline vector knowledge base backend of the fusion-mlx ecosystem: a single-process FastAPI service (default `127.0.0.1:11436`) exposing knowledge base CRUD, document parse/chunk/index, hybrid retrieval (BM25 + vector + RRF), RAG Q&A, GraphRAG, version snapshots, ACL permissions and audit logging. All embedding and LLM calls go through the fusion-mlx HTTP API; this repo never imports MLX directly. Offline by default — data stays on the device unless a cloud embedding fallback is explicitly configured (`FUSION_RAG_FALLBACK_URL`, empty by default).

## Platform block

```yaml
platform:
  serves:            # 谁在消费这个服务 / who consumes this service
    - fusion-gateway (pass-through proxy: /shell/fusion-rag/* on :11432 -> :11436)
    - fusion-cowork / fusion-agent-studio (RAG retrieval over project KBs)
    - fusion-k12-teacher (BNUP K-12 math corpus ingestion + lesson retrieval)
    - fusion-trainer (knowledge-graph export + misconceptions for DPO samples)
    - Claude / Cursor (MCP server at POST /mcp)
    - built-in RAG console served at /shell/fusion-rag/ (and /console)
  builds_on:         # 依赖的底座 / underlying dependencies
    - fusion-mlx (:11432 via gateway, /v1/embeddings + /v1/chat/completions + /v1/rerank)
    - fusion-core (LLM HTTP client, retry pool, config resolution, logging)
    - LanceDB (default local vector store) + SQLite (metadata / audit / auth / cache)
    - optional: fusion-store (HNSW PyO3), Qdrant (per-tenant collections), OCR service (fusion-drawing-parser)
  hosts:             # 平台托管的数据与状态 / data and state hosted here
    - per-KB vector stores (default ~/.fusion-rag/stores) and SQLite metadata
    - KB lifecycle state: documents, chunks, version snapshots, ACL rules, audit trail
    - BNUP textbook corpus (edition/grade/unit/knowledge_point tags)
  integration_surface: # 对外接口 / external contract
    - REST /kb/* (bases, documents, search, ask, versions, templates, permissions, audit, bench, store)
    - REST /api/v1/bnup/* (corpus ingest/retrieve/KG/misconceptions/stats)
    - POST /mcp (MCP JSON-RPC: tools/list, tools/call)
    - GET /health, /ready, /metrics (Prometheus RED metrics)
  extension_points:  # 二次开发挂点 / where to extend
    - StoreBackend ABC (store/store_backend.py): implement a new vector backend
    - auth backend switch: FUSION_RAG_AUTH_BACKEND=apikey|none; gateway/identity tenancy via env flags
    - search templates (preset + custom), rerank backend (llm | cross_encoder)
    - connectors/git_loader.py for new corpus connectors
```

## 核心能力 / Core capabilities

### 1. 多知识库隔离 / Multi-KB isolation

消费方不再需要自己写：向量库建表、租户/库级目录布局、chunk 与 metadata 的落库与清理逻辑——调用 `POST /kb/bases` 建库后，文档、分块、版本快照都由服务管理。约束：单进程架构（H3），不要对同一 stores 目录跑多实例或 `--workers N`；默认 LocalBackend 的向量检索是暴力扫描，单库约 10 万 chunk 内体验良好，更大语料需切到 fusion-store 或 qdrant 后端。

What consumers no longer write: vector table creation, per-KB directory layout, chunk + metadata persistence and cleanup — create a KB via `POST /kb/bases` and the service owns documents, chunks, snapshots. Constraints: single-process by design (do not run multiple instances or `--workers N` against one stores dir); the default LocalBackend vector search is a brute-force scan, comfortable up to roughly 100k chunks per KB — larger corpora should switch to the fusion-store or qdrant backend.

### 2. 混合检索 / Hybrid retrieval (BM25 + vector + RRF)

消费方不再需要自己写：jieba 中文分词、Okapi BM25 索引、alpha/RRF 融合、rerank 两段式（候选池 `rerank_top_n` -> cross-encoder 截断到 `top_k`）、查询改写（hyde/expand/condense）。约束：召回质量依赖语料与查询形态——本仓库基准语料上 RRF recall@10 为 0.9212（`docs/benchmarks/README.md`，2026-10-25 实测），关键词型语料 BM25 占优，融合的价值是任一通道退化时的保险；rerank 默认关闭，设置 `FUSION_RAG_RERANK_MODEL` 后自动开启。

What consumers no longer write: jieba tokenization, Okapi BM25, alpha/RRF fusion, two-stage reranking, query rewriting. Constraints: recall depends on corpus and query shape — on this repo's benchmark corpus RRF reaches recall@10 0.9212 (see docs/benchmarks/README.md, measured 2026-10-25); rerank stays off until `FUSION_RAG_RERANK_MODEL` is set.

### 3. 文档摄取管道 / Document ingestion pipeline

消费方不再需要自己写：PDF/DOCX/EPUB/HTML/Markdown/代码文件的解析、semantic/fixed/code/AST 分块、扫描件 OCR 对接（fusion-drawing-parser）、目录扫描与 MD5+mtime 增量同步、目录 watch。约束：文件读取默认不设根目录限制（local-first），生产暴露端口前应设置 `FUSION_RAG_INGEST_ROOTS` 防路径穿越；单次 scan 默认上限 1000 个文件（`FUSION_RAG_SCAN_MAX_FILES`）；扫描 PDF 在未配置 OCR 服务时显式失败。

What consumers no longer write: format parsers, chunk strategies, OCR wiring, directory scan + MD5/mtime incremental sync, watch loops. Constraints: file reads are unconfined by default — set `FUSION_RAG_INGEST_ROOTS` before exposing the port; one scan caps at 1000 files by default; scanned PDFs fail visibly with no OCR service configured.

### 4. 治理面 / Governance: versions, ACL, audit

消费方不再需要自己写：知识库快照与回滚（硬链接快照）、按 KB 的 ACL 规则与角色继承、每次检索的审计流水与 JSON/CSV 导出、API key 管理。约束：水平扩容时不要用本机 `/kb/auth/keys` 发 key（单机 SQLite auth.db，多节点不共享）——gateway/identity 模式下该接口返回 409，改用共享的 `FUSION_RAG_API_KEY` 或身份服务签发的凭据；审计默认保留 30 天（`FUSION_RAG_AUDIT_RETENTION_DAYS`）。

What consumers no longer write: KB snapshots/rollback, per-KB ACL with role inheritance, search audit trail + export, API key minting. Constraints: in gateway/identity mode local key minting returns 409 — use the shared `FUSION_RAG_API_KEY` or identity-issued credentials; audit retention defaults to 30 days.

### 5. 多租户边界 / Multi-tenant isolation (opt-in)

消费方不再需要自己写：租户声明、KB 与租户的绑定与列表过滤、身份 JWT 校验、按租户隔离的向量集合。设置 `FUSION_RAG_REQUIRE_GATEWAY=1` 后 `/kb/*` 必须经网关（`X-Fusion-Route: gateway-decision`）并携带 `X-Fusion-Tenant`；设置 `FUSION_RAG_REQUIRE_IDENTITY=1` 后租户从 fusion-identity JWT 的 `tid` 权威解析（服务不可达时 fail-closed）。约束：两个开关默认关闭（单租户本地开发不受影响）；`X-Fusion-Tenant` 头本身可伪造，生产多租户必须走 identity 模式；identity 是 HTTP 依赖（:11470），不是 Python import。

What consumers no longer write: tenant stamping, KB scoping, JWT verification, per-tenant vector collections. Constraints: both flags default off; the `X-Fusion-Tenant` header alone is forgeable — production multi-tenant must use identity mode; fusion-identity is an HTTP dependency on :11470, not a Python import.

## 边界 / Non-goals

- 不做模型推理：embedding、chat、rerank 全部依赖 fusion-mlx 在线可用（本地兜底模型仅作 embedding 降级，且冷启动代价高，可用 `FUSION_RAG_EMBED_LOCAL_FALLBACK=0` 关闭）。
- 不做分布式存储：`remote` 后端只是指向另一个 fusion-rag 节点 `/store/*` 的 HTTP 客户端，不是通用分布式向量库。
- 不承诺多机共享状态：auth.db、审计、快照均为本机文件。

No inference here (embedding/chat/rerank all require fusion-mlx reachable), no general distributed vector store (the remote backend is an HTTP client to another fusion-rag node), no multi-node shared state (auth.db, audit, snapshots are local files).
