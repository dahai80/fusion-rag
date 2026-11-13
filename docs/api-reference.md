---
doc_id: fusion-rag/api-reference
product: fusion-rag
class: platform
entry: service
doc_type: api-reference
title: {zh: "Fusion-RAG HTTP API 参考", en: "Fusion-RAG HTTP API Reference"}
audience: engineer
status: draft
version: 0.8.21
updated: 2026-11-07
owner: "@dahai"
source_of_truth:
  - README.md
  - fusion_rag/api/routes.py
  - fusion_rag/api/routes_kb.py
  - fusion_rag/api/routes_search.py
  - fusion_rag/api/routes_admin.py
  - fusion_rag/api/routes_auth.py
---

# Fusion-RAG HTTP API 参考 / HTTP API Reference

服务根地址 `http://127.0.0.1:11436`（`FUSION_RAG_PORT` / `FUSION_RAG_HOST` 可改）。除 `/health`、`/ready`、`/metrics`、`/mcp`、`/console` 外，`/kb/*` 在设置 `FUSION_RAG_API_KEY` 后需带 `X-API-Key` 头；`FUSION_RAG_REQUIRE_GATEWAY=1` 时还需网关决策头；`FUSION_RAG_REQUIRE_IDENTITY=1` 时需 `Authorization: Bearer` JWT。错误响应统一为 `{ "detail": "..." }`。

Base URL is `http://127.0.0.1:11436`. Besides `/health`, `/ready`, `/metrics`, `/mcp` and `/console`, all `/kb/*` routes require an `X-API-Key` header when `FUSION_RAG_API_KEY` is set, gateway proof under `FUSION_RAG_REQUIRE_GATEWAY=1`, and a Bearer JWT under `FUSION_RAG_REQUIRE_IDENTITY=1`. Errors return `{ "detail": "..." }`.

本页只列接口族与代表端点；逐字段参数见 `README.md` 的 API Reference 一节。经网关访问时把前缀换成 `http://127.0.0.1:11432/shell/fusion-rag`，其余路径不变（#109 契约）。

This page lists route families and representative endpoints; full per-field parameters live in the README API Reference section. Behind the gateway, swap the prefix to `http://127.0.0.1:11432/shell/fusion-rag` and keep the paths unchanged.

## 知识库管理 / Knowledge base management

| Method | Path | 说明 / Description |
|--------|------|--------------------|
| GET | `/kb/bases` | 列出 KB（多租户模式下按租户过滤）/ List KBs (tenant-scoped in multi-tenant mode) |
| POST | `/kb/bases` | 建 KB：`name`, `description`, `chunk_strategy`, `embedding_model` 等 / Create a KB |
| GET | `/kb/bases/{kb_id}` | KB 详情 / KB details |
| DELETE | `/kb/bases/{kb_id}` | 删 KB / Delete a KB |
| GET | `/kb/bases/{kb_id}/stats` | 文档数、chunk 数等统计 / doc/chunk counts and stats |
| GET | `/kb/status` | 服务状态 / service status |

## 文档摄取 / Document ingestion

| Method | Path | 说明 / Description |
|--------|------|--------------------|
| POST | `/kb/bases/{kb_id}/documents/ingest` | 内联内容摄取：text/markdown/html/csv 直接索引；pdf/docx/epub 的 `content` 为 base64 二进制，走完整解析管道（含扫描件 OCR）；非法 base64 返回 400 / Inline ingest; binary types are base64-decoded through the full parse pipeline |
| POST | `/kb/bases/{kb_id}/documents` | 上传文件索引 / Upload and index a file |
| POST | `/kb/bases/{kb_id}/documents/batch` | 批量上传 / Batch upload |
| GET | `/kb/bases/{kb_id}/documents` | 文档列表 / List documents |
| PATCH | `/kb/bases/{kb_id}/documents/{doc_id}` | 生命周期元数据：`status` (active/deprecated), `superseded_by` / lifecycle metadata patch |
| PUT | `/kb/bases/{kb_id}/documents/{doc_id}` | 整文档替换重索引 / Replace and re-index |
| DELETE | `/kb/bases/{kb_id}/documents/{doc_id}` | 删除文档及其 chunk（精确 chunk-id 删除，不影响同内容文档）/ Delete doc + chunks by exact chunk ids |
| GET | `/kb/bases/{kb_id}/documents/{doc_id}/status` | 索引状态 / Indexing status |
| POST | `/kb/bases/{kb_id}/scan` | 目录扫描摄取（默认上限 1000 文件）/ Directory scan ingest (1000-file default cap) |
| POST | `/kb/bases/{kb_id}/sync` | MD5+mtime 增量同步 / Incremental sync |
| POST | `/kb/bases/{kb_id}/watch` / `/unwatch` / GET `.../watch/status` | 目录变更监听（进程内上限 16 个）/ Directory watches (cap 16 per process) |

## 检索与问答 / Search and RAG

| Method | Path | 说明 / Description |
|--------|------|--------------------|
| POST | `/kb/bases/{kb_id}/search` | 语义/混合检索：`hybrid`, `hybrid_method` (alpha/rrf), `rerank`, `rerank_backend`, `template`, `rewrite_mode`, `folder_prefix`, `status_filter` / Semantic or hybrid search with rerank, templates, rewrite and status filter |
| POST | `/kb/bases/{kb_id}/ask` | RAG 问答：生成模型默认 `qwen3.5-9b`，响应含 `answer` + `citations[]` + `has_low_confidence` / RAG answer with citations |
| POST | `/kb/bases/{kb_id}/similar` | 近重复/版本漂移检索（按文本或原始向量）/ Near-duplicate search by text or raw embedding |
| GET | `/kb/bases/{kb_id}/templates` / POST / DELETE `.../templates/{name}` | 搜索模板（内置 general/code/design + 自定义）/ Search templates (preset + custom) |
| POST | `/kb/bases/{kb_id}/bench` / GET `.../bench/results` | 检索延迟基准 / Search latency benchmark |

关键检索参数（/search 与 /ask 共有）/ Key search parameters shared by /search and /ask:

- `top_k`（默认 10）、`threshold`（默认 0.0）、`hybrid`（默认 false）、`hybrid_alpha` 0.7、`hybrid_method` rrf
- `rerank`：设了 `FUSION_RAG_RERANK_MODEL` 时默认开启，否则需显式传 true
- `rerank_backend`：`cross_encoder`（fusion-mlx `/v1/rerank`）或 `llm`（旧版提示打分）
- `rerank_top_n`（默认 20）：送入 rerank 的候选池，截断到 `top_k` 前扩召回
- `status_filter`：`active`（默认，剔除已作废文档）/ `deprecated` / `all`（作废 chunk 在上下文中标注，citations 带 `low_confidence` 标记）

## 版本快照 / Version snapshots

| Method | Path | 说明 / Description |
|--------|------|--------------------|
| POST | `/kb/bases/{kb_id}/versions` | 创建硬链接快照 / Create a hard-link snapshot |
| GET | `/kb/bases/{kb_id}/versions` / GET `.../versions/{vid}` | 列表 / 详情 / List / details |
| POST | `/kb/bases/{kb_id}/versions/{vid}/rollback` | 回滚 / Rollback |
| DELETE | `/kb/bases/{kb_id}/versions/{vid}` | 删除快照 / Delete snapshot |

## 治理 / Governance: permissions, audit, auth

| Method | Path | 说明 / Description |
|--------|------|--------------------|
| GET/POST | `/kb/bases/{kb_id}/permissions` | ACL 规则列表 / 新增（新增需 admin）/ ACL rules; adding requires admin |
| DELETE | `/kb/bases/{kb_id}/permissions/{rule_id}` | 删规则（admin）/ Delete a rule (admin) |
| POST | `/kb/bases/{kb_id}/permissions/check` | 校验某操作是否允许 / Check whether an action is allowed |
| GET | `/kb/bases/{kb_id}/audit` / `.../audit/{log_id}` / `.../audit/export` | 审计查询 / 单条 / 导出 (json, csv)；保留期默认 30 天 / Audit query, entry, export; 30-day default retention |
| GET | `/kb/auth/keys` / POST / DELETE `/kb/auth/keys/{key_hash_prefix}` | API key 管理（admin）；gateway/identity 模式下 mint 返回 409 / API key admin; minting returns 409 in gateway/identity mode |
| GET | `/kb/config` | 运行时配置只读视图 / Read-only runtime config view |

## 项目映射 / Project-KB mapping

| Method | Path | 说明 / Description |
|--------|------|--------------------|
| POST/GET/DELETE | `/kb/projects/{project_id}/kb` | 上层应用（cowork/agent-studio 等）把 project 绑定到某个 KB / Bind a consumer project to a KB |

## BNUP 教材语料 / BNUP textbook corpus

| Method | Path | 说明 / Description |
|--------|------|--------------------|
| POST | `/api/v1/bnup/ingest` | 摄取北师大版数学语料 JSON（幂等，重复摄取为覆盖）/ Ingest the BNUP math corpus JSON (idempotent) |
| GET | `/api/v1/bnup/retrieve` | 按 `knowledge_point` / `grade` 混合检索课时 / Retrieve lessons by knowledge point or grade |
| GET | `/api/v1/bnup/knowledge_graph` | 导出 KG 层级 + 知识点平面索引（供 fusion-trainer）/ Export KG hierarchy + flat knowledge-point index |
| GET | `/api/v1/bnup/misconceptions` | 常见错误数据（DPO 负样本；上游未填充前返回空）/ Common-error data for DPO negatives; empty until upstream populates |
| GET | `/api/v1/bnup/stats` | 按年级/学期/知识点的覆盖统计 / Coverage stats per grade, semester, knowledge point |

## 分布式 store 面 / Distributed store surface

`/kb/bases/{kb_id}/store/*` 是 RemoteBackend 的服务端半边（M2M 向量面）：`add_batch`、`search`、`keyword_search`、`delete_by_doc`、`delete_by_chunk_ids` 等。仅当另一 fusion-rag 节点以 `FUSION_RAG_STORE_BACKEND=remote` 指向本节点时使用。

`/kb/bases/{kb_id}/store/*` is the server half of RemoteBackend (machine-to-machine vector surface): add_batch, search, keyword_search, delete_by_doc, delete_by_chunk_ids. Used only when another fusion-rag node points at this one with `FUSION_RAG_STORE_BACKEND=remote`.

## MCP / 系统 / MCP and system

| Method | Path | 说明 / Description |
|--------|------|--------------------|
| POST | `/mcp` | MCP JSON-RPC（tools/list、tools/call），供 Claude/Cursor 直连 / MCP JSON-RPC for Claude/Cursor |
| GET | `/health` | 存活 + 组件健康（含 `llm_auth: ok/rejected/unknown` 信息位）/ Liveness + component health incl. llm_auth info field |
| GET | `/ready` | 就绪（尊重 embedding health()）/ Readiness (respects embedding health) |
| GET | `/metrics` | Prometheus RED 指标（按 endpoint + kb_id 的请求数/延迟直方图/错误数）/ Prometheus RED metrics |
| GET | `/console` | 内置控制台（`/shell/fusion-rag/` 别名）/ Built-in console (alias of /shell/fusion-rag/) |

## Python 内嵌用法 / Embedded Python usage

除 HTTP 外，核心组件可作为库直接使用（同一进程内嵌时注意单例 embedding 客户端与 venv 依赖）。模块级参考见 [api.md](api.md)。

Core components are also usable as a library (mind the singleton embedding client and venv deps when embedding in-process). Module-level reference: [api.md](api.md).
