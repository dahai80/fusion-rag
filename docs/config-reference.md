---
doc_id: fusion-rag/config-reference
product: fusion-rag
class: platform
entry: service
doc_type: config-reference
title: {zh: "Fusion-RAG 配置参考", en: "Fusion-RAG Configuration Reference"}
audience: engineer
status: draft
version: 0.8.21
updated: 2026-11-07
owner: "@dahai"
source_of_truth:
  - README.md
  - start.sh
  - fusion_rag/api/server.py
  - fusion_rag/api/app_state.py
---

# Fusion-RAG 配置参考 / Configuration Reference

全部配置经环境变量注入，无配置文件；运行时可读视图：`GET /kb/config`（需认证）。key 解析复用 fusion-core 的 `fusion_core.config`（端口 `service_port("rag")`、MLX key `client_key()` 别名链 `FUSION_CLIENT_KEY` -> `FUSION_MLX_API_KEY` -> `MLX_API_KEY`、数据根 `service_data_dir`）。

All configuration is env-driven; there is no config file. Runtime view: `GET /kb/config` (auth required). Key resolution reuses fusion-core's config helpers (port via `service_port("rag")`, MLX key via the `client_key()` alias chain `FUSION_CLIENT_KEY` -> `FUSION_MLX_API_KEY` -> `MLX_API_KEY`, data root via `service_data_dir`).

## 服务与服务发现 / Service and discovery

| Variable | Default | 说明 / Description |
|----------|---------|--------------------|
| `FUSION_RAG_PORT` | `11436` | 监听端口 / listen port |
| `FUSION_RAG_HOST` | `127.0.0.1` | 监听地址；绑 `0.0.0.0` 需在 port-registry 说明理由 / bind address; 0.0.0.0 needs a documented reason in the registry |
| `FUSION_REGISTRY` | `~/fusion/architecture/port-registry.yaml` | start.sh 起服前校验端口注册；未注册拒绝启动 / checked by start.sh before serving |
| `FUSION_SKIP_REGISTRY_CHECK` | (empty) | `1` 旁路注册校验 / bypass the registry check |
| `FUSION_RAG_DATA_DIR` | `~/.fusion-rag` | 数据根目录（stores、auth/graph/cache SQLite 派生路径）/ data root |

## fusion-mlx 与 embedding / fusion-mlx and embedding

| Variable | Default | 说明 / Description |
|----------|---------|--------------------|
| `FUSION_MLX_URL` | `http://127.0.0.1:11432/v1` | fusion-mlx（网关）地址 / fusion-mlx base URL |
| `FUSION_RAG_EMBED` | `BGE-M3` | embedding 模型名 / embedding model |
| `FUSION_RAG_EMBED_LOCAL_FALLBACK` | (on) | `0` 关闭本地 sentence-transformers 兜底，MLX 不可达时快速失败（503，约 2.5s）而非请求内冷加载模型 / `0` disables the local fallback so embed fails fast with a retryable 503 |
| `FUSION_RAG_EMBED_CACHE_TTL` | `604800` | embedding 缓存条目 TTL 秒（7 天）/ embedding cache TTL seconds |
| `FUSION_RAG_EMBED_CACHE_MAX_ENTRIES` | `100000` | 缓存行数上限（LRU 驱逐）/ max cache rows before LRU eviction |
| `FUSION_RAG_SKIP_STARTUP_PROBE` | (empty) | `1` 跳过 boot 期 MLX 可达性探测 / skip the boot-time MLX reachability probe |
| `FUSION_RAG_OCR_URL` | (empty) | OCR 服务（fusion-drawing-parser）地址；为空时扫描 PDF 显式失败 / OCR service base URL; scanned PDFs fail visibly when empty |
| `FUSION_RAG_OCR_API_KEY` | (empty) | OCR 服务的 Bearer token / bearer token for the OCR service |
| `FUSION_RAG_OCR_TIMEOUT` | `300` | 单个 OCR 任务轮询预算（秒）/ per-task OCR poll budget in seconds |
| `FUSION_RAG_OCR_TENANT_ID` | (empty) | 无请求租户时（后台 watch 摄取）的 `X-Tenant-Id` 兜底 / tenant header fallback for background ingestion |
| `FUSION_RAG_LOW_OCR_CONFIDENCE` | `0.85` | 低于此 OCR 置信度的 chunk 在结果中标记 `low_confidence` / chunks below this confidence get flagged |

## 认证与租户 / Auth and tenancy

| Variable | Default | 说明 / Description |
|----------|---------|--------------------|
| `FUSION_RAG_API_KEY` | (empty) | 本服务 API key 认证；空 = 关闭 / service API key; empty = auth off |
| `FUSION_RAG_AUTH_BACKEND` | `apikey` | `apikey` 或 `none` / auth backend switch |
| `FUSION_RAG_REQUIRE_GATEWAY` | (empty) | `1` 强制 `/kb/*` 经网关（`X-Fusion-Route: gateway-decision`）并按 `X-Fusion-Tenant` 隔离 KB / require gateway proof + tenant header scoping |
| `FUSION_RAG_REQUIRE_IDENTITY` | (empty) | `1` 用 fusion-identity JWT（`tid` claim）权威解析租户；identity 不可达时 fail-closed（401）/ authoritative tenant resolution via fusion-identity JWT; fail-closed when the identity service is unreachable |
| `FUSION_IDENTITY_URL` | `http://127.0.0.1:11470` | fusion-identity 地址 / fusion-identity base URL |
| `FUSION_IDENTITY_SERVICE_TOKEN` | (empty) | identity 模式必需；缺失即 fail-closed / required in identity mode; missing = fail-closed |

约束 / constraints: 两个租户开关默认关闭；`X-Fusion-Tenant` 单独使用可被直连方伪造，生产多租户必须开 identity 模式。gateway/identity 模式下 `POST /kb/auth/keys` 返回 409（单机 auth.db 不支持水平扩容），请用共享 `FUSION_RAG_API_KEY` 或 identity 签发凭据。

Both tenancy flags default off; the `X-Fusion-Tenant` header alone is forgeable by direct-port callers, so production multi-tenant requires identity mode. In gateway/identity mode local key minting returns 409 — use the shared `FUSION_RAG_API_KEY` or identity-issued credentials.

## 存储后端 / Store backends

| Variable | Default | 说明 / Description |
|----------|---------|--------------------|
| `FUSION_RAG_STORE_BACKEND` | `local` | `local`（LanceDB）/ `fusion-store`（HNSW，需 `pip install -e ../fusion-store`）/ `qdrant`（需 `pip install 'fusion-rag[qdrant]'`）/ `remote` / vector backend switch |
| `FUSION_RAG_STORES_DIR` | `~/.fusion-rag/stores` | 每 KB store 根目录 / per-KB store root |
| `FUSION_RAG_REMOTE_ENDPOINT` | (empty) | `remote` 后端必填：远端 fusion-rag 节点地址 / required for the remote backend |
| `FUSION_RAG_REMOTE_API_KEY` | (empty) | 发往远端的 `X-API-Key` / key sent to the remote node |
| `FUSION_RAG_REMOTE_KB_ID` | (derived) | 远端 KB id，默认取本节点 kb_id / remote KB id, defaults to the local one |
| `FUSION_RAG_REMOTE_TIMEOUT` | `30` | 远端单请求超时秒 / per-request remote timeout seconds |
| `QDRANT_URL` | `:memory:` | qdrant 后端端点；`:memory:` 为进程内模式 / qdrant endpoint; `:memory:` is in-process |
| `QDRANT_API_KEY` | (empty) | 远端 qdrant key / key for a remote qdrant server |
| `QDRANT_COLLECTION_PREFIX` | `tenant_id_` | 按租户隔离的集合名前缀 / per-tenant collection name prefix |

约束 / constraint: `local` 后端的向量检索为暴力扫描（无 ANN），单库约 10 万 chunk 内合适；更大语料换 `fusion-store` 或 `qdrant`。

The local backend's vector search is a brute-force scan (no ANN index), comfortable up to roughly 100k chunks per KB; switch backends for larger corpora.

## 摄取与检索 / Ingestion and retrieval

| Variable | Default | 说明 / Description |
|----------|---------|--------------------|
| `FUSION_RAG_INGEST_ROOTS` | (empty) | 冒号分隔的允许读取根目录；空 = 不设限（local-first）。暴露端口前务必设置以阻断路径穿越 / colon-separated allowed roots; empty = unconfined. Set before exposing the port |
| `FUSION_RAG_SCAN_MAX_FILES` | `1000` | 单次 scan 摄取文件上限 / max files per scan |
| `FUSION_RAG_WATCH_CAP` | `16` | 进程内目录 watch 上限 / max concurrent watches per process |
| `FUSION_RAG_MAX_BATCH_FILES` | (empty) | 批量上传上限 / batch upload cap |
| `FUSION_RAG_MAX_CONTENT_CHARS` | (empty) | 单文档内容字符上限 / per-document content char cap |
| `FUSION_RAG_FETCH_K_MULTIPLIER` | `4` | 过滤检索的过取系数 / over-fetch factor for filtered search |
| `FUSION_RAG_TOKEN_BUDGET` | `8192` | 多轮 RAG 每轮 token 预算 / per-turn token budget for multi-turn RAG |
| `FUSION_RAG_MAX_HISTORY_TURNS` | `10` | RAG 历史保留轮数 / max conversation turns kept |
| `FUSION_RAG_SYSTEM_PROMPT` | (built-in) | 覆盖 RAG 问答系统提示词 / override the RAG system prompt |

## Rerank / Rerank

| Variable | Default | 说明 / Description |
|----------|---------|--------------------|
| `FUSION_RAG_RERANK_BACKEND` | `llm` | `llm`（旧版提示打分）或 `cross_encoder`（fusion-mlx `POST /v1/rerank`）/ legacy prompt-scoring or real cross-encoder via fusion-mlx |
| `FUSION_RAG_RERANK_MODEL` | (empty) | cross-encoder 模型名（如 `bge-reranker-v2-m3`）；设置后 `/search`、`/ask` 默认开 rerank / cross-encoder model; when set, rerank defaults on |
| `FUSION_RAG_RERANK_TOP_N` | `20` | 送入 rerank 的候选池大小 / candidate pool size before truncation to top_k |

## 日志与轨迹 / Logging and trajectories

| Variable | Default | 说明 / Description |
|----------|---------|--------------------|
| `FUSION_RAG_LOG_DIR` | `./logs` | 应用日志目录（RotatingFileHandler 10MB x 5）/ app log dir (10MB x 5 rotating) |
| `FUSION_RAG_LOG_LEVEL` | (empty) | 日志级别 / log level |
| `FUSION_RAG_LOG_FORMAT` | `text` | `text` 或 `json`（每行一个 JSON 对象，供 ELK/Loki）/ text or json line format |
| `FUSION_TRAJECTORY_DIR` | `~/.fusion/trajectories/rag` | 检索轨迹 JSONL 输出目录（D1 飞轮）/ retrieval trajectory JSONL output dir |
| `FUSION_RAG_TRAJECTORY_MAX_MB` | `100` | 轨迹文件轮转阈值 / trajectory rotation threshold |
| `FUSION_RAG_TRAJECTORY_KEEP` | `5` | 轮转保留份数 / rotated files to keep |
| `FUSION_RAG_AUDIT_RETENTION_DAYS` | `30` | 审计保留天数（0 = 永久）/ audit retention days (0 = forever) |

## 云端兜底 / Cloud embedding fallback

| Variable | Default | 说明 / Description |
|----------|---------|--------------------|
| `FUSION_RAG_FALLBACK_URL` | (empty) | 云端 embedding 兜底地址；默认空 = 完全离线 / cloud fallback URL; empty = fully offline |
| `FUSION_RAG_FALLBACK_API_KEY` | (empty) | 兜底服务 key / key for the fallback service |

约束 / constraint: 默认部署零外呼。启用兜底后，数据（查询文本）会发往该 URL——离线合规场景保持为空。

A default deployment makes zero outbound calls. With a fallback configured, query text leaves the device — keep it empty for offline-compliance deployments.
