---
doc_id: fusion-rag/troubleshooting
product: fusion-rag
class: platform
entry: service
doc_type: troubleshooting
title: {zh: "Fusion-RAG 故障排查", en: "Fusion-RAG Troubleshooting"}
audience: engineer
status: draft
version: 0.8.21
updated: 2026-11-07
owner: "@dahai"
source_of_truth:
  - README.md
  - start.sh
  - fusion_rag/api/server.py
  - docs/benchmarks/README.md
---

# Fusion-RAG 故障排查 / Troubleshooting

排查入口：`./start.sh status`（进程 + health）、`logs/stdout.log`（bootstrap）、`logs/fusion-rag.log`（应用，10MB x 5 轮转）。以下按症状定位。

First checks: `./start.sh status`, `logs/stdout.log` (bootstrap), `logs/fusion-rag.log` (app). Symptoms below.

## 启动失败 / Startup failures

### port 11436 for fusion-rag not registered — refusing to start

start.sh 的注册自检（issue #98）发现 port-registry 缺条目。修复：在 `~/fusion/architecture/port-registry.yaml` 的 `fusion-rag:` 下确认 `port: 11436`；确需旁路（独立部署）用 `FUSION_SKIP_REGISTRY_CHECK=1 ./start.sh start`。

The registry self-check found no port-registry entry. Fix the registry entry, or bypass for standalone deploys with `FUSION_SKIP_REGISTRY_CHECK=1`.

### 启动即退出，日志见 MLX 探测 ERROR / exits at boot with an MLX probe ERROR

boot 期可达性探测（issue #104）探测 fusion-mlx `GET /models` 失败即 exit 1。修复：先启动 fusion-mlx（`~/fusion/fusion-mlx/start.sh start`）或 fusion-gateway；MLX 稍后可用的场景用 `FUSION_RAG_SKIP_STARTUP_PROBE=1`。

The boot-time probe failed against fusion-mlx. Start fusion-mlx (or the gateway) first, or set `FUSION_RAG_SKIP_STARTUP_PROBE=1` when MLX comes up later.

### ModuleNotFoundError: No module named 'fusion_core'

未激活 monorepo venv。start.sh 会自动找 `~/fusion/.venv`；手工启动时先 `source ~/fusion/.venv/bin/activate`。

The monorepo venv was not activated. start.sh locates `~/fusion/.venv` automatically; source it manually when launching by hand.

## 检索与 embedding / Search and embedding

### /search 返回 503，日志有 EmbeddingError / fast 503 on search

设置了 `FUSION_RAG_EMBED_LOCAL_FALLBACK=0` 时 MLX 不可达会快速失败（设计行为，约 2.5s 而非分钟级冷加载）。修复 fusion-mlx 后自动恢复；想恢复请求内兜底就去掉该开关。

With `FUSION_RAG_EMBED_LOCAL_FALLBACK=0` an unreachable MLX fails fast by design. Restore fusion-mlx; remove the flag to re-enable in-request fallback.

### 搜索挂起数分钟 / search hangs for minutes

旧的本地兜底冷加载路径（BGE-M3 在 MPS 上约 2-4 分钟）。0.8.17 起并发请求共享一次加载（双检锁）；若仍出现，确认版本并考虑 `FUSION_RAG_EMBED_LOCAL_FALLBACK=0` 强制快速失败。

Cold local-fallback loads took minutes; since 0.8.17 concurrent callers share one load. If it still happens, verify the version or force fail-fast with `FUSION_RAG_EMBED_LOCAL_FALLBACK=0`.

### /ask 返回 502 "LLM credential rejected"

fusion-mlx 对 chat 请求返回 401/403（issue #93）：MLX key 过期或不匹配。更新 `FUSION_CLIENT_KEY`/`FUSION_MLX_API_KEY`（或 `~/.fusion-mlx/settings.json`）；成功生成一次后 `/health` 的 `llm_auth` 位自动恢复 `ok`。日志只出现 key 的 blake2b 哈希前缀，不会泄漏明文。

fusion-mlx rejected the chat credential. Rotate the key via the fusion-core alias chain or `~/.fusion-mlx/settings.json`; the `llm_auth` health field clears after the next successful generation.

### 检索结果为空 / empty results

按序检查：`GET /kb/bases/{kb_id}/stats` 确认 chunk 数 > 0 -> 确认 `status_filter`（默认 `active`，已作废文档被剔除）-> 确认 embedding 模型与建库时一致（换模型 = 换向量空间，需重建索引）-> 用 `/kb/bases/{kb_id}/bench` 跑基准对比通道表现。

Check in order: chunk count via stats, `status_filter` (deprecated docs are dropped by default), embedding-model consistency with the KB (a model change requires reindexing), then run the bench endpoint to compare channels.

### 召回质量差 / poor recall

关键词型语料 BM25 占优、释义型查询伤 BM25——这是通道特性，不是缺陷。基准与消融结论（RRF recall@10 0.9212，2026-10-25）见 `docs/benchmarks/README.md`；改进路径：开 `hybrid`（RRF）、设 `FUSION_RAG_RERANK_MODEL` 启用 cross-encoder、调整 `FUSION_RAG_RERANK_TOP_N`。

Keyword-heavy corpora favor BM25, paraphrased queries hurt it — channel characteristics, not defects. Enable hybrid (RRF), set a cross-encoder model, and raise `FUSION_RAG_RERANK_TOP_N`. Baseline numbers: docs/benchmarks/README.md.

## 摄取 / Ingestion

### 扫描 PDF 报 "no OCR service configured"

未配置 OCR（issue #84 的显式失败设计）。设置 `FUSION_RAG_OCR_URL`（如 fusion-drawing-parser 地址）与可选 `FUSION_RAG_OCR_API_KEY`；后台 watch 摄取还需 `FUSION_RAG_OCR_TENANT_ID`。

No OCR service is configured — this fails visibly by design. Set `FUSION_RAG_OCR_URL` (plus the key and, for background watch ingestion, the tenant fallback header).

### inline ingest pdf/docx 返回 400

`content` 必须是合法 base64 二进制（issue #90）；文本类类型（text/markdown/html/csv）不要 base64。

For pdf/docx/epub the `content` field must be valid base64 binary; text-ish types must not be base64.

### scan 报文件数超限 / scan hits a file cap

单次 scan 默认上限 1000 个文件（`FUSION_RAG_SCAN_MAX_FILES`）。调大该值，或分批扫描。

One scan ingests at most 1000 files by default; raise `FUSION_RAG_SCAN_MAX_FILES` or scan in batches.

### 路径被拒 / path rejected

`FUSION_RAG_INGEST_ROOTS` 已设置而目标路径不在允许根内（防路径穿越设计）。把根目录加进冒号分隔列表。

`FUSION_RAG_INGEST_ROOTS` confinement is active and the path is outside the allowed roots; add the root to the colon-separated list.

## 认证与租户 / Auth and tenancy

### 401 / 403 on /kb/*

按场景：设置过 `FUSION_RAG_API_KEY` 但请求缺 `X-API-Key` -> 补头；`FUSION_RAG_REQUIRE_GATEWAY=1` 但缺 `X-Fusion-Route: gateway-decision` -> 必须经网关；`FUSION_RAG_REQUIRE_IDENTITY=1` 下 JWT 缺失/失效/identity 不可达 -> 一律 401（fail-closed，不回退请求头）。`X-Fusion-Tenant` 与 JWT `tid` 不一致返回 401，属于防伪造设计。

Missing API key header, missing gateway proof, or an invalid/absent JWT with an unreachable identity service all deny access (fail-closed). A `X-Fusion-Tenant` header that disagrees with the JWT `tid` is rejected as a forgery attempt.

### POST /kb/auth/keys 返回 409

gateway/identity 模式下拒绝本机发 key（issue #102：单机 auth.db 不能横向扩容）。改用共享 `FUSION_RAG_API_KEY` 或 identity 签发凭据。

Key minting is refused in gateway/identity mode because the local auth.db does not scale horizontally; use shared config-plane credentials instead.

### 多租户下看不到别人的 KB / KBs invisible across tenants

设计行为：KB 打了租户戳并按请求租户过滤（issue #61/#68）。确认请求携带正确租户（identity 模式看 JWT `tid`），或改用对应租户的凭据。

KBs are tenant-stamped and scoped — by design. Verify the request tenant (the JWT `tid` in identity mode).

## 运维 / Operations

### 磁盘增长 / disk growth

四个常见源：embedding 缓存 SQLite（TTL 604800s / 10 万行 LRU）、审计（默认 30 天，`FUSION_RAG_AUDIT_RETENTION_DAYS=0` 永久）、轨迹 JSONL（100MB x 5 轮转）、快照（硬链接，删除 KB 时清理）。逐一核对 `~/.fusion-rag` 与 `FUSION_TRAJECTORY_DIR`。

Usual suspects: the embedding-cache SQLite (7-day TTL, 100k-row LRU), audit rows (30-day retention unless set to 0), trajectory JSONL (100MB x 5 rotation), and snapshots (hard links, cleaned with the KB).

### 进程名是裸 python / bare python process name

0.8.12 起用 setproctitle 设为 `fusion-rag serve :11436`；未装 setproctitle 时降级为裸进程名（软依赖）。`pip install setproctitle` 后重启。

The process title needs setproctitle (soft dependency); install it and restart.

### 谁在调用我 / who is calling me

`GET /metrics`（需认证）按 endpoint + kb_id 给出请求计数、延迟直方图、错误数。控制台 `#/admin` 有审计与 API key 视图。

`GET /metrics` exposes per-endpoint/per-KB RED metrics; the console admin page shows audit and API-key views.

## 仍未解决 / Still stuck

- 查 README 对应章节与 `git log --oneline` 最近变更
- 按新服务清单核对：port-registry、进程标题、health 语义（AGENTS.md）
- 带 `FUSION_RAG_LOG_FORMAT=json` 复现，附结构化日志提 issue
- Re-run with `FUSION_RAG_LOG_FORMAT=json` and attach the structured logs when filing an issue.
