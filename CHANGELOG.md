---
doc_id: fusion-rag/changelog
product: fusion-rag
class: platform
entry: service
doc_type: changelog
title: {zh: "Fusion-RAG 变更日志", en: "Fusion-RAG Changelog"}
audience: engineer
status: draft
version: 0.8.21
updated: 2026-11-07
owner: @dahai
source_of_truth:
  - README.md
  - docs/benchmarks/README.md
---

# Fusion-RAG 变更日志 / Changelog

逐版本详细说明见 `README.md` 的 "What's New" 章节（以 README 为准）；本文件按版本聚合面向使用者的行为变化。/ Per-version detail lives in the README "What's New" section, which is authoritative; this file aggregates user-facing behavior changes.

## 0.8.21 (2026-10)

- 控制台补齐 API-only 能力的 UI（issue #113）：KB 详情新增 Graph / Versions / Templates 标签页；新增 `#/admin` 页（ACL 规则、审计日志 + 导出、API key 管理，铸造的 key 仅展示一次）。
- `/kb/auth/keys` 在控制台子路径 `/shell/fusion-rag/` 下同步挂载（修复网关代理后 404，#109 契约保持页面相对路径调用）。472 tests passing。
- Console UI now covers graph, versions, templates, ACL, audit and API keys; the auth router is also mounted under the console subpath so it works behind the gateway proxy.

## 0.8.20

- 控制台按 fusion-web-shell 规范重建：React 18 + TypeScript + Tailwind，`@fusion/web-ui` tokens，双语 SDK 桥，四态视图；删除 KB 改走 ConfirmDialog（沙箱 iframe 阻断原生 confirm 导致按钮失效的问题）。构建产物提交于 `fusion_rag/console/`。
- Console rebuilt to the union web UI spec; deletion goes through a typed ConfirmDialog because sandboxed iframes block native confirm().

## 0.8.19

- 控制台兼容网关直通代理（issue #109）：API base 改为页面相对路径；服务端在 `/shell/fusion-rag/` 子路径下挂载完整 `/kb` 路由 + `/health`，注册顺序先于静态挂载。
- Console works behind the gateway pass-through proxy via page-relative API bases and a full /kb mount under the console subpath.

## 0.8.17 / 0.8.16

- 本地 embedding 兜底：共享单次加载（双检锁）+ 新开关 `FUSION_RAG_EMBED_LOCAL_FALLBACK=0` 快速失败（503，约 2.5s）；`/ready` 尊重 `health()` 结果（issue #106）。
- 修复 boot 期 MLX 可达性探测从未真正执行的问题（协程未 await）；MLX 不可达时响亮报错并 exit 1（issue #104）。
- Local embedding fallback now loads once under a lock with an opt-in fail-fast mode; the boot-time MLX probe actually runs and exits 1 on failure.

## 0.8.15

- gateway/identity 模式下 `POST /kb/auth/keys` 返回 409（issue #102）：单机 auth.db 与"无状态水平扩容"契约冲突，规模化部署改用共享 `FUSION_RAG_API_KEY` 或 identity 凭据。
- Local key minting is refused in scaled mode; use config-plane keys or identity-issued credentials.

## 0.8.12 / 0.8.13

- 配置解析收敛到 fusion-core（issue #96）：`client_key()` 别名链、`service_port("rag")`、`service_data_dir`；`FUSION_RAG_DATA_DIR` 为数据根覆盖。
- start.sh 注册自检（issue #98）：端口未在 port-registry 注册则拒绝启动，`FUSION_SKIP_REGISTRY_CHECK=1` 旁路，注册文件缺失仅告警。
- Config resolution moved to fusion-core; start.sh now verifies its port-registry entry before serving.

## 0.8.11

- 精确 chunk-id 删除（issue #94）：同内容文档共享物理 chunk 时删除互不误伤；四个后端 + 远端 `/store/*` 面统一 `delete_by_chunk_ids`。
- LLM 鉴权显性化（issue #93）：上游 401/403 -> `/ask` 502 "LLM credential rejected"、`/health` 增加 `llm_auth` 位；进程标题 `fusion-rag serve :11436`（issue #92）。
- Exact chunk-id deletion across all backends; LLM auth failures surface distinctly; process title set via setproctitle.

## 0.8.10

- inline 摄取 pdf/docx/epub 的 `content` 按 base64 二进制走完整解析管道（含扫描件 OCR），不再把 base64 文本当正文索引；非法 base64 返回 400（issue #90）。
- Inline binary ingest is base64-decoded through the full parse pipeline; invalid base64 returns 400.

## 0.8.9 及更早 / earlier

见 README "What's New" 各版本小节。/ See the README "What's New" sections.
