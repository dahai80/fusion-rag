---
doc_id: fusion-rag/get-started
product: fusion-rag
class: platform
entry: service
doc_type: get-started
title: {zh: "Fusion-RAG 快速上手", en: "Fusion-RAG Get Started"}
audience: engineer
status: draft
version: 0.8.21
updated: 2026-11-07
owner: @dahai
source_of_truth:
  - README.md
  - start.sh
  - pyproject.toml
---

# Fusion-RAG 快速上手 / Fusion-RAG Get Started

## 前置条件 / Prerequisites

- macOS Apple Silicon（M1–M5），Python 3.12+（monorepo 统一 `.python-version`）
- fusion-mlx 已启动（直连 `:11434` 或经 fusion-gateway `:11432`）
- 已有 monorepo venv：`source ~/fusion/.venv/bin/activate`

macOS Apple Silicon (M1-M5), Python 3.12+, fusion-mlx running (`:11434` direct or `:11432` via fusion-gateway), and the monorepo venv activated.

## 1. 安装 / Install

```bash
cd ~/fusion/fusion-rag
source ~/fusion/.venv/bin/activate
pip install -e ".[test]"
```

## 2. 启动服务 / Start the service

```bash
./start.sh start     # 默认 http://127.0.0.1:11436
./start.sh status
```

`start.sh` 会：激活 monorepo venv -> 校验 port-registry 注册（未注册拒绝启动，`FUSION_SKIP_REGISTRY_CHECK=1` 可旁路）-> `nohup python3 -m fusion_rag.api.server`。日志在 `logs/stdout.log`（bootstrap）与 `logs/fusion-rag.log`（应用，10MB x 5 轮转）。

start.sh activates the venv, verifies the port-registry entry (refuses to start if unregistered), then launches `python3 -m fusion_rag.api.server`. Bootstrap logs go to `logs/stdout.log`; app logs to `logs/fusion-rag.log` (10MB x 5 rotating).

如果 fusion-mlx 暂未启动，服务会启动失败（boot 期可达性探测，exit 1）；先起 MLX，或临时 `FUSION_RAG_SKIP_STARTUP_PROBE=1`。

If fusion-mlx is down the boot-time probe fails and the service exits 1; start MLX first, or bypass temporarily with `FUSION_RAG_SKIP_STARTUP_PROBE=1`.

## 3. 建库 + 摄取 + 检索 / Create, ingest, search

```bash
BASE=http://127.0.0.1:11436

# create a KB
curl -s -X POST $BASE/kb/bases -H 'Content-Type: application/json' \
  -d '{"name": "demo", "description": "first KB"}'
# note the "id" field in the response

# ingest inline text (markdown goes straight to chunk + index)
curl -s -X POST $BASE/kb/bases/demo/documents/ingest -H 'Content-Type: application/json' \
  -d '{"doc_name": "notes.md", "content_type": "markdown", "content": "# Notes\n\nFusion-RAG is an offline vector KB backend."}'

# hybrid semantic search
curl -s -X POST $BASE/kb/bases/demo/search -H 'Content-Type: application/json' \
  -d '{"query": "offline vector knowledge base", "hybrid": true, "top_k": 5}'
```

默认无 API key 认证（`FUSION_RAG_API_KEY` 为空即关闭）；需要认证时设置后，所有 `/kb/*` 请求加 `X-API-Key` 头。带 API key 的等价示例：

Auth is off while `FUSION_RAG_API_KEY` is empty; once set, every `/kb/*` call needs the `X-API-Key` header:

```bash
curl -s $BASE/kb/bases -H "X-API-Key: $FUSION_RAG_API_KEY"
```

## 4. RAG 问答 / RAG Q&A

```bash
curl -s -X POST $BASE/kb/bases/demo/ask -H 'Content-Type: application/json' \
  -d '{"question": "What is Fusion-RAG?", "hybrid": true}' | python3 -m json.tool
```

`/ask` 需要 fusion-mlx 提供 chat 模型（默认 `qwen3.5-9b`，可用 `model` 字段覆盖）；响应含 `answer` 与逐 chunk `citations[]`。

`/ask` uses a chat model served by fusion-mlx (default `qwen3.5-9b`, override via `model`); the response carries `answer` plus per-chunk `citations[]`.

## 5. 内置控制台 / Built-in console

浏览器打开 `http://127.0.0.1:11436/console`（经网关则是 `http://127.0.0.1:11432/shell/fusion-rag/`）：KB 管理、文本摄取、检索测试、Graph/版本/模板、ACL/审计/API key 管理都在里面，无需写代码即可验证以上流程。

Open `http://127.0.0.1:11436/console` (behind the gateway: `http://127.0.0.1:11432/shell/fusion-rag/`) to exercise KB management, ingestion, search, graph/versions/templates and admin views without writing code.

## 6. 停止 / Stop

```bash
./start.sh stop      # SIGTERM，30s 优雅排水后强杀
```

## 下一步 / Next steps

- 完整 HTTP 接口：[api-reference.md](api-reference.md)
- 环境变量与多租户开关：[config-reference.md](config-reference.md)
- 部署形态（Docker/systemd）：[../DEPLOYMENT.md](../DEPLOYMENT.md)
- 故障排查：[troubleshooting.md](troubleshooting.md)
