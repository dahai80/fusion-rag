<div align="center">

# Fusion-RAG

**Apple Silicon 原生离线向量知识库底座**

Fusion-MLX 生态的统一本地向量知识库服务——默认离线，数据不出设备（云端 Embedding 回退 `FUSION_RAG_FALLBACK_URL` 需显式开启，默认关闭）。

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-40-success.svg)](tests/)

[English](README.md) · [快速开始](#快速开始) · [API 参考](#api-参考) · [架构](#架构) · [文档](docs/)

</div>

---

## 为什么选择 Fusion-RAG？

| 特性 | Fusion-RAG | Dify RAG | LangChain RAG |
|------|-----------|----------|---------------|
| **MLX 原生** | ✅ fusion-mlx API | ❌ Ollama/云端 | ❌ 云端 API |
| **Apple Silicon 优化** | ✅ LanceDB + MLX | ❌ | ❌ |
| **多知识库隔离** | ✅ | ✅ | ❌ |
| **代码专属分片** | ✅ | ❌ | ❌ |
| **本地离线** | ✅ 默认离线 | ⚠️ 部分 | ❌ |
| **零 API 费用** | ✅ | ❌ | ❌ |

**一句话：** Fusion-RAG 是 Fusion-MLX 生态的统一本地向量知识库底座——所有 Embedding 通过 fusion-mlx HTTP API 调用，不直接调用 MLX。

---

## 快速开始

### 前置条件

- macOS Apple Silicon (M1–M5)
- Python 3.12+
- [fusion-mlx](https://github.com/dahai80/fusion-mlx) 运行在 `localhost:11432`

### 安装

```bash
git clone https://github.com/dahai80/fusion-rag.git
cd fusion-rag
pip install -e ".[test]"
```

### 启动服务

```bash
./start.sh start
# Fusion-RAG 运行在 http://127.0.0.1:11436
```

### 最小示例

```python
import asyncio
from fusion_rag import DocumentParser, Chunker
from fusion_rag.embed.client import EmbeddingClient

async def main():
    # 1. 解析文档
    parser = DocumentParser()
    result = await parser.parse("README.md")
    print(f"解析完成: {result.file_name} ({result.chars} 字符)")

    # 2. 智能分片
    chunker = Chunker(strategy="semantic")
    chunks = await chunker.chunk(result)
    print(f"分片数: {len(chunks)}")

    # 3. 通过 fusion-mlx 生成向量
    embed = EmbeddingClient(model="BGE-M3")
    vectors = await embed.embed_batch([c.text for c in chunks])
    print(f"向量数: {len(vectors)}")

asyncio.run(main())
```

---

## API 参考

Fusion-RAG 通过 `/kb/*` 提供 REST API。

### 知识库管理

| 方法 | 端点 | 说明 |
|------|------|------|
| GET | `/kb/bases` | 列出所有知识库 |
| POST | `/kb/bases` | 创建知识库 |
| GET | `/kb/bases/{id}` | 获取知识库详情 |
| DELETE | `/kb/bases/{id}` | 删除知识库 |
| GET | `/kb/bases/{id}/stats` | 获取知识库统计 |

### 文档操作

| 方法 | 端点 | 说明 |
|------|------|------|
| POST | `/kb/bases/{id}/documents` | 上传并索引文档 |
| POST | `/kb/bases/{id}/documents/ingest` | 内联内容入库：`text`/`markdown`/`html`/`csv` 直接入库；`pdf`/`docx`/`epub` 的 `content` 为 base64 二进制，走完整解析管线（含扫描件 OCR 路径），非法 base64 返回 400——issue #90 |
| PATCH | `/kb/bases/{id}/documents/{doc_id}` | 修改文档生命周期元数据（`status`：active/deprecated、`superseded_by`）——issue #86 |
| POST | `/kb/bases/{id}/scan` | 扫描目录批量索引 |

### 搜索与问答

| 方法 | 端点 | 说明 |
|------|------|------|
| POST | `/kb/bases/{id}/search` | 语义向量检索（支持混合检索、重排、模板、status_filter） |
| POST | `/kb/bases/{id}/ask` | RAG 问答（含来源引用、逐块 citations、混合检索、重排） |
| POST | `/kb/bases/{id}/similar` | 近重复 / 版本漂移检索，支持文本或原始向量输入（issue #83） |

#### 搜索参数（#70 重排相关）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `rerank` | false | 启用重排（设置了 `FUSION_RAG_RERANK_MODEL` 时默认开启） |
| `rerank_backend` | 取自环境变量 | 重排阶段：`cross_encoder`（fusion-mlx `/v1/rerank`，真实交叉编码器如 `bge-reranker-v2-m3`）或 `llm`（传统提示词打分） |
| `rerank_model` | 取自环境变量 | 交叉编码器模型名（覆盖 `FUSION_RAG_RERANK_MODEL`） |
| `rerank_top_n` | 20 | 送入重排的候选池大小，重排后再截断到 `top_k`（提升召回） |
| `status_filter` | "active" | 文档状态过滤：`active`（默认，剔除已作废文档）、`deprecated`（仅已作废）、`all`（全部，已作废片段在 LLM 上下文中标记 `[已作废版本]`）——issue #86 |

### 系统

| 方法 | 端点 | 说明 |
|------|------|------|
| GET | `/kb/status` | 服务状态 |
| GET | `/health` | 健康检查 |

---

## 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    Fusion-RAG HTTP API (FastAPI)                   │
│  /kb/bases  /kb/bases/{id}/documents  /kb/bases/{id}/search     │
│  /kb/bases/{id}/ask  /kb/status  /health                        │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                      RAG 核心引擎                                │
│                                                                  │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────────┐  │
│  │ DocumentParser │  │    Chunker     │  │  KnowledgeBase    │  │
│  │ PDF/DOCX/MD/   │  │ 语义/固定/代码 │  │  管理器 (CRUD)    │  │
│  │ TXT/HTML/代码  │  │ 三种分片策略   │  │  + 隔离           │  │
│  └────────┬───────┘  └───────┬────────┘  └────────┬─────────┘  │
└───────────┼──────────────────┼────────────────────┼────────────┘
            │                  │                    │
┌───────────▼──────────────────▼────────────────────▼────────────┐
│                     存储层                                      │
│                                                                  │
│  ┌────────────────────────────┐  ┌────────────────────────────┐ │
│  │   VectorStore (LanceDB)    │  │   MetadataStore (SQLite)   │ │
│  │   向量存储与检索           │  │   文档/分片元数据         │ │
│  └────────────┬───────────────┘  └──────────────┬─────────────┘ │
└───────────────┼──────────────────────────────────┼───────────────┘
                │                                  │
┌───────────────▼──────────────────────────────────▼───────────────┐
│                    Embedding 客户端                              │
│  通过 HTTP API 调用 fusion-mlx /v1/embeddings                    │
│  不直接调用 MLX、mlx-lm 或 torch                                 │
└───────────────────────────────┬─────────────────────────────────┘
                                │ HTTP
┌───────────────────────────────▼─────────────────────────────────┐
│  fusion-mlx (/v1/embeddings, /v1/chat/completions)               │
│  Apple Silicon MLX 运行时 (Metal GPU)                           │
└─────────────────────────────────────────────────────────────────┘
```

### 核心模块

| 模块 | 文件 | 说明 |
|------|------|------|
| 知识库管理 | `engine/knowledge_base.py` | KB 增删改查、配置、持久化 |
| 文档解析 | `engine/document.py` | PDF/DOCX/MD/TXT/HTML/代码 |
| 智能分片 | `engine/chunker.py` | 语义/固定/代码专属分片 |
| Embedding | `embed/client.py` | 通过 fusion-mlx HTTP API 调用 |
| 向量存储 | `store/vector_store.py` | LanceDB 存储（懒加载） |
| 元数据存储 | `store/metadata_store.py` | SQLite 文档/分片追踪 |
| API 路由 | `api/routes.py` | FastAPI 端点 |
| 服务 | `api/server.py` | FastAPI 服务 |

---

## 数据层依赖

约束文档：[`architecture/fusion-db-conventions-1030.md`](../architecture/fusion-db-conventions-1030.md)（每节点单一共享 Qdrant 实例；不引入其他向量数据库 —— DB-8）。

### 何时需要 Qdrant

- **默认模式（`FUSION_RAG_STORE_BACKEND=local`）** —— 无需外部数据库。向量存于内嵌 LanceDB + SQLite（`~/.fusion-rag/stores/{kb_id}/`）。
- **Qdrant 后端（`FUSION_RAG_STORE_BACKEND=qdrant`）** —— 按租户集合隔离（issue #66）及依赖它的多租户部署必须使用。客户端安装：`pip install 'fusion-rag[qdrant]'`。

### 本地运行 Qdrant

规范实例地址：**`http://localhost:6333`（HTTP）**；gRPC 端口 `6334` 在约束文档中保留，但 **fusion-rag 不使用** —— 后端仅走 HTTP。Qdrant 无原生 Homebrew formula，本地运行需容器运行时（OrbStack 或 Docker）：

```bash
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant:v1.12.4
```

生产环境由 `fusion-supervisor` 的 `docker-compose.sidecar.yml` 提供同一实例（相同端口、相同版本 `v1.12.4`）—— 不要再起第二个 Qdrant。

### 连接与故障表现

连接通过 `QDRANT_URL` 配置（约束文档 DB-3）：

```bash
export FUSION_RAG_STORE_BACKEND=qdrant
export QDRANT_URL=http://localhost:6333
```

默认 `QDRANT_URL=:memory:` 为进程内临时 Qdrant —— 仅限开发/测试，重启后数据不保留。

Qdrant 不可达时的预期表现：**写入（ingest）响亮失败** —— upsert/连接错误向上传播为可见的 4xx/5xx，绝不静默成功。已知缺口（issue #100 中记录为后续项）：搜索路径目前把任何 `get_collection` 失败当作"集合不存在"并返回空结果，因此 Qdrant 不可达时搜索可能表现为空结果而非报错。结果异常为空时请查 `logs/fusion-rag.log` 中的 `QdrantBackend` 错误。

---

## 更新日志

### v0.8.15

- **横向扩容模式下禁用本地密钥创建（issue #102）** — 每实例 SQLite `auth.db` 与文档"无状态负载均衡后横向扩容"契约矛盾：节点 A 上签发的密钥在节点 B 上 401。决策：方案一（仅配置面凭证）。`FUSION_RAG_REQUIRE_GATEWAY=1` 或 `FUSION_RAG_REQUIRE_IDENTITY=1` 开启时，`POST /kb/auth/keys` 返回 409 拒绝创建单节点密钥；扩容部署使用共享 `FUSION_RAG_API_KEY` 环境变量或网关/identity 签发凭证。单租户默认行为不变。README 记录该约束及被否决的方案二（auth.db 迁 PG）。新增 4 个测试；双节点演练已验证。

### v0.8.14

- **数据层依赖文档（issue #100）** — README 新增"数据层依赖"章节：何时需要 Qdrant（仅 `qdrant` 后端/按租户隔离需要，默认本地 LanceDB 后端无需外部数据库）、本地运行方式（需容器运行时，`qdrant/qdrant:v1.12.4`）、规范连接地址 `QDRANT_URL=http://localhost:6333`（仅 HTTP —— gRPC 端口 6334 保留但 fusion-rag 未使用）、写入路径的响亮失败预期，以及约束文档 `architecture/fusion-db-conventions-1030.md` 的指引。issue 中记录后续项：Qdrant 不可达时搜索路径存在静默降级。

### v0.8.13

- **Config Phase 3：legacy 密钥变量名清理 + 启动前端口注册自检（issue #98）** — 代码中最后一处 legacy 环境变量名字面引用清除：密钥解析只读规范名 `FUSION_CLIENT_KEY`，legacy 别名链完全由 `fusion_core.config.client_key()` 内部处理（本仓库不再拼写，`scripts/check-env.py` 对 fusion-rag 报 0 条 legacy 告警 —— 2027-01-01 检查升级为 error 后依然保持绿）。`start.sh` 启动前校验自身在 `port-registry.yaml` 的注册项（config Phase 3.3）：端口未注册直接拒绝启动并给出明确提示（问题在启动时暴露，而非 CI），`FUSION_SKIP_REGISTRY_CHECK=1` 可跳过，注册表文件缺失则告警后继续（独立部署场景）。新增 4 个测试覆盖 拒绝/通过/跳过/文件缺失 四条路径，全量 473 通过。

### v0.8.12

- **配置解析收敛到 fusion_core 单点（issue #96）** — 删除全部手写密钥/环境变量解析，统一走 `fusion_core.config`（config Phase 2.2）：MLX API key 经 `client_key()` 解析（别名链 `FUSION_CLIENT_KEY` → `FUSION_MLX_API_KEY` → `MLX_API_KEY`；dev 缺 key = 空值 + 一次性告警；`FUSION_PROFILE=prod` 缺 key = RuntimeError 快速失败）；服务端口经 `service_port("rag")`（读 `FUSION_RAG_PORT`）；数据目录根经 `service_data_dir`（`FUSION_RAG_DATA_DIR` 可覆盖，默认保持 `~/.fusion-rag`）—— 覆盖 KB 存储根、watch 注册表、auth/graph/缓存/评估 SQLite 路径。`server.py` 用 `client_key() or resolve_api_key()` 保留本地优先的 `~/.fusion-mlx/settings.json` 回退；本地 `_mlx_api_key()` 之类解析函数全部删除（调用点保持一行）。不改环境变量名；`FUSION_RAG_API_KEY`（服务自身鉴权）不变。新增 11 个测试，全量 469 通过；`scripts/check-env.py` 对 fusion-rag 报 0 错误。

### v0.8.11

- **按 chunk-id 精确删除（issue #94）** — 删除一篇文档不再连带删除内容完全相同的另一篇文档的物理分片。两篇文档以相同 `doc_path` 入库（如同一 inline `doc_name` 提交两次）时共用同一个删除键，删除任一篇都会把两篇的向量和 BM25 行一起删掉。所有删除/替换/回滚路径改走新的 `delete_by_chunk_ids` 存储接口（local/qdrant/fusion-store/remote 四个后端均已实现；remote `/store/*` 面新增 `POST /delete_by_chunk_ids`），仅在旧版远端节点缺该接口时回退 `delete_by_doc`（有日志）。新增 5 个测试，全量 458 通过。
- **LLM 凭证问题显性化（issue #93）** — 失效的 `FUSION_MLX_API_KEY` 之前只会让每个 `/ask` 报笼统的 "upstream LLM unavailable"。现在 fusion-mlx 返回 401/403 时会清晰呈现：`/ask` 返回 502 "LLM credential rejected"，启动探针打显眼 ERROR（不退出 —— 检索不受影响），`/health` 报告信息性字段 `llm_auth: ok/rejected/unknown`。凭证轮换后一次成功生成会自动清除旧的 rejected 标记；日志仅用 blake2b 哈希前缀标识密钥，绝不落密钥明文。新增 11 个测试。
- **进程标题（issue #92）** — 服务进程在 `ps`/活动监视器中显示为 `fusion-rag serve :11436`，不再是裸 Python（setproctitle 软依赖）。`start.sh` 按 PID 文件停止，生命周期控制不受影响。新增 3 个测试。

### v0.8.10

- **内联二进制入库修复（issue #90）** — `POST /kb/bases/{id}/documents/ingest` 中 `content_type` 为 `pdf`/`docx`/`epub` 时，`content` 现在按 base64 编码的二进制数据处理：解码为临时文件并走完整 `DocumentParser` 解析管线（PDF 文本抽取、EPUB 章节、扫描件 OCR 路径 #84），而不是把 base64 文本本身分片入库污染检索。非法 base64 返回 400；`text`/`markdown`/`html`/`csv` 直接入库路径不变。新增 5 个测试（PDF/EPUB 解析往返、扫描件 OCR 路径、非法 base64、纯文本回归），全量 439 通过。

### v0.8.8

- **近重复检索（issue #83）** — 新增 `POST /kb/bases/{kb_id}/similar` 端点，用于近重复 / 版本漂移检测：支持 `text`（用与入库相同的服务级模型向量化，分数直接可比）或原始 `embedding` 向量查询；参数 `top_k`、`threshold`、`exclude_doc_id`（库内去重场景）、`snippet_chars`、`status_filter`。返回 `matches[]`，带完整证据锚点（doc_id/chunk_id/page/status）。
- **EPUB + 扫描件 OCR（issue #84）** — `DocumentParser` 新增 EPUB 解析（ebooklib，每个 XHTML 章节转为带标题的段落）。扫描版 PDF（有页无文本）不再被静默索引进空文档：改交给 fusion-drawing-parser OCR 服务（`FUSION_RAG_OCR_URL`、`FUSION_RAG_OCR_API_KEY`、`FUSION_RAG_OCR_TIMEOUT`），未配置 OCR 服务时显式报错。OCR 片段绕过分片器 + 上下文化（本身已是切片粒度），并携带逐块证据元数据。
- **OCR 证据元数据 + 引用（issue #85）** — 来自 OCR 的每个 `/search` 结果和 `/ask` 引用都携带 `ocr_confidence`、`low_confidence`（阈值 `FUSION_RAG_LOW_OCR_CONFIDENCE`，默认 0.85）以及 `slice_ref`（`task_id`/`slice_id`/`url`，可回链源切片图像）。`/ask` 额外返回逐块 `citations[]`（不去重）和顶层 `has_low_confidence` 标记。
- **文档 active/deprecated 状态（issue #86）** — `PATCH /kb/bases/{kb_id}/documents/{doc_id}` 无需重建索引即可翻转文档生命周期 `status`（`active`/`deprecated`，可选 `superseded_by`）。状态在检索时从 documents 表解析，文档作废后立即从默认 `/search`、`/ask`、`/similar` 中消失；`status_filter`（`active`/`deprecated`/`all`）按查询控制，`all` 保留的已作废片段在 LLM 上下文中标记 `[已作废版本]`。
- 新增 19 个测试（EPUB 解析 + OCR 往返 + 失败路径、/similar 校验/排除/状态、PATCH + 状态过滤 + /ask 引用），全量 428 通过，ruff 干净。

### v0.8.3

- **LLM 后端 URL + 鉴权透传（issue #72）** — 所有调用 LLM 的引擎（`Contextualizer`、`QueryRewriter`、`Reranker`）现在与 `EmbeddingClient` 使用相同的 `FUSION_MLX_URL` / `FUSION_MLX_API_KEY`，不再硬编码网关地址 `http://127.0.0.1:11432/v1` 且不带鉴权头。非默认后端（如直连 fusion-mlx `:11434`、使用不同 key）不再在每次 chat-completions / 上下文生成调用上 401。路由层把 `embed.base_url` + `embed.api_key` 传入它构造的每个引擎；`start.sh` 现在启动前激活 monorepo `.venv`（原来用裸 `python3`，未激活 venv 时报 `ModuleNotFoundError: No module named 'fusion_core'`）。新增 4 个回归测试锁定 URL + `Authorization` 头的接线。

---

## 配置

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `FUSION_RAG_PORT` | 11436 | 服务端口 |
| `FUSION_RAG_HOST` | 127.0.0.1 | 监听地址 |
| `FUSION_MLX_URL` | http://127.0.0.1:11432/v1 | fusion-mlx 地址 |
| `FUSION_RAG_EMBED` | BGE-M3 | Embedding 模型 |
| `FUSION_RAG_API_KEY` | (空) | API key 鉴权（空=关闭） |
| `FUSION_RAG_AUTH_BACKEND` | apikey | 鉴权后端：`apikey` 或 `none` |
| `FUSION_RAG_RERANK_BACKEND` | llm | 重排阶段：`llm`（传统 LLM 提示词打分 `Reranker`）或 `cross_encoder`（真实交叉编码器，经 fusion-mlx `POST /v1/rerank`，如 `bge-reranker-v2-m3`）。issue #70 |
| `FUSION_RAG_RERANK_MODEL` | (空) | `cross_encoder` 后端的模型名。设置后 `/search`、`/ask` 默认 `rerank=true`（有模型即默认开启）；留空则除非调用方显式传 `rerank=true` 否则关闭 |
| `FUSION_RAG_RERANK_TOP_N` | 20 | 送入重排的候选池大小，重排后再截断到 `top_k`（在不改变返回数量的前提下提升召回，issue #70） |
| `FUSION_RAG_INGEST_ROOTS` | (空) | 文件入库/扫描/监听的允许根目录（冒号分隔）。空=不限制（本地优先）；设置后限制读取范围，阻断路径遍历/LFI |
| `FUSION_RAG_OCR_URL` | (空) | OCR 服务（fusion-drawing-parser，如 `http://127.0.0.1:11491`）地址，用于扫描版 PDF 入库。留空时扫描件显式报错 "no OCR service configured"（issue #84） |
| `FUSION_RAG_OCR_API_KEY` | (空) | 发送给 OCR 服务的 Bearer 令牌（issue #84） |
| `FUSION_RAG_OCR_TIMEOUT` | 300 | 单个 OCR 任务的轮询总预算（秒，issue #84） |
| `FUSION_RAG_OCR_TENANT_ID` | (空) | OCR 调用 `X-Tenant-Id` 兜底值：请求上下文无租户时使用（后台目录监听入库、单租户部署）。OCR 服务端中间件无条件要求该头（issue #88）；请求内调用会自动透传当前租户 |
| `FUSION_RAG_LOW_OCR_CONFIDENCE` | 0.85 | OCR 片段置信度低于该值时，检索结果与 `/ask` 引用标记 `low_confidence: true`（issue #85） |

### 使用 start.sh

```bash
./start.sh start      # 启动服务
./start.sh stop       # 停止服务
./start.sh restart    # 重启服务
./start.sh status     # 查看状态
```

---

## 开发

```bash
# 安装开发依赖
pip install -e ".[test]"

# 运行测试
pytest tests/

# 带覆盖率运行
pytest tests/ --cov=fusion_rag --cov-report=term-missing
```

### 测试统计
- **40 个测试**, 0 失败
- **核心模块 96%+** 语句覆盖率
- **Python 3.12+** 兼容

---

## 对比

| 维度 | LightRAG | PrivateGPT | **Fusion-RAG** |
|------|----------|-----------|--------------|
| MLX 原生 | ❌ | ❌ | ✅ fusion-mlx API |
| Apple Silicon 优化 | ❌ | ❌ | ✅ LanceDB + MLX |
| 多知识库隔离 | ❌ | ❌ | ✅ |
| 代码分片 | ❌ | ❌ | ✅ |
| 本地离线 | ✅ | ✅ | ✅ 默认离线 |
| 零 API 费用 | ✅ | ✅ | ✅ |

---

## 许可证

MIT

## 致谢

- [fusion-mlx](https://github.com/dahai80/fusion-mlx) — Apple Silicon 模型服务
- [LanceDB](https://github.com/lancedb/lancedb) — 向量数据库
- [LightRAG](https://github.com/HKUDS/LightRAG) — 参考架构