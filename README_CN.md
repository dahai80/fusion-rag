<div align="center">

# Fusion-RAG

**Apple Silicon 原生离线向量知识库底座**

Fusion-MLX 生态的统一本地向量知识库服务——100% 离线，数据不出设备。

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
| **本地离线** | ✅ 100% | ⚠️ 部分 | ❌ |
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
| POST | `/kb/bases/{id}/scan` | 扫描目录批量索引 |

### 搜索与问答

| 方法 | 端点 | 说明 |
|------|------|------|
| POST | `/kb/bases/{id}/search` | 语义向量检索 |
| POST | `/kb/bases/{id}/ask` | RAG 问答（含来源引用） |

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

## 配置

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `FUSION_RAG_PORT` | 11436 | 服务端口 |
| `FUSION_RAG_HOST` | 127.0.0.1 | 监听地址 |
| `FUSION_MLX_URL` | http://127.0.0.1:11432/v1 | fusion-mlx 地址 |
| `FUSION_RAG_EMBED` | BGE-M3 | Embedding 模型 |

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
| 本地离线 | ✅ | ✅ | ✅ 100% |
| 零 API 费用 | ✅ | ✅ | ✅ |

---

## 许可证

MIT

## 致谢

- [fusion-mlx](https://github.com/dahai80/fusion-mlx) — Apple Silicon 模型服务
- [LanceDB](https://github.com/lancedb/lancedb) — 向量数据库
- [LightRAG](https://github.com/HKUDS/LightRAG) — 参考架构