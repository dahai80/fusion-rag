# Fusion-RAG PRD 与实施方案

> 无调用者 — 独立产品规划文档, 不被任何模块 import
> 无 API 影响 — 文档型产出, 不变更现有 API
> 无数据 schema — 纯规划文档, 不定义新 schema
> 用户指令原文: "基于洞察和对开源软件的分析，输出详细的课落地的fusion-rag prd和方案和实施计划prd-ar-plan.md，及关键点，注意以~/claude-home/fusion-mlx,~/fusion/fusion-xx各类产品做好分工，有需求分别提issue和pr，不要修改别的的代码，GUI统一在fusion-studio中，你可以修改fusion-studio的代码"

---

## 目录

1. [产品愿景与定位](#1-产品愿景与定位)
2. [Fusion 生态分工](#2-fusion-生态分工)
3. [当前能力基线](#3-当前能力基线)
4. [PRD: 核心需求](#4-prd-核心需求)
5. [技术方案: P0 关键缺失](#5-技术方案-p0-关键缺失)
6. [技术方案: P1 重要改进](#6-技术方案-p1-重要改进)
7. [技术方案: P2 竞争力提升](#7-技术方案-p2-竞争力提升)
8. [Fusion-Studio GUI 方案](#8-fusion-studio-gui-方案)
9. [跨项目 Issue/PR 分工](#9-跨项目-issuepr-分工)
10. [实施计划与里程碑](#10-实施计划与里程碑)
11. [关键风险与缓解](#11-关键风险与缓解)
12. [验收标准](#12-验收标准)
13. [参考来源](#13-参考来源)

---

## 1. 产品愿景与定位

### 1.1 一句话定位

**Fusion-RAG: Apple Silicon 原生离线 RAG 引擎 — 零云端依赖, Metal GPU 加速, 单进程全功能。**

### 1.2 差异化空间

| 维度 | 开源 RAG 通用 | Fusion-RAG 独有 |
|------|--------------|----------------|
| 硬件 | 通用 x86 + GPU | **Apple Silicon Metal GPU** |
| 网络 | 需 OpenAI/云 API | **完全离线** |
| 部署 | Docker + 微服务栈 | **单进程 + SQLite + LanceDB** |
| 模型推理 | Ollama/云 API | **fusion-mlx 直连 (零网络跳转)** |
| 存储 | ES/Redis/MySQL | **LanceDB + SQLite (嵌入式)** |
| 资源 | 4GB+ | **2GB 可用** |

### 1.3 目标用户

| 用户 | 场景 |
|------|------|
| Apple 生态开发者 | 本地知识库 + 代码检索 |
| 隐私敏感企业 | 完全离线, 数据不出机 |
| 研究/学术 | 低成本 RAG 实验 |
| Fusion Studio 用户 | GUI 一体化 RAG 体验 |

### 1.4 目标对标

不追求 LlamaIndex 的生态广度, 不追求 RAGFlow 的文档深度, 而是做到:

> **"Apple Silicon 上最好的离线 RAG 引擎, 检索质量对标 Anthropic Contextual Retrieval 水平"**

---

## 2. Fusion 生态分工

### 2.1 生态架构

```
┌─────────────────────────────────────────────────────────────┐
│                   Fusion Studio (SwiftUI)                    │
│   统一 GUI: RAG 管理面板 / 聊天 / 检索可视化 / 配置         │
│   仓库: ~/fusion/fusion-studio (可修改)                      │
├─────────────────────────────────────────────────────────────┤
│  fusion-rag (Python)          │  fusion-mlx (Python/MLX)    │
│  RAG 引擎 + API Server        │  推理引擎 + 模型管理         │
│  仓库: ~/fusion/fusion-kb     │  仓库: ~/claude-home/fusion-mlx│
│  (可修改)                      │  (提 Issue, 不改代码)        │
├───────────────────────────────┴─────────────────────────────┤
│  fusion-cli (Rust)  │  fusion-core │  其他 fusion-xx 产品     │
│  CLI + 后台服务      │  共享库       │  (提 Issue, 不改代码)    │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 职责边界

| 仓库 | 职责 | 可否修改代码 | 交互方式 |
|------|------|-------------|----------|
| **fusion-rag** (fusion-kb) | RAG Pipeline + API Server | ✅ 可修改 | HTTP API (port 11436) |
| **fusion-studio** | GUI 前端 | ✅ 可修改 | JSON-RPC / HTTP 调用 fusion-rag |
| **fusion-mlx** | MLX 推理引擎 | ❌ 仅提 Issue | HTTP API (port 11434) |
| **fusion-cli** | CLI 工具 | ❌ 仅提 Issue | 进程调用 |
| **fusion-core** | 共享库 | ❌ 仅提 Issue | Python import |
| 其他 fusion-xx | 垂直产品 | ❌ 仅提 Issue | 各自 API |

### 2.3 跨项目依赖关系

```
fusion-rag ──HTTP──→ fusion-mlx (Embedding + Chat + Rerank)
fusion-studio ──HTTP──→ fusion-rag (/kb/* endpoints)
fusion-studio ──HTTP──→ fusion-mlx (/v1/* endpoints)
fusion-rag ──process──→ fusion-cli (start.sh 调用)
```

---

## 3. 当前能力基线

### 3.1 已实现能力

| 模块 | 文件 | 功能 | 质量评估 |
|------|------|------|----------|
| **文档解析** | `engine/document.py` | PDF/DOCX/MD/TXT/HTML/代码 | ★★★ 基础可用 |
| **分块** | `engine/chunker.py` | 语义/固定/代码 + 递归分块 | ★★★ 基础可用 |
| **预处理** | `engine/preprocessor.py` | 清洗/归一化/去重 | ★★★ 基础可用 |
| **Embedding** | `embed/client.py` | fusion-mlx /v1/embeddings | ★★★ 基础可用 |
| **向量存储** | `store/vector_store.py` | LanceDB cosine search | ★★ keyword_search 有性能问题 |
| **元数据存储** | `store/metadata_store.py` | SQLite 文档/chunk 元数据 | ★★★ 基础可用 |
| **混合检索** | `engine/reranker.py` HybridSearch | Alpha 加权向量+关键词 | ★★ 关键词搜索太弱 |
| **Reranker** | `engine/reranker.py` Reranker | LLM 逐个评分 | ★ 极慢, 不实用 |
| **多路检索** | `engine/retrievers.py` | MMR/ContextCompress/Fusion | ★★ 有框架但效果有限 |
| **多轮对话** | `engine/rag_chain.py` MultiTurnRAG | 内存历史 (max 20) | ★★ 无 token 管理 |
| **文档链** | `engine/rag_chain.py` DocumentChain | stuff/refine/map_reduce | ★★★ 基础可用 |
| **流式输出** | `engine/streaming.py` SSEStreamer | SSE 流式 | ★★★ 基础可用 |
| **缓存** | `engine/streaming.py` ResultCache | SQLite 查询缓存 | ★★★ 基础可用 |
| **API Server** | `api/server.py` + `api/routes.py` | FastAPI 10 个端点 | ★★★ 基础可用 |
| **知识库管理** | `engine/knowledge_base.py` | KB CRUD + 持久化 | ★★★ 基础可用 |

### 3.2 关键问题清单

| # | 问题 | 严重性 | 根因 |
|---|------|--------|------|
| Q1 | keyword_search 加载全表到 Python | P0 | 无 BM25, 用零向量+子串计数 |
| Q2 | Reranker 顺序逐文档 LLM 评分 | P0 | 10 文档 = 10 次 LLM 调用, 延迟 30s+ |
| Q3 | 无 Contextual Retrieval | P0 | chunk 丢失上下文, 检索失败率高 |
| Q4 | MultiTurnRAG 无 token 计数 | P1 | 历史可能超上下文窗口 |
| Q5 | 混合检索仅 Alpha 加权 | P1 | 无 RRF, 对异常分数不鲁棒 |
| Q6 | 无查询改写/扩展 | P1 | 多轮指代消解失败 |
| Q7 | LanceDB delete 返回 DeleteResult | Bug | 3 个测试失败 |
| Q8 | 无 API 认证 | P1 | 任何人可访问 |
| Q9 | 无 RAG 评估 | P1 | 无法衡量检索质量 |

### 3.3 测试状态

- 129 passed, 3 failed
- 3 个失败: LanceDB `delete()` 返回 `DeleteResult` 而非 `int`
- 覆盖率: 84%

---

## 4. PRD: 核心需求

### 4.1 需求总览

| 优先级 | 需求 | 目标指标 | 依赖 |
|--------|------|----------|------|
| **P0-1** | 真正 BM25 检索 | keyword_search 召回率 > 80% | rank_bm25 |
| **P0-2** | Contextual Retrieval | 检索失败率降低 49% | fusion-mlx chat API |
| **P0-3** | Batch Reranker | rerank 延迟 < 3s (10 docs) | fusion-mlx chat API |
| **P0-4** | 修复 LanceDB delete bug | 3 个测试全过 | — |
| **P1-1** | RRF 融合检索 | 混合检索 F1 提升 > 10% | P0-1 |
| **P1-2** | 查询改写 | 多轮对话准确率提升 | fusion-mlx chat |
| **P1-3** | Token 计数 + 历史管理 | 上下文窗口零溢出 | — |
| **P1-4** | Embedding 缓存 | 重复文本零重算 | SQLite |
| **P1-5** | API 认证 | API Key 验证 | — |
| **P2-1** | 轻量级 GraphRAG | 实体关系检索 | fusion-mlx NER |
| **P2-2** | MCP Server | Claude/Cursor 可直接调用 | — |
| **P2-3** | 检索评估框架 | Recall@K 基准测试 | — |
| **P2-4** | Fusion Studio RAG GUI | 可视化管理+聊天+调试 | fusion-studio |

### 4.2 P0 需求详细规格

#### P0-1: BM25 检索

**现状**: `keyword_search()` 用零向量 + 全表 Python 子串计数, 10000 行全加载到内存

**目标**:
- 实现 BM25 算法 (Okapi BM25)
- 替换当前 `keyword_search()` 实现
- 支持中文分词 (jieba)
- 搜索延迟 < 100ms (100K chunks)
- 与向量检索融合 (HybridSearch)

**方案**:
- 引入 `rank_bm25` 库 (轻量, 纯 Python, 无 Torch)
- BM25 索引持久化到 SQLite (倒排索引)
- 中文使用 jieba 分词, 英文使用空格分词
- 索引更新: 增量添加, 文档删除时重建

**不修改**: fusion-mlx, fusion-studio

#### P0-2: Contextual Retrieval

**现状**: chunk 孤立, 丢失文档级上下文, 检索失败率高

**目标**:
- 每个 chunk 摄入时自动生成上下文说明 (50-100 tokens)
- 上下文化后的 chunk 一起做 Embedding
- 上下文化后的 chunk 一起进 BM25 索引
- 检索失败率降低 49% (参考 Anthropic 基准)

**方案**:
- 新增 `engine/contextualizer.py`
- 调用 fusion-mlx `/v1/chat/completions` 生成上下文
- Prompt 模板: Anthropic 推荐模板
- 存储: `metadata_json` 增加 `context` 字段
- 触发: 文档摄入时可选 (默认开启)
- 成本: 利用 fusion-mlx 本地推理, 零 API 费用

**依赖**: 需向 fusion-mlx 提 Issue 确认 Prompt Caching 支持 (降低上下文化成本)

#### P0-3: Batch Reranker

**现状**: Reranker 逐文档调用 LLM 评分, 10 文档 ~30s

**目标**:
- 单次 LLM 调用批量评分所有文档
- Rerank 延迟 < 3s (10 docs)
- 支持分数解析 (0-10 评分)
- 保留 fallback: LLM 不可用时退回向量分数

**方案**:
- 修改 `Reranker._score_relevance()` → `Reranker._batch_score()`
- 单次 Prompt 包含所有文档, LLM 返回 JSON 评分
- 超过 20 文档时分批处理
- 新增 `Reranker.batch_rerank()` 方法

**不修改**: fusion-mlx

#### P0-4: 修复 LanceDB Delete Bug

**现状**: `delete()` 返回 `DeleteResult` 对象而非 `int`, 3 个测试失败

**目标**: 所有测试通过

**方案**: 修改 `VectorStore.delete_by_doc()` 返回值处理

---

## 5. 技术方案: P0 关键缺失

### 5.1 P0-1: BM25 检索

#### 架构设计

```
新增文件: fusion_rag/engine/bm25_index.py

BM25Index
├── __init__(store_path: str)
├── add_documents(chunks: list[dict])     # 增量添加
├── remove_document(doc_path: str)         # 删除文档 chunks
├── search(query: str, top_k: int) -> list # BM25 检索
├── save()                                  # 持久化到 SQLite
├── load()                                  # 从 SQLite 加载
└── count() -> int                          # 索引文档数

修改文件:
  fusion_rag/store/vector_store.py    → keyword_search() 调用 BM25Index
  fusion_rag/engine/reranker.py       → HybridSearch 使用 BM25Index
  fusion_rag/engine/knowledge_base.py → 摄入时同步更新 BM25 索引
```

#### 核心实现逻辑

```python
# bm25_index.py 核心结构
import jieba
import json
import math
import logging
import sqlite3
from pathlib import Path
from collections import Counter

logger = logging.getLogger(__name__)

class BM25Index:
    def __init__(self, store_path: str, k1: float = 1.5, b: float = 0.75):
        self.store_path = Path(store_path)
        self.k1 = k1
        self.b = b
        self._corpus_size = 0
        self._avgdl = 0.0
        self._df = Counter()        # 文档频率
        self._doc_len = {}          # 文档长度
        self._inverted = {}         # 倒排索引: token -> {doc_id: tf}
        self._doc_texts = {}        # doc_id -> 原始文本
        self._db = self._init_db()

    def _tokenize(self, text: str) -> list[str]:
        # 中英混合分词
        import re
        chinese = re.findall(r'[一-鿿]+', text)
        english = re.findall(r'[a-zA-Z0-9]+', text)
        tokens = []
        for seg in chinese:
            tokens.extend(jieba.lcut(seg))
        tokens.extend(w.lower() for w in english)
        return [t for t in tokens if len(t) > 1]

    def add_documents(self, chunks: list[dict]) -> None:
        for chunk in chunks:
            doc_id = chunk["id"]
            text = chunk.get("text", "")
            self._doc_texts[doc_id] = text
            tokens = self._tokenize(text)
            self._doc_len[doc_id] = len(tokens)
            tf = Counter(tokens)
            for token, freq in tf.items():
                if token not in self._inverted:
                    self._inverted[token] = {}
                self._inverted[token][doc_id] = freq
            self._corpus_size += 1
        self._df = Counter()
        for token, postings in self._inverted.items():
            self._df[token] = len(postings)
        total_len = sum(self._doc_len.values())
        self._avgdl = total_len / max(self._corpus_size, 1)
        self.save()

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        query_tokens = self._tokenize(query)
        scores = {}
        for token in query_tokens:
            if token not in self._inverted:
                continue
            df = self._df.get(token, 0)
            idf = math.log((self._corpus_size - df + 0.5) / (df + 0.5) + 1)
            for doc_id, tf in self._inverted[token].items():
                dl = self._doc_len.get(doc_id, 1)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / max(self._avgdl, 1))
                scores[doc_id] = scores.get(doc_id, 0) + idf * numerator / denominator
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [{"id": did, "score": s, "text": self._doc_texts.get(did, "")} for did, s in ranked]

    def _init_db(self) -> sqlite3.Connection:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(str(self.store_path))
        db.execute("""CREATE TABLE IF NOT EXISTS bm25_meta (
            key TEXT PRIMARY KEY, value TEXT)""")
        return db

    def save(self) -> None:
        data = {
            "corpus_size": self._corpus_size,
            "avgdl": self._avgdl,
            "doc_len": self._doc_len,
            "df": dict(self._df),
            "doc_texts_count": len(self._doc_texts),
        }
        for k, v in data.items():
            self._db.execute("INSERT OR REPLACE INTO bm25_meta VALUES (?, ?)",
                           (k, json.dumps(v, ensure_ascii=False)))
        self._db.commit()

    def remove_document(self, doc_path: str) -> None:
        to_remove = [did for did, txt in self._doc_texts.items() if doc_path in txt]
        for did in to_remove:
            del self._doc_texts[did]
            self._doc_len.pop(did, None)
            self._corpus_size -= 1
        for token in list(self._inverted.keys()):
            self._inverted[token] = {k: v for k, v in self._inverted[token].items()
                                      if k not in to_remove}
            if not self._inverted[token]:
                del self._inverted[token]
        self._df = Counter()
        for token, postings in self._inverted.items():
            self._df[token] = len(postings)
        self.save()
```

#### 依赖变更

```toml
# pyproject.toml 新增
dependencies = [
    # ... 现有依赖 ...
    "rank_bm25>=0.2.2",   # BM25 参考实现 (用于对比验证)
    "jieba>=0.42.1",      # 中文分词
]
```

#### VectorStore 改造

```python
# vector_store.py 改造要点
class VectorStore:
    def __init__(self, vector_path: str, dimension: int = 1024):
        # ... 现有代码 ...
        self._bm25 = None  # 延迟初始化

    @property
    def bm25(self):
        if self._bm25 is None:
            from .bm25_store import BM25Store
            bm25_path = str(Path(self.vector_path).parent / "bm25_index.json")
            self._bm25 = BM25Store(bm25_path, self.dimension)
        return self._bm25

    def keyword_search(self, query: str, top_k: int = 10) -> list[dict]:
        # 替换原有实现: 直接调用 BM25
        return self.bm25.search(query, top_k)
```

### 5.2 P0-2: Contextual Retrieval

#### 架构设计

```
新增文件: fusion_rag/engine/contextualizer.py

Contextualizer
├── __init__(mlx_url: str, model: str)
├── contextualize(chunks: list[dict], doc_text: str) -> list[dict]
│   # 为每个 chunk 生成上下文说明
│   # 返回: chunks 增加 "context" 字段
├── _generate_context(client, chunk_text, doc_text) -> str
│   # 调用 fusion-mlx 生成上下文
└── _build_prompt(chunk_text, doc_text) -> str
    # Anthropic 推荐模板

修改文件:
  fusion_rag/engine/knowledge_base.py -> 摄入时调用 Contextualizer
  fusion_rag/store/vector_store.py    -> add() 存储 context 字段
  fusion_rag/api/routes.py            -> 添加 contextualize 参数
```

#### 核心实现

```python
# contextualizer.py
import logging
import httpx

logger = logging.getLogger(__name__)

CONTEXT_PROMPT = """<document>
{doc_text}
</document>
Here is the chunk we want to situate within the whole document
<chunk>
{chunk_text}
</chunk>
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""


class Contextualizer:
    def __init__(self, mlx_url: str = "http://localhost:11434/v1",
                 model: str = "qwen3.5-9b", enabled: bool = True):
        self.mlx_url = mlx_url.rstrip("/")
        self.model = model
        self.enabled = enabled

    async def contextualize(self, chunks: list[dict],
                            doc_text: str) -> list[dict]:
        if not self.enabled or not doc_text:
            return chunks
        # 截断文档以适应上下文窗口
        doc_truncated = doc_text[:8000]
        async with httpx.AsyncClient(timeout=30.0) as client:
            for chunk in chunks:
                try:
                    context = await self._generate_context(
                        client, chunk.get("text", ""), doc_truncated)
                    chunk["context"] = context
                except Exception as e:
                    logger.warning("Context generation failed for chunk %s: %s",
                                 chunk.get("id", "?"), e)
                    chunk["context"] = ""
        return chunks

    async def _generate_context(self, client: httpx.AsyncClient,
                                chunk_text: str, doc_text: str) -> str:
        prompt = CONTEXT_PROMPT.format(
            doc_text=doc_text, chunk_text=chunk_text)
        resp = await client.post(
            f"{self.mlx_url}/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 100,
                "temperature": 0.0,
            })
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        return content[:200]  # 限制上下文长度
```

#### 摄入流程改造

```python
# knowledge_base.py 修改要点
class KnowledgeBase:
    def __init__(self, ...):
        # ... 现有代码 ...
        self._contextualizer = Contextualizer(mlx_url, model)

    async def add_document(self, file_path: str, ...):
        # 1. 解析
        text = self._parser.parse(file_path)
        # 2. 分块
        chunks = self._chunker.chunk(text, ...)
        # 3. 上下文化 (新增)
        chunks = await self._contextualizer.contextualize(chunks, text)
        # 4. Embedding (上下文化后的文本)
        for chunk in chunks:
            embed_text = chunk.get("context", "") + " " + chunk.get("text", "")
            chunk["vector"] = await self._embedder.embed(embed_text)
        # 5. 存储
        self._vector_store.add_batch(chunks)
        # 6. BM25 索引更新
        self._vector_store.bm25.add_documents(chunks)
```

### 5.3 P0-3: Batch Reranker

#### 架构设计

```
修改文件: fusion_rag/engine/reranker.py

Reranker (改造)
├── __init__(mlx_url, model, batch_size=20)
├── rerank(query, documents, top_k)       # 对外接口不变
├── _batch_score(client, query, docs)     # 新增: 批量评分
├── _score_relevance(...)                 # 保留: 兜底单文档评分
└── _parse_scores(content, n)             # 新增: 解析 LLM 输出

HybridSearch (改造)
├── search(query_vector, query_text, top_k, threshold, filters, method)
│   # method: "alpha" (现有) | "rrf" (新增)
├── _alpha_fusion(vector_results, keyword_results)  # 现有
└── _rrf_fusion(vector_results, keyword_results)    # 新增
```

#### 核心实现

```python
# reranker.py 改造要点
class Reranker:
    def __init__(self, mlx_url: str = "http://localhost:11434/v1",
                 model: str = "qwen3.5-9b", batch_size: int = 20):
        self.mlx_url = mlx_url.rstrip("/")
        self.model = model
        self.batch_size = batch_size

    async def rerank(self, query: str, documents: list[dict],
                     top_k: int = 5) -> list[dict]:
        if not documents:
            return []
        # 分批处理
        all_scored = []
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i:i + self.batch_size]
            scores = await self._batch_score(query, batch)
            for doc, score in zip(batch, scores):
                doc["score"] = score
                all_scored.append(doc)
        all_scored.sort(key=lambda x: x["score"], reverse=True)
        return all_scored[:top_k]

    async def _batch_score(self, query: str,
                           docs: list[dict]) -> list[float]:
        doc_list = "\n".join(
            f"[{i}] {doc.get('text', '')[:500]}"
            for i, doc in enumerate(docs))
        prompt = (
            f"Rate the relevance of each document to the query "
            f"on a scale of 0.0 to 10.0.\n\n"
            f"Query: {query}\n\n"
            f"Documents:\n{doc_list}\n\n"
            f"Output ONLY a JSON array of {len(docs)} scores, e.g. [8.5, 3.2, ...]:"
        )
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{self.mlx_url}/chat/completions",
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 200,
                        "temperature": 0.0,
                    })
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"].strip()
                return self._parse_scores(content, len(docs))
        except Exception as e:
            logger.warning("Batch rerank failed, fallback: %s", e)
            return [doc.get("score", 5.0) for doc in docs]

    def _parse_scores(self, content: str, expected: int) -> list[float]:
        import re
        # 尝试解析 JSON 数组
        try:
            import json
            scores = json.loads(content)
            if isinstance(scores, list):
                return [float(s) for s in scores[:expected]]
        except Exception:
            pass
        # Fallback: 提取数字
        nums = re.findall(r'\d+\.?\d*', content)
        if len(nums) >= expected:
            return [float(n) for n in nums[:expected]]
        # 最终兜底
        return [5.0] * expected
```

### 5.4 P0-4: LanceDB Delete Bug 修复

```python
# vector_store.py 修改
def delete_by_doc(self, doc_path: str) -> int:
    safe = doc_path.replace("'", "''")
    try:
        result = self.table.delete(f"doc_path = '{safe}'")
        # LanceDB 返回 DeleteResult 或 int
        if hasattr(result, '__int__'):
            return int(result)
        if isinstance(result, int):
            return result
        return 0  # DeleteResult 无法直接转 int 时
    except Exception:
        return 0
```

---

## 6. 技术方案: P1 重要改进

### 6.1 P1-1: RRF 融合检索

**背景**: 当前 HybridSearch 仅使用 Alpha 加权融合, 对异常分数敏感

**方案**: 增加 RRF (Reciprocal Rank Fusion) 方法

```python
# HybridSearch 改造
class HybridSearch:
    def __init__(self, vector_store, alpha: float = 0.7,
                 method: str = "rrf"):
        self.vector_store = vector_store
        self.alpha = alpha
        self.method = method  # "alpha" | "rrf"

    async def search(self, query_vector, query_text, top_k=10,
                     threshold=0.0, filters=None) -> list[dict]:
        vector_results = self.vector_store.search(query_vector, top_k=top_k * 2)
        keyword_results = self.vector_store.keyword_search(query_text, top_k=top_k * 2)
        if filters:
            vector_results = self._apply_filters(vector_results, filters)
            keyword_results = self._apply_filters(keyword_results, filters)
        if self.method == "rrf":
            return self._rrf_fusion(vector_results, keyword_results,
                                    top_k, threshold)
        return self._alpha_fusion(vector_results, keyword_results,
                                  top_k, threshold)

    @staticmethod
    def _rrf_fusion(vector_results, keyword_results, top_k, threshold,
                    k: int = 60) -> list[dict]:
        # RRF: score = Sigma 1/(k + rank_i)
        scores = {}
        for rank, r in enumerate(vector_results, 1):
            rid = r.get("id", "")
            scores[rid] = scores.get(rid, 0) + 1.0 / (k + rank)
        for rank, r in enumerate(keyword_results, 1):
            rid = r.get("id", "")
            scores[rid] = scores.get(rid, 0) + 1.0 / (k + rank)
        result_map = {r["id"]: r for r in vector_results}
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for rid, s in ranked:
            if s < threshold:
                continue
            entry = result_map.get(rid, {"id": rid})
            entry["score"] = s
            results.append(entry)
        return results[:top_k]
```

### 6.2 P1-2: 查询改写

**背景**: 多轮对话中 "它" / "这个问题" 等指代无法解析

**方案**: 新增 `engine/query_rewriter.py`

```python
# query_rewriter.py
class QueryRewriter:
    def __init__(self, mlx_url, model):
        self.mlx_url = mlx_url.rstrip("/")
        self.model = model

    async def rewrite(self, query: str, history: list[dict]) -> str:
        if not history:
            return query
        # 用 LLM 做指代消解和查询扩展
        history_text = "\n".join(
            f"{'用户' if h['role']=='user' else '助手'}: {h['content'][:200]}"
            for h in history[-6:])
        prompt = (
            f"基于对话历史, 改写用户的最新问题, 消除代词指代, "
            f"使其成为独立可检索的查询。\n\n"
            f"对话历史:\n{history_text}\n\n"
            f"最新问题: {query}\n\n"
            f"改写后的查询 (直接输出, 无解释):"
        )
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{self.mlx_url}/chat/completions",
                json={"model": self.model,
                      "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": 200, "temperature": 0.0})
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
```

### 6.3 P1-3: Token 计数 + 历史管理

```python
# MultiTurnRAG 改造
class MultiTurnRAG:
    MAX_CONTEXT_TOKENS = 6000  # 留 2000 给生成
    MAX_HISTORY_TURNS = 10

    def _estimate_tokens(self, text: str) -> int:
        # 粗略估算: 中文 ~1.5 token/字, 英文 ~1.3 token/word
        import re
        chinese = len(re.findall(r'[一-鿿]', text))
        english = len(re.findall(r'[a-zA-Z]+', text))
        return int(chinese * 1.5 + english * 1.3)

    def _trim_history(self, context_text: str) -> list[dict]:
        context_tokens = self._estimate_tokens(context_text)
        budget = self.MAX_CONTEXT_TOKENS - context_tokens
        trimmed = []
        used = 0
        for h in reversed(self._history):
            h_tokens = self._estimate_tokens(h.get("content", ""))
            if used + h_tokens > budget:
                break
            trimmed.insert(0, h)
            used += h_tokens
        if len(trimmed) > self.MAX_HISTORY_TURNS:
            trimmed = trimmed[-self.MAX_HISTORY_TURNS:]
        return trimmed
```

### 6.4 P1-4: Embedding 缓存

```python
# 新增: engine/embedding_cache.py
class EmbeddingCache:
    def __init__(self, db_path: str):
        self._db = sqlite3.connect(db_path)
        self._db.execute("""CREATE TABLE IF NOT EXISTS embed_cache (
            text_hash TEXT PRIMARY KEY,
            model TEXT,
            embedding BLOB,
            created_at REAL)""")

    def get(self, text: str, model: str) -> list[float] | None:
        h = hashlib.md5(text.encode()).hexdigest()
        row = self._db.execute(
            "SELECT embedding FROM embed_cache WHERE text_hash=? AND model=?",
            (h, model)).fetchone()
        if row:
            return json.loads(row[0])
        return None

    def set(self, text: str, model: str, embedding: list[float]) -> None:
        h = hashlib.md5(text.encode()).hexdigest()
        self._db.execute(
            "INSERT OR REPLACE INTO embed_cache VALUES (?, ?, ?, ?)",
            (h, model, json.dumps(embedding), time.time()))
        self._db.commit()
```

### 6.5 P1-5: API 认证

```python
# api/auth.py 新增
from fastapi import Header, HTTPException

_API_KEYS: set[str] = set()

def load_api_keys(keys_path: str = "~/.fusion-rag/api_keys.txt"):
    global _API_KEYS
    path = Path(keys_path).expanduser()
    if path.exists():
        _API_KEYS = {line.strip() for line in path.read_text().splitlines()
                     if line.strip() and not line.startswith("#")}

async def verify_api_key(x_api_key: str = Header(None)) -> None:
    if not _API_KEYS:
        return  # 无 key 配置时跳过认证
    if x_api_key not in _API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
```

---

## 7. 技术方案: P2 竞争力提升

### 7.1 P2-1: 轻量级 GraphRAG

**参考**: LightRAG (NetworkX 后端), RAGFlow (轻量图)

**方案**: 新增 `engine/graph_rag.py`

```
GraphRAG
├── EntityExtractor      # LLM 实体提取
├── RelationExtractor    # LLM 关系提取
├── GraphStore           # NetworkX 存储 (可选 Neo4j)
├── GraphSearch          # 实体/关系检索
└── GraphRAGPipeline     # 图+向量融合检索
```

**关键决策**:
- 存储后端: NetworkX (嵌入式, 零依赖) — 不引入 Neo4j
- 实体提取: 调用 fusion-mlx chat API
- 触发方式: 文档摄入时可选 (默认关闭, 因为成本高)
- 检索模式: local (实体邻居) + global (关系路径)

**需向 fusion-mlx 提 Issue**: 是否支持 NER 模型 (如 GLiNER)

### 7.2 P2-2: MCP Server

**方案**: 新增 `api/mcp_server.py`

```python
# MCP Server 实现
# 标准 MCP 协议, 让 Claude/Cursor 等 Agent 可直接调用 RAG

MCP Tools:
  - rag_search(query, kb_id, top_k) -> 检索结果
  - rag_ask(query, kb_id) -> RAG 问答
  - rag_list_bases() -> 知识库列表
  - rag_add_document(path, kb_id) -> 添加文档
  - rag_get_document(doc_id) -> 获取文档
```

### 7.3 P2-3: 检索评估框架

**方案**: 新增 `engine/evaluator.py`

```python
class RAGEvaluator:
    def evaluate_recall(self, queries: list[dict], top_k: int = 10) -> float:
        # 计算 Recall@K

    def evaluate_mrr(self, queries: list[dict]) -> float:
        # 计算 MRR

    def evaluate_groundedness(self, queries: list[dict]) -> float:
        # LLM 评分: 回答是否基于上下文

    def benchmark(self, kb_id: str, sample_size: int = 50) -> dict:
        # 运行完整基准测试
```

---

## 8. Fusion Studio GUI 方案

### 8.1 当前状态

fusion-studio 已有 `RAGPipelineView.swift` (729 行), 包含:
- `DocumentChunk`, `RetrievalResult` 数据模型
- `RetrievalStrategy` 枚举 (dense/sparse/hybrid/rerank)
- `ChunkStrategy` 枚举
- `RAGEngine` 类 (本地 Swift 实现)
- 基础 UI 框架

**问题**: 当前 RAGEngine 是纯 Swift 本地实现, 未连接 fusion-rag 后端 API。

### 8.2 GUI 改造方案

#### 目标界面

```
+-------------------------------------------------------------+
|  Fusion Studio - 知识库                                      |
+------+------------------------------------------------------+
|      |  [知识库列表]  [+] 新建                                |
| 侧   |  +----------------------------------------------+    |
| 边   |  | 知识库: 技术文档库                               |    |
| 栏   |  | 文档数: 128 | Chunks: 3456 | 大小: 2.3GB        |    |
|      |  +----------------------------------------------+    |
|      |  | [文档] [搜索] [聊天] [设置]                      |    |
|      |  +----------------------------------------------+    |
|      |  | 文档管理:                                       |    |
|      |  |  api_design.md  328 chunks                     |    |
|      |  |  architecture.md  156 chunks                   |    |
|      |  |  + 添加文档/扫描目录                             |    |
|      |  +----------------------------------------------+    |
|      |  | 检索调试:                                       |    |
|      |  |  [________________] [搜索]                      |    |
|      |  |  策略: [hybrid] RRF  top_k: [10]               |    |
|      |  |  -- 结果 --                                     |    |
|      |  |  1. [0.92] api_design.md:42 - REST API...      |    |
|      |  |  2. [0.87] architecture.md:15 - 微服务...       |    |
|      |  +----------------------------------------------+    |
|      |  | RAG 聊天:                                       |    |
|      |  |  根据 API 设计文档, REST 接口...                  |    |
|      |  |  来源: api_design.md:42, arch.md:15             |    |
|      |  |  [___________________________] [发送]           |    |
|      |  +----------------------------------------------+    |
+------+------------------------------------------------------+
```

#### 改造要点

| 改造项 | 文件 | 说明 |
|--------|------|------|
| RAGEngine -> APIClient | `RAGPipelineView.swift` | 删除本地 Swift RAG 逻辑, 改为调用 fusion-rag HTTP API |
| 新增 KBListView | `Modules/KnowledgeBase/KBListView.swift` | 知识库列表 + 创建 + 删除 |
| 新增 KBChatView | `Modules/KnowledgeBase/KBChatView.swift` | RAG 聊天 + 来源引用 |
| 新增 KBSettingsView | `Modules/KnowledgeBase/KBSettingsView.swift` | 检索策略/Rerank/上下文化配置 |
| 新增 SearchDebugView | `Modules/KnowledgeBase/SearchDebugView.swift` | 检索结果可视化 + 分数调试 |
| IPC 集成 | `Bridge/IPCClient.swift` | 新增 `rag.*` IPC 命名空间 |

#### IPC API 设计

```
rag.list_bases          -> GET /kb/bases
rag.create_base         -> POST /kb/bases
rag.delete_base         -> DELETE /kb/bases/{id}
rag.add_document        -> POST /kb/bases/{id}/documents
rag.scan_directory      -> POST /kb/bases/{id}/scan
rag.search              -> POST /kb/bases/{id}/search
rag.ask                 -> POST /kb/bases/{id}/ask
rag.stats               -> GET /kb/bases/{id}/stats
rag.health              -> GET /kb/status
```

### 8.3 Studio 实施策略

1. **Phase 1**: 重构 RAGPipelineView, 对接 fusion-rag API
2. **Phase 2**: 新增 KBListView + KBChatView
3. **Phase 3**: SearchDebugView + 配置面板

**注意**: 只修改 `~/fusion/fusion-studio/`, 不修改其他仓库代码。

---

## 9. 跨项目 Issue/PR 分工

### 9.1 fusion-rag (可修改, 直接 PR)

| PR | 内容 | 优先级 |
|----|------|--------|
| PR-1 | BM25 检索 (bm25_index.py + VectorStore 改造) | P0 |
| PR-2 | Contextual Retrieval (contextualizer.py + 摄入流程改造) | P0 |
| PR-3 | Batch Reranker (reranker.py 改造) | P0 |
| PR-4 | LanceDB delete bug 修复 | P0 |
| PR-5 | RRF 融合 (HybridSearch 改造) | P1 |
| PR-6 | 查询改写 (query_rewriter.py) | P1 |
| PR-7 | Token 计数 + 历史管理 | P1 |
| PR-8 | Embedding 缓存 (embedding_cache.py) | P1 |
| PR-9 | API 认证 (auth.py + routes 改造) | P1 |
| PR-10 | GraphRAG (graph_rag.py) | P2 |
| PR-11 | MCP Server (mcp_server.py) | P2 |
| PR-12 | 评估框架 (evaluator.py) | P2 |

### 9.2 fusion-studio (可修改, 直接 PR)

| PR | 内容 | 优先级 |
|----|------|--------|
| PR-S1 | 重构 RAGPipelineView, 对接 fusion-rag API | P1 |
| PR-S2 | 新增 KBListView + KBChatView | P1 |
| PR-S3 | SearchDebugView + 配置面板 | P2 |
| PR-S4 | IPC rag.* 命名空间 | P1 |

### 9.3 fusion-mlx (不可修改, 提 Issue)

| Issue | 内容 | 优先级 | 原因 |
|-------|------|--------|------|
| Issue-MLX-1 | Prompt Caching 支持 | P0 | Contextual Retrieval 成本优化 |
| Issue-MLX-2 | Rerank 专用端点 /v1/rerank | P1 | 专用 Reranker 模型支持 |
| Issue-MLX-3 | NER 模型支持 (GLiNER) | P2 | GraphRAG 实体提取 |
| Issue-MLX-4 | BGE-Reranker 模型加载 | P1 | Cross-Encoder Reranking |
| Issue-MLX-5 | Embedding batch API 优化 | P1 | 大批量文档摄入性能 |

### 9.4 fusion-cli (不可修改, 提 Issue)

| Issue | 内容 | 优先级 |
|-------|------|--------|
| Issue-CLI-1 | rag 子命令 (start/stop/status/search) | P2 |

### 9.5 其他 fusion-xx (不可修改, 提 Issue)

| 项目 | Issue | 优先级 |
|------|-------|--------|
| fusion-agent-studio | RAG 工具节点集成 | P2 |
| fusion-code | 代码知识库 RAG 集成 | P2 |

---

## 10. 实施计划与里程碑

### 10.1 Phase 1: 检索质量 (2 周, 8/4 - 8/17)

| 周 | 任务 | PR | 交付物 |
|----|------|-----|--------|
| W1 | BM25 检索 | PR-1 | bm25_index.py + VectorStore 改造 + 测试 |
| W1 | LanceDB bug 修复 | PR-4 | 3 个测试修复 |
| W2 | Batch Reranker | PR-3 | reranker.py 改造 + 测试 |
| W2 | Contextual Retrieval | PR-2 | contextualizer.py + 摄入流程改造 + 测试 |

**里程碑 M1**: 检索质量达标 — BM25 + Contextual + Batch Rerank 全部可用

**验收**:
- BM25 搜索延迟 < 100ms (1000 chunks)
- Rerank 10 文档延迟 < 3s
- Contextual Retrieval 在测试集上 Recall@10 提升 > 30%
- 所有测试通过 (132/132)

### 10.2 Phase 2: 检索策略 + 用户体验 (3 周, 8/18 - 9/7)

| 周 | 任务 | PR | 交付物 |
|----|------|-----|--------|
| W3 | RRF 融合 | PR-5 | HybridSearch RRF 方法 + 测试 |
| W3 | 查询改写 | PR-6 | query_rewriter.py + 测试 |
| W4 | Token 计数 | PR-7 | MultiTurnRAG 历史管理 + 测试 |
| W4 | Embedding 缓存 | PR-8 | embedding_cache.py + 测试 |
| W4 | API 认证 | PR-9 | auth.py + routes 改造 + 测试 |
| W5 | Studio 对接 | PR-S1, PR-S4 | RAGPipelineView 重构 + IPC |

**里程碑 M2**: 用户体验达标 — 查询改写 + 历史管理 + Studio 基础集成

**验收**:
- 混合检索 F1 提升 > 10% (vs Phase 1)
- 多轮对话无上下文溢出
- Studio 可展示知识库列表 + 搜索结果

### 10.3 Phase 3: 竞争力 + GUI 完善 (4 周, 9/8 - 10/5)

| 周 | 任务 | PR | 交付物 |
|----|------|-----|--------|
| W6 | GraphRAG 基础 | PR-10 | entity/relation 提取 + NetworkX 存储 |
| W6 | Studio 聊天 | PR-S2 | KBListView + KBChatView |
| W7 | GraphRAG 检索 | PR-10 | graph + vector 融合检索 |
| W7 | MCP Server | PR-11 | mcp_server.py |
| W8 | 评估框架 | PR-12 | evaluator.py + 基准数据集 |
| W8 | Studio 完善 | PR-S3 | SearchDebugView + 配置面板 |

**里程碑 M3**: 竞争力达标 — GraphRAG + MCP + 评估 + Studio 完整

**验收**:
- GraphRAG 检索可用 (实体/关系查询)
- MCP Server 可被 Claude/Cursor 调用
- 评估框架可输出 Recall@K / MRR 报告
- Studio 完整 RAG 管理 + 聊天 + 调试体验

### 10.4 里程碑总览

```
2026-08
  W1-W2 --- M1: 检索质量 --- BM25 + Contextual + Batch Rerank
      |
2026-09
  W3-W5 --- M2: 用户体验 --- RRF + QueryRewrite + Token管理 + Studio基础
      |
2026-09 ~ 10
  W6-W8 --- M3: 竞争力 --- GraphRAG + MCP + Eval + Studio完善
```

---

## 11. 关键风险与缓解

| 风险 | 概率 | 影响 | 缓解策略 |
|------|------|------|----------|
| fusion-mlx 无 Prompt Caching | 中 | Contextual 成本高 | 降级: 跳过上下文或用小模型; 同时提 Issue 推动 |
| BM25 中文分词质量差 | 低 | 关键词检索不准 | 使用 jieba + 自定义词典; 长期可考虑 pkuseg |
| Batch Reranker LLM 输出不稳定 | 中 | 评分解析失败 | 多层 fallback: JSON -> 正则 -> 默认分数 |
| GraphRAG LLM 成本过高 | 高 | 用户不启用 | 默认关闭, 文档量大时提示; 小文档可直接启用 |
| fusion-mlx 不支持 Rerank 模型 | 中 | 无专用 Cross-Encoder | 继续用 LLM Rerank; 同时提 Issue 请求 BGE-Reranker |
| Studio SwiftUI 编译问题 | 低 | GUI 延迟 | RAG 功能不依赖 GUI, API 先行 |
| LanceDB 版本不兼容 | 低 | 向量存储损坏 | 锁版本 + 迁移脚本 |

---

## 12. 验收标准

### 12.1 功能验收

| # | 验收项 | 通过标准 |
|---|--------|----------|
| F1 | BM25 搜索 | 中文/英文查询均可用, 延迟 < 100ms @ 10K chunks |
| F2 | Contextual Retrieval | Recall@10 比无上下文提升 > 30% |
| F3 | Batch Reranker | 10 文档 rerank < 3s, 评分解析成功率 > 95% |
| F4 | RRF 融合 | F1 比 Alpha 加权提升 > 10% |
| F5 | 查询改写 | 多轮指代查询改写正确率 > 80% |
| F6 | Token 管理 | 0 次上下文溢出 (100 轮对话测试) |
| F7 | Embedding 缓存 | 重复文本缓存命中率 > 90% |
| F8 | API 认证 | 无 key 拒绝访问, 有 key 正常访问 |
| F9 | 所有测试 | 132+ 测试全绿, 覆盖率 > 85% |

### 12.2 性能验收

| # | 指标 | 基线 | 目标 |
|---|------|------|------|
| P1 | 单文档摄入 (10页 PDF) | ~30s | < 20s (缓存 Embedding) |
| P2 | 搜索延迟 (10K chunks) | ~500ms | < 200ms (BM25 + 优化) |
| P3 | Rerank 延迟 (10 docs) | ~30s | < 3s (Batch) |
| P4 | 多轮对话 (5轮) | ~8s/轮 | < 5s/轮 (缓存 + Token 管理) |
| P5 | 上下文化 (100 chunks) | N/A | < 120s (本地 fusion-mlx) |

### 12.3 质量验收

| # | 指标 | 目标 |
|---|------|------|
| Q1 | 测试覆盖率 | > 85% |
| Q2 | 测试全绿 | 0 失败 |
| Q3 | 类型提示 | 所有公开 API 有类型提示 |
| Q4 | 日志 | 所有核心路径有 logger |
| Q5 | 文档 | README.md + API 文档同步更新 |

---

## 13. 参考来源

### 内部分析

1. `claude-rag-insight.md` — Claude RAG 深度洞察报告 (1746 行, 23 章)
2. `rag-oss-analysis.md` — 开源 RAG 系统分析报告 (17 章, 6 项目)

### 开源项目

3. LightRAG — 图增强 RAG, 6 种搜索模式
4. anything-llm — 全栈 RAG 应用, Workspace 隔离
5. DSPy — 声明式 RAG, 自动优化
6. Haystack — Pipeline 架构, AutoMerging/SentenceWindow
7. LlamaIndex — 最大生态, 78 向量DB + 104 LLM
8. RAGFlow — 企业级 DeepDoc, 15 种解析器

### 关键技术参考

9. Anthropic Contextual Retrieval — 检索失败率降低 67%
10. LanceDB RRF 融合 — 内置 RRF/Linear/MRR 四种方法
11. BGE-M3 — 统一 dense + sparse + multi-vec 单模型
12. FlashRank — 超轻量 CPU Reranker (4MB 模型)
13. MXBai Reranker v2 — 多语言 SOTA Reranker

### Fusion 生态

14. fusion-mlx — Apple Silicon 推理引擎 (port 11434)
15. fusion-studio — 统一 macOS 桌面客户端 (SwiftUI)
16. RAGPipelineView.swift — 现有 Studio RAG 模块 (729 行)

---

> 本 PRD 基于深度洞察和开源分析, 所有方案均为 fusion-rag + fusion-studio 内可落地方案,
> 跨项目依赖通过 Issue 沟通, 遵循 "不修改别的代码" 原则。
