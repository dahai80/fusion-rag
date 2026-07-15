# Fusion-KB 业界对标与差距分析

> 对比 Dify RAG、LlamaIndex、LangChain RAG、Haystack、Chroma 等业界最佳方案，
> 定位 Fusion-KB 当前能力缺口。

---

## 一、对标产品功能矩阵

### 1.1 Dify RAG（最完整 RAG 平台）

| 能力维度 | Dify RAG | Fusion-KB | 差距 | 优先级 |
|---------|----------|-----------|------|--------|
| **文档解析** | PDF/DOCX/MD/TXT/HTML/图片/音频 | PDF/DOCX/MD/TXT/HTML/代码 | 🟡 缺图片OCR、音频 | P2 |
| **分片策略** | 固定/语义/递归/代码 | 固定/语义/代码 | 🟡 缺递归分片 | P1 |
| **Embedding** | 30+ 模型（云端+本地） | 1个（BGE-M3 via fusion-mlx） | 🟡 多模型选择 | P1 |
| **检索方式** | 向量+关键词+混合+重排 | 向量+关键词 | 🔴 缺重排 | P0 |
| **Rerank** | ✅ 内置 Rerank 模型 | ❌ | 🔴 缺重排 | P0 |
| **知识库数量** | 无限 | 无限 | ✅ 持平 | — |
| **知识库隔离** | ✅ 完整 | ✅ 完整 | ✅ 持平 | — |
| **文档更新** | 增量+全量+自动 | 增量（手动触发） | 🟡 缺自动同步 | P1 |
| **检索策略** | 向量/关键词/混合/N选1 | 向量+关键词 | 🔴 缺混合检索策略 | P0 |
| **RAG 问答** | 多轮对话+引用+溯源 | 单轮+引用 | 🟡 缺多轮对话 | P1 |
| **性能监控** | ✅ 完整看板 | ❌ | 🟡 缺监控 | P2 |
| **Web UI** | ✅ 完整管理后台 | ❌（纯API） | 🟡 非必须（API定位） | — |
| **API 文档** | ✅ OpenAPI 完整 | ✅ 手动文档 | ✅ 持平 | — |
| **多用户** | ✅ 团队协作 | ❌ | 🟡 非MVP | P2 |
| **文件管理** | ✅ 可视化文件管理 | ❌（通过API） | 🟡 非必须 | — |

### 1.2 LlamaIndex（最流行数据框架）

| 能力维度 | LlamaIndex | Fusion-KB | 差距 | 优先级 |
|---------|-----------|-----------|------|--------|
| **数据连接器** | 160+（数据库/API/SaaS/文件） | 7种（本地文件） | 🔴 缺数据库连接器 | P0 |
| **索引类型** | 向量/摘要/树/关键词/知识图谱 | 向量 | 🔴 缺多种索引 | P1 |
| **检索策略** | 融合/路由/递归/重排 | 向量+关键词 | 🔴 缺融合检索 | P0 |
| **后处理** | 去重/重排/过滤/上下文压缩 | 无 | 🔴 缺后处理 | P1 |
| **节点解析** | 5种分片+自定义 | 3种分片 | 🟡 缺递归分片 | P1 |
| **元数据提取** | 自动提取+自定义 | 基础元数据 | 🔴 缺自动提取 | P1 |
| **回调/日志** | 完整回调系统 | 基础日志 | 🟡 缺结构化日志 | P2 |

### 1.3 LangChain RAG（最流行 RAG 框架）

| 能力维度 | LangChain RAG | Fusion-KB | 差距 | 优先级 |
|---------|--------------|-----------|------|--------|
| **文档加载器** | 100+ | 7种 | 🔴 缺网页/数据库/API | P0 |
| **文本分割器** | 6种（递归/字符/代码/语义等） | 3种 | 🟡 缺递归分割 | P1 |
| **向量存储** | 15+（Chroma/Pinecone/Weaviate等） | 1种（LanceDB） | 🟡 单一存储 | P2 |
| **检索器** | 10+（向量/上下文压缩/MMR/重排） | 2种 | 🔴 缺多种检索器 | P0 |
| **文档链** | 多种链（stuff/refine/map_reduce） | 无 | 🔴 缺文档链 | P1 |
| **流式输出** | ✅ 完整SSE | ❌ | 🟡 缺流式 | P1 |
| **工具集成** | ✅ 大量工具 | ❌ | 🟡 非必须 | — |

### 1.4 Haystack（RAG 框架）

| 能力维度 | Haystack | Fusion-KB | 差距 | 优先级 |
|---------|---------|-----------|------|--------|
| **Pipeline** | 可视化Pipeline编排 | 无 | 🔴 缺Pipeline | P1 |
| **文件转换** | 10+转换器 | 7种解析 | 🟡 缺预处理 | P1 |
| **预处理器** | 清洗/标准化/去重 | 无 | 🔴 缺预处理 | P1 |
| **评估** | RAGAS 集成 | 无 | 🟡 缺评估 | P2 |
| **缓存** | 结果缓存 | 无 | 🟡 缺缓存 | P1 |

### 1.5 Chroma（向量数据库）

| 能力维度 | Chroma | Fusion-KB | 差距 | 优先级 |
|---------|--------|-----------|------|--------|
| **过滤** | 元数据过滤 | 无 | 🔴 缺过滤 | P0 |
| **批量操作** | 完整批量API | 基础批量 | 🟡 缺批量删除/更新 | P1 |
| **集合管理** | 集合隔离+配置 | 知识库隔离 | ✅ 持平 | — |
| **客户端** | Python/JS/Go | Python | 🟡 单语言 | P2 |

---

## 二、能力缺口总表（按P0/P1/P2排序）

### P0（核心体验 · 1-2周补齐）

| # | 缺口 | 对标 | 工作量 | 说明 |
|---|------|------|--------|------|
| 1 | **Rerank 重排** | Dify/LlamaIndex | 2天 | 检索结果重排，大幅提升精度 |
| 2 | **混合检索策略** | Dify/LlamaIndex | 2天 | 向量+关键词融合权重，自适应 |
| 3 | **元数据过滤** | Chroma | 1天 | 按文档类型/日期/标签过滤 |
| 4 | **数据库连接器** | LlamaIndex | 2天 | SQLite/PostgreSQL 数据源 |
| 5 | **网页加载器** | LangChain | 1天 | URL 内容抓取索引 |
| 6 | **多种检索器** | LangChain | 3天 | MMR/上下文压缩/融合检索 |

### P1（体验增强 · 1个月补齐）

| # | 缺口 | 对标 | 工作量 | 说明 |
|---|------|------|--------|------|
| 7 | 递归分片 | LangChain | 1天 | 递归字符分割 |
| 8 | 多模型 Embedding | Dify | 1天 | 支持切换 Embedding 模型 |
| 9 | 文档预处理 | Haystack | 2天 | 清洗/标准化/去重 |
| 10 | 多轮对话 RAG | Dify | 3天 | 带历史会话的 RAG |
| 11 | 文档链 | LangChain | 2天 | stuff/refine/map_reduce |
| 12 | 流式输出 SSE | LangChain | 1天 | 流式返回 RAG 结果 |
| 13 | 自动同步 | Dify | 3天 | 文件系统监控自动更新 |
| 14 | 测评缓存 | 自研 | 1天 | SQLite 结果缓存 |
| 15 | 元数据提取 | LlamaIndex | 2天 | 自动提取文档元数据 |

### P2（生态完善 · 2个月+）

| # | 缺口 | 对标 | 说明 |
|---|------|------|------|
| 16 | 图片OCR | Dify | 通过 MLX-VLM 提取图片文字 |
| 17 | 音频转录 | Dify | 通过 Whisper 转录语音 |
| 18 | 性能监控 | Dify | 请求量/延迟/Token 统计看板 |
| 19 | RAGAS 评估 | Haystack | 检索精度/生成质量评估 |
| 20 | 多向量存储 | LangChain | 支持 Chroma 等更多后端 |
| 21 | 多用户 | Dify | 团队协作权限管理 |

---

## 三、核心补齐方案

### 3.1 Rerank 重排（P0）

```python
# fusion_kb/engine/reranker.py
class Reranker:
    """Rerank search results to improve precision.

    Uses fusion-mlx's cross-encoder model via /v1/chat/completions.
    """

    def __init__(self, mlx_url: str = "http://localhost:11434/v1"):
        self.mlx_url = mlx_url

    async def rerank(self, query: str, documents: list[str], top_k: int = 5) -> list[dict]:
        """Rerank documents by relevance to query."""
        pairs = [f"Query: {query}\nDocument: {doc}" for doc in documents]
        scores = []
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            for pair in pairs:
                resp = await client.post(f"{self.mlx_url}/chat/completions", json={
                    "model": "BGE-Reranker",
                    "messages": [{"role": "user", "content": f"Rate relevance (0-10): {pair}"}],
                    "max_tokens": 10,
                    "temperature": 0.0,
                })
                # Parse score from response
                text = resp.json()["choices"][0]["message"]["content"]
                try:
                    score = float(text.strip()[:4])
                except ValueError:
                    score = 5.0
                scores.append(score)

        # Sort by score descending
        indexed = list(enumerate(zip(documents, scores)))
        indexed.sort(key=lambda x: x[1][1], reverse=True)
        return [
            {"index": i, "text": doc, "score": score}
            for i, (doc, score) in indexed[:top_k]
        ]
```

### 3.2 混合检索（P0）

```python
# fusion_kb/engine/hybrid_search.py
class HybridSearch:
    """Hybrid search combining vector similarity and keyword matching."""

    def __init__(self, vector_store, alpha: float = 0.7):
        self.vector_store = vector_store
        self.alpha = alpha  # Vector weight (0.7 = 70% vector, 30% keyword)

    async def search(self, query_vector: list[float], query_text: str,
                     top_k: int = 10, threshold: float = 0.0) -> list[dict]:
        # Vector search
        vector_results = self.vector_store.search(query_vector, top_k=top_k * 2)
        # Keyword search
        keyword_results = self.vector_store.keyword_search(query_text, top_k=top_k * 2)

        # Fuse results with weighted scores
        scores = {}
        for r in vector_results:
            scores[r["id"]] = scores.get(r["id"], 0) + self.alpha * r.get("score", 0)
        for r in keyword_results:
            scores[r["id"]] = scores.get(r["id"], 0) + (1 - self.alpha) * r.get("score", 0)

        # Sort by fused score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [{"id": rid, "score": score} for rid, score in ranked[:top_k] if score >= threshold]
```

### 3.3 数据库连接器（P0）

```python
# fusion_kb/engine/connectors.py
class DatabaseConnector:
    """Connect to SQLite/PostgreSQL databases and index table data."""

    def __init__(self, db_type: str = "sqlite", connection_string: str = ""):
        self.db_type = db_type
        self.connection_string = connection_string

    async def fetch_tables(self, schema: str = "public") -> list[dict]:
        """List all tables and their schemas."""
        if self.db_type == "sqlite":
            import sqlite3
            conn = sqlite3.connect(self.connection_string)
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [{"name": row[0]} for row in cursor.fetchall()]
            conn.close()
            return tables
        # PostgreSQL support
        import asyncpg  # lazy import
        conn = await asyncpg.connect(self.connection_string)
        rows = await conn.fetch("""
            SELECT table_name, column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = $1
        """, schema)
        await conn.close()
        # Group by table
        tables = {}
        for row in rows:
            tname = row["table_name"]
            if tname not in tables:
                tables[tname] = {"name": tname, "columns": []}
            tables[tname]["columns"].append({"name": row["column_name"], "type": row["data_type"]})
        return list(tables.values())
```

---

## 四、补齐路线图

### Week 1-2: P0 核心补齐

```
Day 1-2:  Rerank 重排模块
Day 3-4:  混合检索策略
Day 5:    元数据过滤
Day 6-7:  数据库连接器 + 网页加载器
```

### Week 3-4: P0 收尾 + P1 启动

```
Day 1-2:  多种检索器 (MMR/上下文压缩)
Day 3-4:  递归分片 + 多模型 Embedding
Day 5-6:  文档预处理 + 多轮对话 RAG
Day 7:    测试 + 修复
```

### Week 5-6: P1 收尾

```
Day 1-2:  文档链 + 流式输出 SSE
Day 3-4:  自动同步 + 缓存
Day 5-6:  元数据提取 + 测试
Day 7:    发布 v0.2
```

---

## 五、一句话总结

> **Fusion-KB 当前对标 Dify RAG 有 6 个 P0 缺口（重排/混合检索/过滤/数据库连接器/网页加载/多种检索器），**
> **核心缺失在检索精度和连接器生态。这些补齐后，Fusion-KB 在 Apple Silicon 本地 RAG 场景将具备业界领先能力。**
> **而我们的差异化优势（MLX 原生 Embedding、100% 本地离线、知识库隔离）是 Dify/LlamaIndex/LangChain 无法复制的。**