# Claude RAG 深度洞察报告

> 基于 Anthropic 官方研究、工程实践与 Fusion-RAG 项目实现的深度分析
> 
> 生成日期: 2026-07-28

---

## 目录

1. [RAG 技术全景](#1-rag-技术全景)
2. [Anthropic Contextual Retrieval 核心原理](#2-anthropic-contextual-retrieval-核心原理)
3. [Chunking 策略深度解析](#3-chunking-策略深度解析)
4. [Embedding 模型选型与优化](#4-embedding-模型选型与优化)
5. [混合检索: 向量 + BM25 的融合艺术](#5-混合检索-向量--bm25-的融合艺术)
6. [Reranking: 精排的力量](#6-reranking-精排的力量)
7. [Prompt Caching: 成本优化的关键](#7-prompt-caching-成本优化的关键)
8. [Agentic RAG 架构模式](#8-agentic-rag-架构模式)
9. [多轮 RAG 与文档链](#9-多轮-rag-与文档链)
10. [流式响应与缓存策略](#10-流式响应与缓存策略)
11. [多 Agent 研究系统](#11-多-agent-研究系统)
12. [上下文工程: 上下文腐烂与压缩](#12-上下文工程-上下文腐烂与压缩)
13. [工具使用与 RAG 集成](#13-工具使用与-rag-集成)
14. [Citations API: 引用与溯源](#14-citations-api-引用与溯源)
15. [多模态 RAG: 超越文本](#15-多模态-rag-超越文本)
16. [Extended Thinking 与 RAG 质量](#16-extended-thinking-与-rag-质量)
17. [RAG 评估框架](#17-rag-评估框架)
18. [生产部署模式](#18-生产部署模式)
19. [RAG 方案对比: Naive → Advanced → Agentic](#19-rag-方案对比-naive--advanced--agentic)
20. [Fusion-RAG 实现差距分析](#20-fusion-rag-实现差距分析)
21. [最佳实践与推荐架构](#21-最佳实践与推荐架构)
22. [Anthropic RAG 完整文章索引](#22-anthropic-rag-完整文章索引)
23. [参考文献](#23-参考文献)

---

## 1. RAG 技术全景

### 1.1 为什么需要 RAG

大语言模型(LLM)存在三个根本限制:

- **知识截止**: 训练数据有时效性,无法获取最新信息
- **幻觉问题**: 模型倾向生成看似合理但事实上错误的内容
- **私有知识**: 企业内部文档、代码库等不在训练数据中

RAG(Retrieval-Augmented Generation)通过**检索-增强-生成**三步流程解决这些问题:

```
用户查询 → 检索相关文档 → 注入上下文 → LLM 生成回答
```

### 1.2 RAG 进化路径

| 代际 | 特征 | 检索方式 | 典型问题 |
|------|------|----------|----------|
| RAG 1.0 | 纯向量检索 | Embedding 相似度 | 丢失精确匹配 |
| RAG 1.5 | 混合检索 | Embedding + BM25 | 上下文断裂 |
| RAG 2.0 | 上下文检索 | Contextual Embedding + Contextual BM25 | 成本较高 |
| RAG 2.0+ | 上下文检索 + 精排 | + Reranking + Prompt Caching | 延迟增加 |

### 1.3 RAG vs 长上下文

Anthropic 的明确建议:

> **如果知识库 < 200K tokens(约 500 页), 直接把全部内容放进 prompt, 配合 Prompt Caching, 无需 RAG。**

200K tokens 是 Claude 的上下文窗口。Prompt Caching 让这种方案的延迟降低 >2x, 成本降低最高 90%。只有当知识库超出上下文窗口时, 才需要 RAG。

---

## 2. Anthropic Contextual Retrieval 核心原理

### 2.1 核心问题: 上下文断裂

传统 RAG 的致命缺陷是**分块时丢失上下文**。

**示例**: 金融知识库中, 用户问 "ACME Corp 2023 Q2 的营收增长是多少?"

原始 chunk:
```
"The company's revenue grew by 3% over the previous quarter."
```

这个 chunk 缺少:
- 哪个公司? → ACME Corp
- 哪个时间段? → Q2 2023
- 前一季度营收? → $314M

Embedding 模型看到的是孤立的文本片段, 无法建立与原始文档的联系。

### 2.2 Contextual Retrieval 方案

**核心思想**: 在每个 chunk 前添加由 LLM 生成的简短上下文说明。

**上下文化后的 chunk**:
```
"This chunk is from an SEC filing on ACME corp's performance in Q2 2023;
 the previous quarter's revenue was $314 million.
 The company's revenue grew by 3% over the previous quarter."
```

### 2.3 上下文生成 Prompt

Anthropic 使用 Claude 3 Haiku 生成上下文, Prompt 模板:

```
<document>
{{WHOLE_DOCUMENT}}
</document>
Here is the chunk we want to situate within the whole document
<chunk>
{{CHUNK_CONTENT}}
</chunk>
Please give a short succinct context to situate this chunk within the
overall document for the purposes of improving search retrieval of the chunk.
Answer only with the succinct context and nothing else.
```

关键设计点:
- 使用 `<document>` 和 `<chunk>` XML 标签明确分隔
- 要求 "short succinct context", 通常 50-100 tokens
- "Answer only with the succinct context" — 避免模型输出无关内容

### 2.4 两种上下文化技术

#### Contextual Embeddings

```
contextualized_chunk = context + original_chunk
embedding = embed(contextualized_chunk)
```

将上下文信息与原始 chunk 拼接后再做 embedding, 使向量编码包含文档级别的语义。

#### Contextual BM25

```
contextualized_chunk = context + original_chunk
bm25_index.add(contextualized_chunk)
```

将上下文信息与原始 chunk 拼接后加入 BM25 索引, 使关键词匹配也能命中上下文中的实体。

### 2.5 性能数据 (Anthropic 官方基准)

| 方案 | 检索失败率 (1 - Recall@20) | 相对改善 |
|------|---------------------------|----------|
| 基线 (Embedding + BM25) | 5.7% | — |
| + Contextual Embeddings | 3.7% | ↓ 35% |
| + Contextual Embeddings + Contextual BM25 | 2.9% | ↓ 49% |
| + Contextual Emb + Contextual BM25 + Reranking | 1.9% | ↓ 67% |

**关键洞察**: 所有收益可叠加 — 上下文化 + BM25 + Reranking 组合效果最强。

### 2.6 成本分析

使用 Prompt Caching 后:

| 参数 | 值 |
|------|-----|
| chunk 大小 | ~800 tokens |
| 文档大小 | ~8K tokens |
| 上下文指令 | ~50 tokens |
| 每个 chunk 生成上下文 | ~100 tokens |
| **成本** | **$1.02 / 百万文档 tokens** |

Prompt Caching 的关键作用: 文档只需加载到缓存一次, 后续每个 chunk 的上下文生成都复用缓存, 极大降低输入 token 开销。

### 2.7 为什么其他方法效果差

Anthropic 实验了多种方案:

| 方案 | 效果 | 原因 |
|------|------|------|
| 添加通用文档摘要 | 有限 | 摘要太泛, 无法定位具体 chunk |
| 假设文档嵌入(HyDE) | 低 | 生成的假设文档可能偏离真实内容 |
| 基于摘要的索引 | 低 | 摘要丢失细节, 匹配粒度不对 |
| **Contextual Retrieval** | **显著** | chunk 级精准上下文, 保留细节 |

---

## 3. Chunking 策略深度解析

### 3.1 分块的重要性

分块是 RAG 流水线的起点, 直接影响:
- 检索精度: chunk 太大 → 语义模糊; 太小 → 上下文不足
- Embedding 质量: 语义完整性取决于 chunk 的边界
- Token 预算: 每个检索结果消耗 prompt 空间

### 3.2 三大分块策略

#### Fixed-Size Chunking

```
[text][text][text] → [chunk1][chunk2][chunk3]  (固定大小 + overlap)
```

- 最简单, 最可预测
- 适合: 无明确结构的文本(日志、邮件)
- 缺点: 可能切断句子/段落

**参数建议**:
- chunk_size: 256-512 tokens
- overlap: chunk_size 的 10-15%

#### Semantic Chunking

```
## Section 1\nContent...\n## Section 2\nContent...  →  按 Markdown 标题分割
```

- 按语义边界(标题、段落、章节)分割
- 适合: 结构化文档(Markdown、技术文档)
- 优势: 保持语义完整性

#### Code-Aware Chunking

```
def func1(): ... \nclass MyClass: ...  →  按函数/类定义分割
```

- 按代码结构(函数、类、方法)分割
- 适合: 源代码文件
- 支持多语言: Python(def/class)、Go(func)、JS(function)、Java(public)

### 3.3 递归分块 (LangChain 模式)

Fusion-RAG 的 `RecursiveChunker` 实现了 LangChain 风格的递归分割:

```python
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
```

逻辑:
1. 先用 `\n\n`(段落) 尝试分割
2. 如果某段仍超长, 降级用 `\n`(行)
3. 继续降级用 `. `(句子)、空格、字符
4. 每级都有 overlap 保持上下文

### 3.4 Contextual Retrieval 对 Chunking 的影响

上下文化后, chunk 边界的选择变得**更灵活**:
- 传统 RAG: 需要在 chunk 中包含足够上下文 → 倾向更大 chunk
- Contextual RAG: LLM 自动补充上下文 → 可以用更小 chunk 获得更精细检索

**推荐**: Contextual Retrieval 下, chunk_size 可降至 200-400 tokens, 通过更细粒度提高检索精度。

---

## 4. Embedding 模型选型与优化

### 4.1 Anthropic 推荐排名

| 排名 | 模型 | 提供商 | 特点 |
|------|------|--------|------|
| 1 | Gemini Text 004 | Google | 最佳综合性能 |
| 2 | Voyage 3 | Voyage AI | 紧随其后, 也有 Reranker |
| 3 | OpenAI text-embedding-3-large | OpenAI | 通用性强 |

### 4.2 离线场景: BGE-M3

Fusion-RAG 使用 BGE-M3 作为默认 embedding 模型, 通过 fusion-mlx API 调用:

- 多语言支持(中英文)
- 多功能: Dense + Sparse + ColBERT
- 1024 维向量
- Apple Silicon 本地运行, 无需网络

### 4.3 Embedding 维度与检索质量

| 维度 | 特点 |
|------|------|
| 384 | 小模型, 速度快, 精度低 |
| 768 | 平衡选择 |
| 1024 | 高精度, 推荐 |
| 1536+ | 收益递减, 存储成本高 |

### 4.4 批量 Embedding 优化

Fusion-RAG 的 `EmbeddingClient` 实现:

```python
batch_size = 16          # 每批处理 16 条
semaphore = Semaphore(4) # 最多 4 个并发请求
max_retries = 3          # 失败重试 3 次
```

**建议优化**:
- 增加 batch_size 到 32-64(BGE-M3 支持)
- 实现自适应批量: 根据 embedding 服务负载动态调整
- 添加 embedding 结果缓存(相同文本 → 相同向量)

---

## 5. 混合检索: 向量 + BM25 的融合艺术

### 5.1 为什么需要混合检索

| 场景 | 向量检索 | BM25 |
|------|----------|------|
| "ACME Corp Q2 营收" | 语义理解 | 精确匹配 "ACME" |
| "错误代码 TS-999" | 可能泛化到其他错误码 | 精确匹配 "TS-999" |
| "如何提高模型性能" | 理解意图 | 关键词太泛 |

**Anthropic 结论**: Embedding + BM25 > 单独 Embedding

### 5.2 融合策略: Alpha 加权

Fusion-RAG 的 `HybridSearch` 实现:

```python
alpha = 0.7  # 向量权重
final_score = alpha * vector_score + (1 - alpha) * keyword_score
```

| alpha | 行为 |
|-------|------|
| 1.0 | 纯向量检索 |
| 0.7 | 偏重语义(推荐默认) |
| 0.5 | 均衡 |
| 0.3 | 偏重关键词 |
| 0.0 | 纯关键词检索 |

### 5.3 Reciprocal Rank Fusion (RRF)

RRF 是另一种融合方法, 基于排名而非原始分数:

```python
rrf_score = sum(1 / (k + rank_i) for each ranking)
# k 通常 = 60
```

优势:
- 不需要归一化不同量纲的分数
- 对极端值更鲁棒
- 业界广泛使用

**建议**: Fusion-RAG 可增加 RRF 作为融合策略选项。

### 5.4 Fusion-RAG 当前 BM25 实现的问题

当前 `VectorStore.keyword_search()` 是简化的 TF 实现:

```python
# 当前: 简单子串计数
score = text.count(query_lower) / max(len(text.split()), 1)
```

**缺失**:
- 无 IDF(逆文档频率)权重
- 无 BM25 饱和函数
- 无文档长度归一化
- 全表扫描(零向量搜索 + Python 过滤), 性能差

**建议**: 使用 LanceDB 的全文搜索或引入 whoosh/rank_bm25 库实现真正的 BM25。

---

## 6. Reranking: 精排的力量

### 6.1 为什么需要 Reranking

初始检索(top-150)可能包含大量低相关性 chunk。Reranking 是二次筛选, 确保 top-20 高质量。

**Anthropic 数据**:
- 初始检索失败率: 2.9%
- + Reranking: 1.9%
- **额外改善: 34%**

### 6.2 Reranking 流程

```
初始检索(top-150) → Reranker 评分 → 取 top-20 → 注入 prompt
```

### 6.3 Reranker 类型

#### 专用 Reranker 模型

| 模型 | 提供商 | 特点 |
|------|--------|------|
| Cohere Reranker | Cohere | Anthropic 基准使用, API 调用 |
| Voyage Reranker | Voyage AI | 与 Voyage Embedding 协同 |
| BGE-Reranker | BAAI | 开源, 可本地运行 |
| Jina Reranker | Jina AI | 轻量, 多语言 |

#### LLM-as-Judge Reranker

Fusion-RAG 当前实现: 用 LLM 对每个文档评分 0-10:

```python
prompt = f"Rate the relevance of the following document to the query
           on a scale of 0 to 10..."
```

**问题**:
- 每个文档一次 LLM 调用, top-150 → 150 次 LLM 请求
- 延迟极高: 顺序评分, 无批处理
- 评分不稳定: LLM 输出 float 解析可能失败
- 成本高: 每次检索都消耗大量 token

**建议**:
1. 短期: 添加专用 BGE-Reranker 本地模型(fusion-mlx 可托管)
2. 中期: 实现 batch reranking, 一次 LLM 调用评分多个文档
3. 长期: 实现 Cross-Encoder 架构的专用 reranker

### 6.4 Reranking 的延迟/成本权衡

| 评分数量 | 延迟增加 | 成本增加 | 质量改善 |
|----------|----------|----------|----------|
| top-20 | 低 | 低 | 有限 |
| top-50 | 中 | 中 | 较好 |
| top-100 | 高 | 高 | 好 |
| top-150 | 很高 | 很高 | 最佳(Anthropic 推荐) |

**建议**: 默认 top-100, 可配置; 对延迟敏感场景降至 top-50。

---

## 7. Prompt Caching: 成本优化的关键

### 7.1 原理

Claude API 的 Prompt Caching 机制:
- 首次请求: 完整处理 prompt, 缓存前缀
- 后续请求: 如果 prompt 前缀匹配缓存, 跳过已缓存部分的计算
- 缓存命中: 延迟降低 >2x, 成本降低最高 90%

### 7.2 在 Contextual Retrieval 中的应用

生成上下文时, 同一文档的多个 chunk 共享文档前缀:

```
请求 1: <document>[整个文档]</document> + <chunk>[chunk_1]</chunk> + prompt
请求 2: <document>[整个文档]</document> + <chunk>[chunk_2]</chunk> + prompt  ← 缓存命中!
请求 3: <document>[整个文档]</document> + <chunk>[chunk_3]</chunk> + prompt  ← 缓存命中!
```

8K 文档, 800 token chunk → ~10 个 chunk, 只有第 1 个需要完整处理, 后续 9 个都命中缓存。

### 7.3 Fusion-RAG 中的缓存应用

当前 `ResultCache` 使用 SQLite 缓存 RAG 问答结果:

```python
# 查询 hash + 上下文 hash → 缓存命中
query_hash = md5(query)
context_hash = md5(context)
```

**建议扩展**:
1. 在上下文生成阶段实现文档级缓存(类似 Claude Prompt Caching)
2. 缓存 embedding 结果: 相同文本 → 复用向量
3. 缓存 reranking 结果: 相同 query + 相同文档集 → 复用排序

### 7.4 缓存失效策略

| 场景 | 策略 |
|------|------|
| 文档更新 | 删除相关 chunk 的缓存, 重新 embedding |
| 知识库重建 | 全量缓存失效 |
| 模型切换 | embedding 缓存全部失效(维度可能不同) |
| TTL 过期 | 定期清理旧缓存(默认 7 天) |

---

## 8. Agentic RAG 架构模式

### 8.1 Anthropic 的 Agent 设计哲学

来自 "Building Effective Agents" (2024.12):

> **最成功的实现不使用复杂框架, 而是简单、可组合的模式。**

核心原则:
1. **保持简单**: 能用单次 LLM 调用解决, 就不用 Agent
2. **透明性**: 显式展示 Agent 的规划步骤
3. **精心设计工具接口**: ACI(Agent-Computer Interface) 和 API 一样重要

### 8.2 五种 Agent 工作流模式

#### 1) Prompt Chaining (提示链)

```
[LLM Step 1] → 检查点 → [LLM Step 2] → 检查点 → [最终输出]
```

RAG 应用: 文档解析 → 摘要 → 检索 → 生成

#### 2) Routing (路由)

```
[输入] → [分类器] → 路径A: 技术文档检索
                  → 路径B: 代码搜索
                  → 路径C: 通用知识
```

RAG 应用: 根据查询类型选择不同的检索策略

#### 3) Parallelization (并行化)

- **Sectioning**: 独立子任务并行执行
- **Voting**: 多次执行取投票/集成

RAG 应用: 并行检索多个知识库, 多角度评估文档相关性

#### 4) Orchestrator-Workers (编排-执行)

```
[Orchestrator LLM] → 分析任务 → 分配子任务
                                  |
                    [Worker 1] [Worker 2] [Worker N]
                                  |
                    [Orchestrator] → 综合结果
```

RAG 应用: 复杂查询拆分为多轮检索, 动态决定检索范围

#### 5) Evaluator-Optimizer (评估-优化)

```
[Generator LLM] → 初始回答 → [Evaluator LLM] → 评估
                    ^                              |
                    +---- 反馈 --------------------+
```

RAG 应用: 检索质量评估 → 调整检索策略 → 重新生成

### 8.3 Agentic RAG 架构

```
用户查询
    |
[查询分析 Agent] → 理解意图, 提取关键词
    |
[路由 Agent] → 选择检索策略
    |
[检索 Agent] → 执行混合检索 + Reranking
    |
[评估 Agent] → 检查检索结果质量
    | (质量不足)
[扩展 Agent] → 改写查询, 扩大检索范围
    | (质量足够)
[生成 Agent] → 基于上下文生成回答
    |
[验证 Agent] → 事实核查, 引用验证
    |
最终输出
```

### 8.4 Fusion-RAG 当前的 Agent 化程度

| 能力 | 状态 | 说明 |
|------|------|------|
| 查询分析 | 缺失 | 无查询改写/扩展 |
| 路由 | 缺失 | 无多策略选择 |
| 检索执行 | 已有 | 混合检索 + MMR |
| 评估 | 缺失 | 无检索质量评估 |
| 自适应扩展 | 缺失 | 无查询重试机制 |
| 生成 | 已有 | 多轮对话 + 文档链 |
| 验证 | 缺失 | 无事实核查 |

---

## 9. 多轮 RAG 与文档链

### 9.1 多轮对话 RAG

Fusion-RAG 的 `MultiTurnRAG` 实现:

```python
# 对话历史管理
self._history: list[dict] = []  # 最多保留 20 条

# 上下文注入
messages = [
    {"role": "system", "content": "Answer based on context. Cite sources."},
    ...history,
    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
]
```

### 9.2 三种文档链策略

#### Stuff (全量注入)

```python
context = "\n\n".join(all_docs)
# 单次 LLM 调用, 所有文档一次性注入
```

- 最简单
- 适合: 文档总量 < 上下文窗口
- 缺点: 文档太多时超出 token 限制

#### Refine (迭代精炼)

```python
answer = ""
for doc in docs:
    answer = llm(f"Previous answer: {answer}\nNew doc: {doc}\nRefine.")
```

- 逐文档迭代改进答案
- 适合: 文档间有补充关系
- 缺点: 顺序执行, 延迟高; 早期文档影响过大

#### Map-Reduce (映射-归约)

```python
# Phase 1: Map - 并行处理每个文档
summaries = await gather(*[llm(f"Summarize doc for query: {doc}") for doc in docs])
# Phase 2: Reduce - 合并摘要
final = llm(f"Based on summaries: {summaries}\nFinal answer:")
```

- 并行处理, 速度最快
- 适合: 文档量大, 独立性强
- 缺点: 两次 LLM 调用; 摘要可能丢失细节

### 9.3 策略选择指南

| 场景 | 推荐策略 | 原因 |
|------|----------|------|
| 文档 < 5 个, 总量 < 4K tokens | Stuff | 简单高效 |
| 文档 5-20 个, 有层次关系 | Refine | 迭代改进 |
| 文档 > 20 个, 相互独立 | Map-Reduce | 并行加速 |
| 文档 > 50 个 | 先检索 top-K, 再 Stuff/Refine | 控制 token 预算 |

### 9.4 多轮 RAG 的高级模式

#### 查询改写 (Query Rewriting)

```python
# 原始查询: "它的增长呢?"
# 上下文: 之前讨论了 ACME Corp 的营收
# 改写后: "ACME Corp 的营收增长是多少?"
```

当前 Fusion-RAG **缺失** 此能力, 历史对话直接拼接, 未做指代消解。

#### 混合记忆 (Hybrid Memory)

```python
# 短期记忆: 最近 N 轮对话
# 长期记忆: 对话摘要 + 关键实体
# 工作记忆: 当前检索结果
```

---

## 10. 流式响应与缓存策略

### 10.1 SSE 流式输出

Fusion-RAG 的 `SSEStreamer` 实现:

```python
async with client.stream("POST", url, json={..., "stream": True}) as resp:
    async for line in resp.aiter_lines():
        if line.startswith("data: "):
            # 解析 SSE 数据, 转发给客户端
```

SSE 格式:
```
data: {"content": "根据"}

data: {"content": "文档"}

data: {"content": "记录..."}

data: [DONE]
```

### 10.2 流式场景下的 Reranking

流式响应与 Reranking 的时序:

```
[检索] → [Reranking(同步)] → [开始流式输出]
```

Reranking 必须在流式输出前完成(需要完整排序), 这增加了首 token 延迟。

**优化**: 可以考虑两阶段流式:
1. 先返回未经精排的快速结果(低延迟)
2. Reranking 完成后追加精排结果

### 10.3 元数据提取

`MetadataExtractor` 使用 LLM 自动提取文档元数据:

```python
# 提取字段: title, author, date, language, topics, summary
# 用于: 过滤、排序、知识图谱构建
```

### 10.4 结果缓存

`ResultCache` 使用 SQLite 存储:

```sql
CREATE TABLE rag_cache (
    query_hash TEXT PRIMARY KEY,   -- MD5(查询)
    context_hash TEXT,             -- MD5(上下文)
    answer TEXT,                   -- 缓存的回答
    sources TEXT,                  -- JSON 来源列表
    created_at REAL                -- 创建时间
);
```

缓存 key 设计: query_hash + context_hash 双重匹配, 确保相同查询在不同上下文下不会错误命中。

---

## 11. 多 Agent 研究系统

### 11.1 Anthropic 的 Multi-Agent Research 架构

2025年6月, Anthropic 公布了其 Research 功能背后的多 Agent 架构:

```
[用户查询]
    |
[LeadResearcher Agent (Opus)]
    |→ 规划研究策略
    |→ 生成子任务
    |
    +→ [SubAgent 1 (Sonnet)] → 搜索 + 分析
    +→ [SubAgent 2 (Sonnet)] → 搜索 + 分析
    +→ [SubAgent 3 (Sonnet)] → 搜索 + 分析
    |
    +→ [CitationAgent] → 处理文档, 识别引用位置
    |
[LeadResearcher] → 综合结果 → 生成报告
```

**核心洞察**: "搜索的本质是压缩: 从海量语料中蒸馏出洞见。子 Agent 通过各自独立的上下文窗口并行运行, 实现压缩。"

### 11.2 与传统 RAG 的根本区别

| 维度 | 传统 RAG | Multi-Agent Research |
|------|----------|---------------------|
| 检索策略 | 静态检索: 查询 → 固定 top-K chunk | 动态搜索: 多步迭代, 自适应调整 |
| 上下文 | 单一上下文窗口 | 每个 Agent 独立上下文, 互不干扰 |
| 搜索深度 | 1 次检索 | 多轮搜索, 逐层深入 |
| 适应性 | 查询固定不改写 | 根据中间结果动态改写查询 |
| 性能基准 | — | 比单 Agent Opus 4 提升 90.2% |

### 11.3 Multi-Agent RAG 的关键设计

#### Lead Agent 职责
- 接收用户查询, 制定研究计划
- 分配子任务给 SubAgents
- 综合所有子 Agent 结果
- 生成最终报告(带引用)

#### SubAgent 职责
- 执行具体搜索任务
- 分析搜索结果
- 返回压缩后的发现(不是原始文本)

#### CitationAgent 职责
- 处理长文档
- 识别需要引用的确切位置
- 确保引用准确性

### 11.4 Token 消耗模型

| 交互类型 | 相对 Token 消耗 |
|----------|----------------|
| 普通对话 | 1x |
| 单 Agent | ~4x |
| 多 Agent 系统 | ~15x |

**关键发现**: 提升模型智能的收益 > 翻倍 token 预算的收益。

### 11.5 对 Fusion-RAG 的启示

1. 当前 Fusion-RAG 是单 Agent 架构(检索 → 生成)
2. 可引入 "Lead-Sub" 模式: 复杂查询拆分为多路并行检索
3. fusion-mlx 可同时运行多个推理实例, 支持 SubAgent 并行
4. 引用验证: 当前无事实核查, 可添加 CitationAgent

---

## 12. 上下文工程: 上下文腐烂与压缩

### 12.1 上下文腐烂 (Context Rot)

Anthropic 官方确认的关键现象:

> "Needle-in-a-haystack 基准测试揭示了上下文腐烂: 随着上下文窗口中 token 数量增加, 模型准确回忆信息的能力下降。虽然某些模型的退化更温和, 但这一特征在所有模型中都存在。"

**原因**: Transformer 的 n-squared 逐对注意力机制在上下文大小和注意力聚焦之间产生自然张力。

**影响**:
- 直接否定 "长上下文解决一切" 的简单思路
- RAG + 精选 top-K chunk > 塞满整个上下文窗口
- 即使知识库 < 200K tokens, 也应优选高质量检索而非全量注入

### 12.2 有效上下文工程原则

来自 Anthropic "Effective Context Engineering for AI Agents" (2025.09):

**核心定义**: "找到最小的高信号 token 集, 最大化期望结果的概率。"

**八条原则**:

1. **系统提示要极其清晰** — 使用简单直接的语言, 适当的抽象层级
2. **工具返回 token 高效的信息** — 不要把原始大块数据塞进上下文
3. **Just-In-Time 检索** — 运行时按需加载, 不要预加载所有数据
4. **混合策略** — 预检索一些数据(速度), 同时允许 Agent 自主探索(深度)
5. **长期任务用压缩** — 摘要对话历史, 重新初始化
6. **保留关键信息** — 架构决策、未解决 bug、实现细节
7. **丢弃冗余** — 重复工具输出、过时调试日志
8. **多 Agent 分治** — 每个 Agent 有独立上下文窗口, 避免单个窗口过载

### 12.3 Just-In-Time vs Pre-Chunking 检索

| 策略 | 传统 RAG | 现代 Agent RAG |
|------|----------|---------------|
| 数据准备 | 预处理所有数据为 chunk + embedding | 维护轻量标识符(文件路径、链接) |
| 检索时机 | 查询时一次性检索 | Agent 运行时按需动态加载 |
| 上下文管理 | 固定 top-K chunk | 渐进式披露, 按需深入 |
| Token 效率 | 可能包含无关 chunk | 仅加载需要的最小数据 |

**混合推荐**: "预检索一些数据确保速度, 同时允许 Agent 自主进一步探索。"

### 12.4 压缩 (Compaction) 策略

长期任务中上下文窗口会填满, 需要压缩:

```python
# Claude Code 的压缩策略
class ContextCompactor:
    def compact(self, conversation: list) -> str:
        """压缩对话历史为摘要"""
        preserve = [
            "架构决策",
            "未解决的 bug",
            "实现细节",
            "当前待修改文件",
        ]
        discard = [
            "重复的工具输出",
            "过时的调试日志",
            "已解决的中介状态",
        ]
        summary = llm.summarize(conversation, preserve=preserve, discard=discard)
        return summary
```

Claude Code 的具体做法: 保留摘要 + 最近 5 个访问过的文件。

### 12.5 对 Fusion-RAG 的影响

| 当前问题 | 改进方向 |
|----------|----------|
| 无上下文长度管理 | 添加 token 计数, 超限时压缩历史 |
| 固定 top-K chunk | 根据查询复杂度动态调整 K |
| 无 Just-In-Time 检索 | 实现 Agent 按需加载, 而非预加载 |
| 全量文档链注入 | 根据上下文预算选择 Stuff/Refine/Map-Reduce |

---

## 13. 工具使用与 RAG 集成

### 13.1 Tool Search Tool (2025.11)

Anthropic 的重大创新: Agent 不再预加载所有工具定义, 而是按需搜索。

**原理**:
```python
# 传统: 所有工具定义都在 prompt 中
tools = [tool_1, tool_2, ..., tool_100]  # 100 个工具 = 大量 token

# Tool Search Tool: 只加载核心工具 + 搜索入口
tools = [core_tool_1, core_tool_2, tool_search_tool]  # 3 个
# Agent 需要时搜索: tool_search("database query") → 返回相关工具定义
```

**效果**:
- **Token 使用减少 85%**
- Opus 4 准确率: 49% → 74%
- Opus 4.5 准确率: 79.5% → 88.1%

### 13.2 程序化工具调用 (Programmatic Tool Calling)

Claude 可以通过代码编排工具, 而非逐个 API 往返:

```python
# 传统: 逐个工具调用
result_1 = call_tool("search", query="error logs")
result_2 = call_tool("filter", data=result_1, type="critical")
result_3 = call_tool("aggregate", data=result_2)

# 程序化: Claude 写代码调用多个工具
code = """
logs = search("error logs")
critical = [l for l in logs if l.type == "critical"]
summary = aggregate(critical)
return summary
"""
# 单次执行, 中间结果不进入模型上下文
```

**效果**: 某些 MCP 工作流中 token 减少 **98.7%**。

### 13.3 为 Agent 编写有效工具

Anthropic 的工具设计原则:

1. **选择正确的工具实现** — 不该用工具的别用工具
2. **命名空间隔离** — 清晰的工具边界
3. **返回有意义的上下文** — 工具返回值应包含 Agent 需要的信息
4. **优化工具响应的 token 效率** — 压缩返回数据
5. **Prompt Engineering 工具描述** — 工具描述和规范本身就是 prompt

### 13.4 MCP (Model Context Protocol) 与 RAG

MCP 为 RAG 提供了标准化接口:

```python
# MCP 服务器作为代码 API
class RAGMCPServer:
    def search(self, query: str) -> list[dict]:
        """搜索知识库"""
        pass

    def get_document(self, doc_id: str) -> dict:
        """获取完整文档"""
        pass

    def list_knowledge_bases(self) -> list[str]:
        """列出可用知识库"""
        pass
```

MCP 的渐进式披露: Agent 先看到知识库列表, 按需搜索和加载文档, 避免一次性注入所有数据。

### 13.5 对 Fusion-RAG 的启示

| 能力 | 当前状态 | 建议 |
|------|----------|------|
| MCP 接口 | 缺失 | 实现 MCP Server, 让 Claude/Cursor 等 Agent 可直接调用 |
| 工具搜索 | 缺失 | 对大量知识库实现按需发现 |
| 程序化调用 | 缺失 | 支持 Claude 写代码调用 RAG API |
| Token 高效响应 | 缺失 | API 返回压缩结果, 非 chunk 全文 |

---

## 14. Citations API: 引用与溯源

### 14.1 Anthropic Citations API (Beta)

Claude 支持精确引用, 指向源文档的具体段落:

```python
message = client.messages.create(
    model="claude-sonnet-4-20250514",
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "document",
                "source": {
                    "type": "text",
                    "media_type": "text/plain",
                    "data": document_text
                },
                "title": "Source Document Title",
                "context": "Brief description of the document"
            },
            {"type": "text", "text": "Question about the document"}
        ]
    }]
)
# 响应中包含 citations, 引用源文档的具体字符范围
```

### 14.2 引用在 RAG 中的价值

| 能力 | 无引用 | 有引用 |
|------|--------|--------|
| 用户信任 | 低 — 无法验证 | 高 — 可点击溯源 |
| 幻觉检测 | 困难 | 自动 — 引用不匹配即可发现 |
| 调试 | 靠猜测 | 精确定位到源 chunk |
| 合规 | 不满足审计要求 | 满足金融/法律/医疗审计 |

### 14.3 多 Agent 引用模式

Anthropic Research 系统使用专门的 CitationAgent:

```
[LeadResearcher] → 生成报告
[CitationAgent] → 处理文档, 识别引用位置
                 → 确保所有声明都有源归属
```

### 14.4 对 Fusion-RAG 的启示

当前 Fusion-RAG 的 RAG 响应包含 "Cite sources" 指令, 但无结构化引用:
- 缺失: 无 character-level 引用定位
- 缺失: 无引用验证机制
- 建议: 在 chunk metadata 中存储字符偏移, 响应中返回结构化引用

---

## 15. 多模态 RAG: 超越文本

### 15.1 Claude 的多模态能力

| 输入类型 | 支持格式 | RAG 应用 |
|----------|----------|----------|
| 图像 | PNG, JPG, GIF, WebP | 图表理解、OCR |
| PDF | 直接上传 | 文档解析 + RAG |
| 文本 | Plain text, Markdown | 传统 RAG |

### 15.2 多模态 RAG 模式

#### 图像增强检索

```
[用户查询] → [检索文本 chunk + 关联图像]
    → [文本 chunk + 图像一起传入 Claude]
    → [Claude 理解图像内容, 结合文本生成回答]
```

#### OCR + RAG

```
[扫描文档] → [Claude Vision OCR] → [提取文本]
    → [传统 RAG Pipeline] → [生成回答]
```

#### 图表理解

```
[财务报表 PDF] → [渲染为图像] → [Claude 分析图表]
    → [结合文本 RAG 结果] → [综合回答]
```

### 15.3 多模态最佳实践

- 裁剪相关图像区域, 提高准确性
- 使用子 Agent 处理复杂图像分析
- 结合 Vision + Tools 实现迭代视觉分析
- PDF 技能: Claude Agent Skills 包含 PDF 处理能力

### 15.4 Fusion-RAG 当前的多模态状态

| 能力 | 状态 |
|------|------|
| PDF 文本提取 | 已有 (PyMuPDF) |
| 图像 OCR | 缺失 |
| 图表理解 | 缺失 |
| 图像 embedding | 缺失 |
| PDF 渲染为图像 | 缺失 |

---

## 16. Extended Thinking 与 RAG 质量

### 16.1 Extended Thinking vs Think Tool

| 维度 | Extended Thinking | Think Tool |
|------|-------------------|------------|
| 时机 | 生成响应**之前**深度思考 | 生成过程中**穿插**思考 |
| 适用 | 编程、数学、物理(信息完整) | 长工具链、策略密集、顺序决策 |
| 更新 | 2025.12 后推荐优先用 Extended Thinking | 仍适用于特定场景 |
| RAG 适用 | 查询分解、多跳推理 | 检索后评估、迭代检索 |

### 16.2 Think Tool 对 RAG 的改善

Tau-Bench 评估(客户服务场景):

| 配置 | 航空 pass^1 | 零售 pass^1 |
|------|-------------|-------------|
| 基线 | 0.332 | 0.783 |
| Think Tool | 0.404 | 0.812 |
| Think Tool + 优化 Prompt | 0.584 | — |
| **相对改善** | **54%** | **4%** |

### 16.3 RAG 场景下的推荐 Prompt 模式

```
## Using the think tool
Before taking any action or responding to the user after receiving tool results,
use the think tool as a scratchpad to:
- List the specific rules that apply to the current request
- Check if all required information is collected
- Verify that the planned action complies with all policies
- Iterate over tool results for correctness
```

### 16.4 Interleaved Thinking (交错思考)

Extended Thinking 的一个关键特性: **在工具调用之间思考**。

RAG 场景:
```
[查询] → [思考: 如何分解查询] → [检索工具调用]
    → [思考: 检索结果是否足够] → [可能再次检索]
    → [思考: 综合分析结果] → [生成回答]
```

这比传统 RAG 的 "一次检索 → 直接生成" 更强大, Claude 可以在每步之间推理。

### 16.5 对 Fusion-RAG 的启示

| 当前 | 建议 |
|------|------|
| 检索后直接生成 | 添加 "评估" 步骤: 检查检索结果质量 |
| 单次检索 | 支持迭代检索: 质量不足时自动扩展 |
| 无思考过程 | 利用 fusion-mlx 的思考能力, 在生成前推理 |

---

## 17. RAG 评估框架

### 17.1 Anthropic 的评估哲学

来自 "Demystifying Evals for AI Agents" (2026.01):

**关键定义**:

| 概念 | 定义 |
|------|------|
| Task | 单个测试, 有定义的输入和成功标准 |
| Trial | 对 Task 的一次尝试(运行多次以评估一致性) |
| Grader | 评分逻辑(一个 Task 可有多个 Grader) |
| Transcript | Trial 的完整记录(messages 数组) |
| Outcome | 最终环境状态(如数据库中是否存在预订) |
| Eval Harness | 端到端运行评估的基础设施 |
| Eval Suite | 衡量特定能力的 Task 集合 |

### 17.2 pass^k 指标

Anthropic 推荐使用 **pass^k** 而非 pass@k:

| 指标 | 含义 | 侧重点 |
|------|------|--------|
| pass@k | k 次尝试中至少 1 次成功 | 最好情况 |
| pass^k | k 次尝试全部成功 | 一致性和可靠性 |

RAG 系统需要**一致性** — 用户期望每次都得到正确答案, 不是偶尔正确。

### 17.3 RAG 专属评估维度

| 维度 | 指标 | 说明 |
|------|------|------|
| 检索准确性 | Recall@K, MRR | 检索到相关 chunk 的能力 |
| 回答落地性 | Groundedness | 回答是否基于检索到的上下文 |
| 回答完整性 | Completeness | 是否覆盖了所有相关信息 |
| 引用准确性 | Citation Accuracy | 引用是否正确指向源材料 |
| 一致性 | pass^k | 多次运行是否得到相同结果 |

### 17.4 多层评分策略

```
Level 1: 精确字符串匹配 — 简单场景
Level 2: LLM-as-Judge — 复杂回答评估
Level 3: 环境状态验证 — 检查实际效果(如数据库状态)
```

**避免过度严格的验证器**: 拒绝合理但措辞不同的回答。

### 17.5 评估工作流

```
1. 生成评估任务 — 基于真实用户场景
2. 程序化运行 — 直接 LLM API 调用, 非交互
3. 简单 Agent 循环 — while-loop 包装 LLM + Tool 交替
4. 输出推理块 — Agent 在工具调用和响应前输出推理
5. 跟踪多指标 — 顶层准确率 + 延迟 + Token 用量 + 单任务成本
```

### 17.6 Fusion-RAG 评估现状

| 维度 | 状态 |
|------|------|
| 单元测试 | 101 个测试, 84% 覆盖率 |
| RAG 评估 | 缺失 — 无端到端 RAG 质量评估 |
| 检索评估 | 缺失 — 无 Recall@K 基准 |
| 回答评估 | 缺失 — 无 Groundedness 评分 |
| 引用评估 | 缺失 — 无引用准确性检查 |

---

## 18. 生产部署模式

### 18.1 Managed Agents 架构

Anthropic 的生产级 Agent 架构 (2026.04):

**三大核心抽象**:

| 抽象 | 职责 | 关键特性 |
|------|------|----------|
| Session | 事件日志(只追加) | 跨 Harness 故障持久化 |
| Harness | 调用 Claude + 路由工具 | 可替换 |
| Sandbox | 执行环境 | 可丢弃, 隔离 |

**恢复模式**: Harness 故障时, 用 `wake(sessionId)` 获取事件日志, 从最后事件恢复。

**安全模式**: 凭证不暴露给 Sandbox, 通过 Proxy 从安全 Vault 获取。

### 18.2 长期运行 Agent 模式

跨多个上下文窗口的 RAG 任务:

```
[Initializer Agent] → 设置环境(init.sh), 创建进度文件
[Coding Agent] → 增量推进, 结构化更新进度文件
[下一个上下文窗口] → 读取进度文件, 继续工作
```

**进度文件模式**: `claude-progress.txt` 作为上下文窗口之间的交接文档。

### 18.3 Agent Skills 模式

Skills 是有组织的指令/脚本/资源文件夹, Agent 按需发现和加载:

```
skill/
├── SKILL.md        # YAML frontmatter (name, description)
├── scripts/        # 可执行脚本
└── references/     # 参考文档
```

渐进式披露: 元数据 → 完整 SKILL.md → 关联参考文件。

### 18.4 生产最佳实践

1. **启动简单** — 知识库 < 200K tokens, 全量注入 + Prompt Caching, 不用 RAG
2. **始终运行评估** — "评估使问题和行为变化在影响用户前可见"
3. **使用 Prompt Caching** — 对成本优化至关重要, 尤其是上下文化预处理
4. **上下文卫生** — "好的上下文工程 = 找到最小的高信号 token 集"
5. **Poka-yoke 工具** — 设计工具使错误难以发生(如: 要求绝对路径)
6. **投资 ACI** — "像投资 HCI 一样投资 Agent-Computer Interface"
7. **三大 Agent 设计原则**: (a) 保持简单 (b) 优先透明 (c) 精心设计工具文档
8. **Token 预算意识** — 多 Agent 系统消耗 ~15x chat token, 确保任务价值匹配成本

### 18.5 Fusion-RAG 生产化清单

| 项目 | 状态 | 说明 |
|------|------|------|
| 健康检查 | 已有 | EmbeddingClient.health() |
| 进程管理 | 已有 | start.sh start/stop/restart |
| 日志 | 已有 | logging 模块 |
| 缓存 | 已有 | ResultCache (SQLite) |
| 认证/授权 | 缺失 | 无 API Key 验证 |
| 速率限制 | 缺失 | 无请求限流 |
| 指标监控 | 缺失 | 无 Prometheus/指标导出 |
| 持久化会话 | 缺失 | MultiTurnRAG 历史在内存中 |
| 备份/恢复 | 缺失 | 无数据备份机制 |
| 滚动更新 | 缺失 | 无零停机更新策略 |

---

## 19. RAG 方案对比: Naive → Advanced → Agentic

### 19.1 四代 RAG 架构对比

| 维度 | Naive RAG | Advanced RAG | Modular RAG | Agentic RAG |
|------|-----------|-------------|-------------|-------------|
| 检索 | 单次检索 | 混合 + Rerank | 可配置 | 动态, 多步 |
| 上下文 | Chunk 丢失 | 上下文化 | 模块依赖 | Just-In-Time |
| 反馈 | 无 | 无 | 有限 | 完整闭环 |
| 适应性 | 无 | 静态 | 半静态 | 实时 |
| 成本 | 低 | 中 | 中 | 高(4-15x chat) |
| 准确性 | 基线 | +67% recall | 变化 | +90% (研究任务) |
| 代表 | LangChain RAG | Anthropic Contextual | 可插拔模块 | Multi-Agent |

### 19.2 Naive RAG 的问题

- 单次检索: embed 查询 → top-K chunk → 生成响应
- 无上下文保持: 分块破坏语义连续性
- 无反馈循环: 不检查检索质量
- Anthropic 评价: "传统 RAG 在编码信息时丢失上下文"

### 19.3 Advanced RAG (Anthropic Contextual Retrieval)

- **预检索**: 用 Claude 对 chunk 做上下文注释, 优化分块边界
- **检索**: 混合检索(Contextual Embedding + Contextual BM25)
- **后检索**: Reranking(top-150 → top-20)
- **关键洞察**: 索引前的上下文丰富 > 查询时补救

### 19.4 Modular RAG

- 独立模块: 查询变换、检索、Reranking、生成、评估
- Anthropic Agent Skills 模式: 可组合、可发现的能力模块
- MCP 提供模块间的集成标准

### 19.5 Agentic RAG (Anthropic 推荐)

- Claude 自主决定何时/如何检索
- 多步搜索, 动态适应
- 工具使用实现迭代检索
- 多 Agent 架构并行探索
- 渐进式披露, Just-In-Time 上下文加载
- 本质: "搜索的本质是压缩 — 子 Agent 通过独立上下文窗口并行实现压缩"

### 19.6 选型决策树

```
知识库大小?
├── < 200K tokens → 全量注入 + Prompt Caching (不用 RAG)
└── > 200K tokens → 需要检索
    ├── 查询简单, 精度要求不高 → Naive RAG
    ├── 查询复杂, 需要高精度 → Advanced RAG (Contextual + Reranking)
    └── 需要动态适应, 多步推理 → Agentic RAG
        ├── 单一知识库 → 单 Agent + 工具
        └── 多知识库/复杂研究 → Multi-Agent
```

---

## 20. Fusion-RAG 实现差距分析

### 20.1 P0: 关键缺失 (直接影响检索质量)

| 缺失能力 | 影响 | 当前状态 | 建议方案 |
|----------|------|----------|----------|
| **Contextual Retrieval** | 检索失败率高 49% | 缺失 | 实现 chunk 上下文化, 通过 fusion-mlx 生成 |
| **真正的 BM25** | 关键词检索不准 | 简化 TF | 引入 rank_bm25 或 whoosh |
| **专用 Reranker** | 延迟高、不稳定 | LLM 评分 | 添加 BGE-Reranker 本地模型 |

### 20.2 P1: 重要改进 (影响用户体验)

| 缺失能力 | 影响 | 当前状态 | 建议方案 |
|----------|------|----------|----------|
| **查询改写** | 多轮对话指代消解失败 | 缺失 | 用 LLM 做查询扩展/改写 |
| **检索质量评估** | 无法判断检索结果是否足够 | 缺失 | 添加 relevance threshold |
| **Embedding 缓存** | 重复文本重复计算 | 缺失 | SQLite/embedding cache |
| **自适应 chunk 大小** | 固定大小不适合所有文档 | 固定策略 | 根据文档类型自动调整 |
| **文档级 Prompt Caching** | 上下文化成本高 | 缺失 | 文档缓存 + chunk 迭代 |

### 20.3 P2: 锦上添花 (提升竞争力)

| 缺失能力 | 影响 | 当前状态 | 建议方案 |
|----------|------|----------|----------|
| **Agentic RAG 路由** | 单一检索策略 | 缺失 | 查询分类 → 多策略路由 |
| **RRF 融合** | Alpha 加权不够鲁棒 | Alpha only | 增加 RRF 选项 |
| **知识图谱** | 无法处理实体关系 | 缺失 | 轻量级实体-关系提取 |
| **多模态检索** | 仅支持文本 | 缺失 | 图像/表格 embedding |
| **分布式索引** | 单机限制 | 缺失 | LanceDB 分布式模式 |

### 20.4 当前代码具体问题

#### keyword_search 性能问题

```python
# 当前: 零向量搜索 + 全表 Python 过滤
zero_vec = [0.0] * self.dimension
all_rows = self.table.search(zero_vec).limit(10000).to_list()
# 问题: 10000 行全部加载到 Python, 然后做子串计数
```

建议: 使用 LanceDB 的 `where` 子句或全文搜索 API。

#### Reranker 顺序评分

```python
# 当前: 逐个文档评分, 顺序执行
for doc in documents:
    score = await self._score_relevance(client, query, text)
```

建议: 并发评分, 或使用 batch API。

#### 上下文缓存缺失

```python
# MultiTurnRAG: 无上下文长度管理
if len(self._history) > 20:
    self._history = self._history[-20:]
# 问题: 20 条历史可能超出 token 限制, 也可能不够
```

建议: 基于 token 计数的历史管理, 重要信息保留。

---

## 21. 最佳实践与推荐架构

### 21.1 Anthropic 的完整 RAG 推荐栈

按优先级排序:

1. **首先考虑不用 RAG** — 知识库 < 200K tokens, 直接全量注入 prompt + Prompt Caching
2. **上下文化分块** — 用 Claude 为每个 chunk 生成 50-100 token 上下文说明
3. **混合检索** — Contextual Embeddings + Contextual BM25, rank fusion 合并
4. **添加 Reranking** — 初始检索 top-150, Reranker 精排取 top-20
5. **使用 Prompt Caching** — 缓存系统提示、工具定义、常用上下文块(缓存读 90% 成本节省)
6. **复杂研究用多 Agent** — Lead Agent 协调 SubAgents 并行检索
7. **长期任务用压缩** — 实现对话摘要、结构化笔记、记忆工具
8. **工具密集型 RAG** — 用 Tool Search Tool(延迟加载) + Programmatic Tool Calling(最小化上下文消耗)

### 21.2 Anthropic 的六条黄金法则

1. **Embedding + BM25 > 单独 Embedding** — 永远使用混合检索
2. **Voyage/Gemini 是最佳 Embedding** — 离线场景用 BGE-M3
3. **Top-20 chunks > Top-10/5** — 更多上下文 = 更好结果
4. **Contextual Retrieval 大幅改善检索** — 必须实现
5. **Reranking > 无 Reranking** — 始终添加精排步骤
6. **所有收益可叠加** — 全部用上效果最佳

### 21.3 推荐的 Fusion-RAG v2 架构

```
+--------------------------------------------------+
|                   API Layer                       |
|          FastAPI + SSE Streaming                  |
+--------------------------------------------------+
|                RAG Pipeline                       |
|                                                   |
|  Query -> [Query Rewriter] -> [Router]           |
|              |              |                     |
|     [Vector Search]   [BM25 Search]              |
|              |              |                     |
|          [Hybrid Fusion]                          |
|              |                                    |
|          [Reranker]                               |
|              |                                    |
|     [Context Compression]                         |
|              |                                    |
|     [LLM Generation] -> Stream Response          |
+--------------------------------------------------+
|              Index Pipeline                       |
|                                                   |
|  Document -> [Preprocessor] -> [Chunker]         |
|      |                              |            |
|  [Context Generator]         [Recursive Chunker]  |
|      |                              |            |
|  [Contextual Embedding]    [BM25 Index]           |
|      |                              |            |
|  [Vector Store(LanceDB)]  [Metadata(SQLite)]      |
+--------------------------------------------------+
|           Infrastructure                          |
|                                                   |
|  fusion-mlx API: Embedding + Chat + Reranker     |
|  Apple Silicon: Metal GPU + Unified Memory        |
|  Storage: LanceDB + SQLite + Cache               |
+--------------------------------------------------+
```

### 21.4 Contextual Retrieval 实现路线图

#### Phase 1: 基础上下文化

```python
async def contextualize_chunk(chunk: str, document: str) -> str:
    prompt = f"""<document>
{document}
</document>
Here is the chunk we want to situate within the whole document
<chunk>
{chunk}
</chunk>
Please give a short succinct context to situate this chunk within the
overall document for the purposes of improving search retrieval of the chunk.
Answer only with the succinct context and nothing else."""

    response = await mlx_client.chat(prompt)
    return response + chunk  # 上下文 + 原始 chunk
```

#### Phase 2: 缓存优化

```python
class DocumentContextCache:
    async def get_or_generate(self, doc_id: str, chunk_index: int,
                               chunk: str, document: str) -> str:
        cache_key = f"{doc_id}:{chunk_index}"
        if cached := self.cache.get(cache_key):
            return cached
        context = await contextualize_chunk(chunk, document)
        self.cache.set(cache_key, context)
        return context
```

#### Phase 3: 并行处理

```python
async def contextualize_document(document: str, chunks: list[str]) -> list[str]:
    tasks = [
        contextualize_chunk(chunk, document)
        for chunk in chunks
    ]
    return await asyncio.gather(*tasks)
```

### 21.5 BM25 实现路线图

```python
from rank_bm25 import BM25Okapi

class BM25Index:
    def __init__(self):
        self.corpus = []
        self.bm25 = None

    def index(self, documents: list[str]):
        tokenized = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)
        self.corpus = documents

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [(i, scores[i]) for i in top_indices[:top_k]]
```

### 21.6 Reranker 实现路线图

```python
# 方案 1: 专用 Cross-Encoder (推荐)
class CrossEncoderReranker:
    def __init__(self, mlx_url: str):
        self.mlx_url = mlx_url

    async def rerank(self, query: str, documents: list[dict], top_k: int = 20):
        # 调用 fusion-mlx 的 rerank API
        # 或加载本地 cross-encoder 模型
        pass

# 方案 2: Batch LLM Reranker (短期)
class BatchLLMReranker:
    async def rerank(self, query: str, documents: list[dict], top_k: int = 20):
        docs_text = "\n".join(f"[{i}] {doc['text'][:200]}" for i, doc in enumerate(documents))
        prompt = f"""Rate each document's relevance to the query (0-10).
Query: {query}
{docs_text}
Output JSON: {{"scores": [score_0, score_1, ...]}}"""
        # 单次调用, 批量评分
```

### 21.7 性能基准目标

| 指标 | 当前 | Phase 1 目标 | 最终目标 |
|------|------|-------------|----------|
| 检索失败率 | ~8% (估算) | ~5% | ~2% |
| 端到端延迟 | ~3s | ~2s | ~1.5s |
| 首Token延迟 | ~1s | ~800ms | ~500ms |
| Reranking 延迟 | ~5s (LLM) | ~500ms | ~200ms |
| Embedding 吞吐 | ~10 docs/s | ~50 docs/s | ~100 docs/s |
| 缓存命中率 | 0% | 30% | 60% |

---

## 22. Anthropic RAG 完整文章索引

### 22.1 官方工程/研究文章

| 标题 | URL | 日期 | 核心贡献 |
|------|-----|------|----------|
| Introducing Contextual Retrieval | anthropic.com/engineering/contextual-retrieval | 2024.09 | Contextual Embeddings + Contextual BM25, 检索失败率降低 49%/67% |
| Building Effective Agents | anthropic.com/research/building-effective-agents | 2024.12 | 五种 Agent 工作流模式, 简单可组合 > 复杂框架 |
| Raising the Bar on SWE-bench Verified | anthropic.com/engineering/swe-bench-sonnet | 2025.01 | SWE-bench 基准, Agent 能力评估 |
| The "Think" Tool | anthropic.com/engineering/claude-think-tool | 2025.03 | 思考工具: Tau-Bench 航空领域 54% 改善 |
| How We Built Our Multi-Agent Research System | anthropic.com/engineering/multi-agent-research-system | 2025.06 | 多 Agent 研究架构, 比单 Agent 提升 90.2% |
| Writing Effective Tools for Agents | anthropic.com/engineering/writing-tools-for-agents | 2025.09 | 工具设计五原则, ACI 优化 |
| Effective Context Engineering for AI Agents | anthropic.com/engineering/effective-context-engineering-for-ai-agents | 2025.09 | 上下文腐烂、Just-In-Time 检索、压缩策略 |
| Equipping Agents for the Real World | anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills | 2025.10 | Agent Skills 框架 |
| Code Execution with MCP | anthropic.com/engineering/code-execution-with-mcp | 2025.11 | MCP 作为代码 API, 渐进式披露 |
| Introducing Advanced Tool Use | anthropic.com/engineering/advanced-tool-use | 2025.11 | Tool Search Tool(85% token 减少), Programmatic Tool Calling(98.7% token 减少) |

### 22.2 Cookbooks

| 标题 | 说明 |
|------|------|
| Prompt Caching Cookbook | Prompt Caching 实现指南 |
| Contextual Retrieval Cookbook | Contextual Retrieval 完整实现 |

### 22.3 性能基准汇总

| 基准 | 指标 | 最佳结果 | 来源 |
|------|------|----------|------|
| Contextual Retrieval | 检索失败率 1-Recall@20 | 1.9% (Reranked Contextual) | Contextual Retrieval |
| BrowseComp | 多 Agent vs 单 Agent | +90.2% | Multi-Agent Research |
| Tau-Bench (Airline) | Think Tool pass^1 | 0.570 (vs 0.370 baseline) | Think Tool |
| SWE-bench Verified | 解决率 | Claude Haiku 4.5: 73.3% | SWE-bench |
| OSWorld | 计算机使用 | Sonnet 4.5: 61.4% | — |
| Tool Accuracy | Opus 4 + Tool Search | 74% (vs 49%) | Advanced Tool Use |
| Tool Accuracy | Opus 4.5 + Tool Search | 88.1% (vs 79.5%) | Advanced Tool Use |

### 22.4 Claude 模型规格 (2026.07)

| 模型 | 上下文窗口 | 输入价格 | 输出价格 |
|------|-----------|----------|----------|
| Claude Fable 5 | 1M tokens | $10/MTok | $50/MTok |
| Claude Mythos 5 | 1M tokens | $10/MTok | $50/MTok |
| Claude Opus 5 | 1M tokens | $5/MTok | $25/MTok |
| Claude Sonnet 5 | 1M tokens | $3/MTok | $15/MTok |
| Claude Haiku 4.5 | 200K tokens | $1/MTok | $5/MTok |

### 22.5 Prompt Caching 规格

| 参数 | 值 |
|------|-----|
| 最小可缓存长度 | 1,024 tokens (Sonnet); 4,096 tokens (Opus/Haiku 4.5) |
| 缓存 TTL | 5 分钟(每次命中刷新); 1 小时 TTL 可选(2x 基础输入价) |
| 缓存写入价格 | 1.25x 基础输入价 |
| 缓存读取价格 | 0.1x 基础输入价(90% 折扣) |
| 每请求断点限制 | 最多 4 个显式断点 |
| 推荐模式 | 自动缓存: `cache_control={"type": "ephemeral"}` |

---

## 23. 参考文献

1. **Contextual Retrieval** — Anthropic Engineering Blog, 2024.09
   - https://www.anthropic.com/engineering/contextual-retrieval
   - 核心贡献: Contextual Embeddings + Contextual BM25, 检索失败率降低 49%, 加 Reranking 降低 67%

2. **Building Effective Agents** — Anthropic Research, 2024.12
   - https://www.anthropic.com/research/building-effective-agents
   - 核心贡献: 五种 Agent 工作流模式, 简单可组合 > 复杂框架

3. **Multi-Agent Research System** — Anthropic Engineering Blog, 2025.06
   - https://www.anthropic.com/engineering/multi-agent-research-system
   - 核心贡献: 多 Agent 研究架构, 比单 Agent 提升 90.2%

4. **Effective Context Engineering** — Anthropic Engineering Blog, 2025.09
   - https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
   - 核心贡献: 上下文腐烂、Just-In-Time 检索、压缩策略

5. **Advanced Tool Use** — Anthropic Engineering Blog, 2025.11
   - https://www.anthropic.com/engineering/advanced-tool-use
   - 核心贡献: Tool Search Tool(85% token 减少), Programmatic Tool Calling(98.7% token 减少)

6. **Writing Tools for Agents** — Anthropic Engineering Blog, 2025.09
   - https://www.anthropic.com/engineering/writing-tools-for-agents
   - 核心贡献: 工具设计五原则, ACI 优化

7. **Code Execution with MCP** — Anthropic Engineering Blog, 2025.11
   - https://www.anthropic.com/engineering/code-execution-with-mcp
   - 核心贡献: MCP 作为代码 API, 渐进式披露

8. **The "Think" Tool** — Anthropic Engineering Blog, 2025.03
   - https://www.anthropic.com/engineering/claude-think-tool
   - 核心贡献: 思考工具, Tau-Bench 航空领域 54% 改善

9. **Prompt Caching** — Anthropic
   - 延迟降低 >2x, 成本降低最高 90%
   - Contextual Retrieval 成本降至 $1.02/百万文档 tokens

10. **BM25 (Best Matching 25)** — Robertson & Zaragoza, 2009
    - TF-IDF 的改进, 添加饱和函数和文档长度归一化

11. **MMR (Maximal Marginal Relevance)** — Carbonell & Goldstein, 1998
    - 平衡相关性和多样性: lambda*rel - (1-lambda)*div

12. **Reciprocal Rank Fusion** — Cormack et al., 2009
    - 基于排名的融合: score = sum(1/(k + rank_i))

13. **BGE-M3** — BAAI, 2024
    - Multi-lingual, Multi-function, Multi-granularity embedding

14. **LanceDB** — LanceDB Inc.
    - Apple Silicon 优化的向量数据库, 零配置嵌入式

---

## 附录 A: Contextual Retrieval Prompt 模板

### 英文版 (Anthropic 原始)

```
<document>
{{WHOLE_DOCUMENT}}
</document>
Here is the chunk we want to situate within the whole document
<chunk>
{{CHUNK_CONTENT}}
</chunk>
Please give a short succinct context to situate this chunk within the
overall document for the purposes of improving search retrieval of the chunk.
Answer only with the succinct context and nothing else.
```

### 中文版 (适配)

```
<document>
{{WHOLE_DOCUMENT}}
</document>
以下是需要放在整个文档上下文中理解的文本片段
<chunk>
{{CHUNK_CONTENT}}
</chunk>
请给出简短的上下文说明,将此片段置于整个文档的语境中,以提高搜索检索的准确性。
仅输出上下文说明,不要输出其他内容。
```

### 代码文档版

```
<document>
{{WHOLE_DOCUMENT}}
</document>
Here is the code chunk we want to situate within the whole codebase
<chunk>
{{CHUNK_CONTENT}}
</chunk>
Please give a short succinct context to situate this code chunk within the
overall file or module. Include: what function/class it belongs to,
what it does, and its dependencies. Answer only with the succinct context.
```

---

## 附录 B: Fusion-RAG 检索效果估算

基于 Anthropic 基准数据推算 Fusion-RAG 在不同配置下的检索失败率:

| 配置 | 估算失败率 | 说明 |
|------|-----------|------|
| 当前: Embedding + 简化 TF | ~8-10% | 无真正 BM25, 无上下文化 |
| + 真正 BM25 | ~6-7% | 混合检索改善 |
| + Contextual Retrieval | ~4-5% | 上下文化大幅改善 |
| + 专用 Reranker | ~2-3% | 接近 Anthropic 最佳水平 |
| + 查询改写 + 自适应 | ~1.5-2% | 接近 SOTA |

---

## 附录 C: 关键术语表

| 术语 | 全称 | 说明 |
|------|------|------|
| RAG | Retrieval-Augmented Generation | 检索增强生成 |
| BM25 | Best Matching 25 | 经典关键词排序算法 |
| MMR | Maximal Marginal Relevance | 最大边际相关性, 平衡相关性与多样性 |
| RRF | Reciprocal Rank Fusion | 倒数排名融合 |
| SSE | Server-Sent Events | 服务器推送事件, 流式输出协议 |
| IDF | Inverse Document Frequency | 逆文档频率 |
| TF | Term Frequency | 词频 |
| HyDE | Hypothetical Document Embedding | 假设文档嵌入 |
| ACI | Agent-Computer Interface | 智能体-计算机接口 |
| Cross-Encoder | — | 交叉编码器, 用于 Reranking 的模型架构 |
