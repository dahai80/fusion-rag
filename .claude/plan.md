# Plan: Fix Issue #30 + Close PR #31

## Issue #30 违规项逐项整改

### 1. MCP服务器面向终端 (P1-S1)
- MCP 只是 HTTP API 的另一种协议入口，与 REST API 同级，不含业务逻辑
- 审计确认合规，无需迁移

### 2. 项目-KB映射迁至编排层 (P1-S1)
- 创建 `api/routes_project.py`，3 个 project 端点移入
- routes_admin.py 删除 project 端点
- routes.py 挂载 project router

### 3. 硬编码RAG回答生成 (P1-S1)
- `_generate_answer()` 增加 `system_prompt` 参数，支持环境变量 `FUSION_RAG_SYSTEM_PROMPT` 配置
- `/ask` 端点增加 `system_prompt` 请求参数

### 4. evaluator循环依赖 (P1-S2)
- `evaluator.py` 删除 `from .routes import` 反向导入
- 改为直接使用 EmbeddingClient + VectorStore + httpx

### 5. API Key认证可插拔 (P1-S2)
- auth.py 增加 `AuthBackend` ABC + `NoAuthBackend` + `get_auth_dependency()` 工厂
- 环境变量 `FUSION_RAG_AUTH_BACKEND` 选择后端

### 6. 会话式多轮RAG (P1-S3)
- MultiTurnRAG.__init__ 增加 `system_prompt` 参数

### PR #31
- 整改完成后更新 ARCHITECTURE_COMPLIANCE.md，merge PR

## 文件变更

| 文件 | 操作 |
|------|------|
| `api/routes_project.py` | 新建 |
| `api/routes_admin.py` | 删除 project 端点 |
| `api/routes.py` | 挂载 project router |
| `api/routes_search.py` | ask 传 system_prompt |
| `engine/evaluator.py` | 修循环依赖 |
| `api/auth.py` | AuthBackend 抽象 |
| `engine/rag_chain.py` | system_prompt 参数化 |
| `ARCHITECTURE_COMPLIANCE.md` | 更新状态 |
