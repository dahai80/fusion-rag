# 架构合规整改计划

审计日期: 2026-08-02
关联 Issue: #30
违规等级: P1
合规评级: C

层级定位: 二、核心网关引擎 - 全局项目向量知识库
核心职责: 向量索引构建、文档嵌入、语义检索

违规项与整改:
1. MCP服务器面向终端 - api/mcp_server.py 直接服务Claude Desktop/Cursor - 迁至独立面向层或fusion-gateway - P1-S1
2. 项目-KB映射 - api/routes.py 937-974行 - 迁至编排层 - P1-S1
3. 硬编码RAG回答生成 - api/routes.py 802-934行含观点性系统提示 - /ask端点拆分: KB只返回chunks, 调用方生成回答; 系统提示改为可配置 - P1-S1
4. evaluator循环依赖 - engine/evaluator.py反向导入API层 - 修复 - P1-S2
5. API Key认证 - 属服务基础设施, 应可插拔 - P1-S2
6. 会话式多轮RAG - rag_chain.py管会话状态+硬编码提示 - 会话管理提取到调用方 - P1-S3

合规标准: fusion-kb应只包含向量索引/文档嵌入/语义检索/chunk管理, 不应面向终端用户/硬编码业务逻辑/管会话状态
