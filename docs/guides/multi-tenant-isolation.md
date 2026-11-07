---
doc_id: fusion-rag/guide-multi-tenant-isolation
product: fusion-rag
class: platform
entry: service
doc_type: guide
title: {zh: "指南：多租户隔离部署", en: "Guide: multi-tenant isolation deployment"}
audience: engineer
status: draft
version: 0.8.21
updated: 2026-11-07
owner: @dahai
source_of_truth:
  - README.md
  - fusion_rag/api/tenant.py
  - fusion_rag/api/identity.py
---

# 指南：多租户隔离部署 / Multi-tenant isolation deployment

fusion-rag 的租户边界是逐级增强的三档，全部默认关闭（单租户本地开发零配置）。/ Tenant isolation ships in three escalating tiers, all off by default (single-tenant local dev needs zero config).

## 第一档：服务级 API key / Tier 1: service-level API key

```bash
export FUSION_RAG_API_KEY=<key-value-supplied-by-operator>
./start.sh restart
```

所有 `/kb/*` 需 `X-API-Key` 头。只解决"谁都能调"，不隔离数据。

With a key set, every `/kb/*` call needs the `X-API-Key` header. This gates access only; it does not isolate data.

## 第二档：网关租户头 / Tier 2: gateway tenant header

```bash
export FUSION_RAG_REQUIRE_GATEWAY=1
./start.sh restart
```

效果（issue #61）：`/kb/*` 必须携带 `X-Fusion-Route: gateway-decision`（证明经 fusion-gateway），KB 建库时打 `X-Fusion-Tenant` 租户戳，list/get 按租户过滤。`/health`、`/ready`、`/metrics`、`/mcp` 保持开放以便观测。约束：该头可被直连 `:11436` 的调用方伪造——生产多租户不要停在这一档。

Requests must transit the gateway and carry the tenant header; KBs are stamped and scoped per tenant while observability routes stay open. Constraint: the header is forgeable by direct-port callers, so production multi-tenant should not stop at this tier.

## 第三档：identity JWT 权威解析 / Tier 3: authoritative identity JWT

```bash
export FUSION_RAG_REQUIRE_IDENTITY=1
export FUSION_IDENTITY_URL=http://127.0.0.1:11470
export FUSION_IDENTITY_SERVICE_TOKEN=<service-token-supplied-by-operator>
./start.sh restart
```

效果（issue #68）：每个 `/kb/*` 请求需 `Authorization: Bearer <JWT>`，fusion-rag 调 fusion-identity `POST /api/v1/auth/verify`（服务令牌门控）校验；租户取 JWT `tid` claim。`X-Fusion-Tenant` 降级为纵深防御（与 `tid` 不一致即 401）。identity 服务不可达 = fail-closed（401），不会回退到请求头。约束：fusion-identity 是 HTTP 依赖（:11470），不是 Python import；`FUSION_IDENTITY_SERVICE_TOKEN` 缺失即 fail-closed。

Every `/kb/*` request needs a JWT verified against fusion-identity; the tenant comes from the `tid` claim and the header is demoted to defense-in-depth. An unreachable identity service fails closed. Constraints: fusion-identity is an HTTP dependency on :11470, and a missing service token fails closed.

## 与存储后端的配合 / store-backend interplay

qdrant 后端（`FUSION_RAG_STORE_BACKEND=qdrant`）按 `{QDRANT_COLLECTION_PREFIX}{tenant_id}` 隔离集合；无租户请求落到共享 `fusion_rag_kb_{kb_id}` 集合。local 后端不做租户级目录隔离，隔离只发生在 API 层。

The qdrant backend isolates collections per tenant prefix; with no tenant, requests land in the shared collection. The local backend isolates only at the API layer.

## 扩容时的 key 策略 / key policy when scaling

两档多租户开关任一开启后，`POST /kb/auth/keys` 返回 409（issue #102）：单机 auth.db 不参与横向扩容。用共享 `FUSION_RAG_API_KEY` 或 identity 签发的凭据；不要试图把 auth.db 换成共享数据库（评估过、已拒绝的方案）。

Local key minting returns 409 once either tenancy flag is on; use the shared env key or identity-issued credentials. Backing auth.db with a shared database was evaluated and rejected.
