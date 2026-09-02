# Deployment Guide — Fusion-RAG

Production deployment constraints, operational runbook, and performance baseline for enterprise commercial release.

> Current release: **0.8.0rc4** (release candidate — not yet stable). PEP 440 version lives in `pyproject.toml`; git tag is hyphenated (`v0.8.0-rc.4`).

---

## 1. Hard Constraints (read first)

These are architectural boundaries, not bugs. Violating them causes data corruption.

### 1.1 Single-process per stores dir (H3)

`FUSION_RAG_STORES_DIR` (default `~/.fusion-rag/stores`) holds per-KB LanceDB vector tables + BM25 sqlite + the in-process directory-watch registry. The watch loop and registry live in **process memory** — there is no cross-process coordination.

**DO NOT**:
- Run `uvicorn --workers N` (multi-worker) against one stores dir.
- Run two fusion-rag processes against the same `FUSION_RAG_STORES_DIR`.
- Docker `scale: N` or replicas sharing one volume.

Consequence of violation: double directory watches, registry corruption, concurrent LanceDB writes.

**Horizontal scaling is allowed** only behind a **stateless load balancer** with a **distinct stores volume per replica** (each replica owns an independent set of KBs). There is no shared-store multi-node mode.

### 1.2 Single embedding model (D7)

One `EmbeddingClient` is built from `FUSION_RAG_EMBED` at startup; all KBs share it. A KB created with a different `embedding_model` is **rejected at ingest (400)** rather than silently persisting cross-model vectors. To change the model: re-create the KB or restart the service with a new `FUSION_RAG_EMBED`.

### 1.3 External inference dependency

fusion-rag is **not self-contained**. All embedding + chat inference goes through fusion-mlx at `FUSION_MLX_URL` (default `http://127.0.0.1:11432/v1`). fusion-mlx must be running and reachable, or embedding/RAG endpoints fail. `/ready` returns 503 when the dependency is down.

---

## 2. Cold-Start Window (operational)

The embedding model **lazy-loads on first call**. After a cold fusion-mlx start, the **first** `/v1/embeddings` request returns 502 while fusion-mlx loads the weights, then succeeds. `EmbeddingClient` retries transparently, so this is a **one-time startup latency**, not a functional gap.

**Runbook**:
- After starting fusion-mlx, send one warm-up request (any search or ingest) before declaring fusion-rag ready.
- `/ready` probes fusion-mlx reachability but cannot predict the lazy-load delay. Budget ~5-15s for the first embedding call after a cold MLX start.
- Keep fusion-mlx warm (do not idle-stop it) if SLA demands sub-second first-response.

---

## 3. Deployment Artifacts

| Artifact | Use case |
|----------|----------|
| `Dockerfile` (repo root) + `deploy/docker-compose.yml` | Container deploy. fusion-mlx stays on host (Apple Silicon metal), reached via `host.docker.internal`. Stores on named volume. |
| `deploy/fusion-rag.service` | systemd bare-metal/VM. `TimeoutStopSec=40` drains in-flight requests before SIGKILL. |
| `start.sh` | Dev/single-user. nohup, logs to `logs/stdout.log` + `logs/fusion-rag.log` (rotated). |

### Docker

```bash
docker build -t fusion-rag:0.8.0rc4 .          # from repo root
docker compose -f deploy/docker-compose.yml up -d
```

Stores persist on the `fusion-rag-stores` named volume. Back up **only after** `POST /kb/bases/{kb_id}/checkpoint` (folds BM25 WAL + compacts LanceDB fragments for a consistent snapshot).

### systemd

```bash
sudo cp deploy/fusion-rag.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now fusion-rag
```

Adjust `WorkingDirectory` + `ExecStart` to your install path. fusion-mlx must run as its own unit before this one.

---

## 4. Health, Readiness, Metrics

| Endpoint | Purpose | Auth |
|----------|---------|------|
| `GET /health` | Liveness — process up | none |
| `GET /ready` | Readiness — deps reachable (503 when down) | none |
| `GET /metrics` | Prometheus text format (RED metrics) | none |
| `GET /kb/status` | KB count + embedding health | none |

Probes: use `/health` for liveness, `/ready` for readiness. `/metrics` for scrape.

---

## 5. Multi-Tenant Isolation (#62)

Opt-in via `FUSION_RAG_REQUIRE_GATEWAY=1`. When on:
- `/kb/*` requests must carry `X-Fusion-Route: gateway-decision` (gateway origin signal) or are rejected 403.
- KB list/get scoped to `X-Fusion-Tenant` header (authoritative, gateway-derived).
- `/health`, `/ready`, `/metrics`, `/mcp`, auth routes, and `/store/*` (M2M) are **exempt** from the gateway-origin gate.

Default **OFF** — single-tenant local-first dev is unaffected.

---

## 6. Security Checklist (pre-production)

- [ ] Set `FUSION_RAG_API_KEY` (empty = auth disabled, local-only).
- [ ] Set `FUSION_RAG_AUTH_BACKEND=apikey` (or `none` for trusted-network).
- [ ] Bind `FUSION_RAG_HOST=127.0.0.1` (default) unless behind a reverse proxy.
- [ ] Restrict `ReadWritePaths` / stores dir permissions.
- [ ] Enable `FUSION_RAG_REQUIRE_GATEWAY=1` for multi-tenant.
- [ ] Enable `FUSION_RAG_LOG_FORMAT=json` for aggregator ingestion.
- [ ] Set audit retention `FUSION_RAG_AUDIT_RETENTION_DAYS` (default 30, 0=forever).

---

## 7. Performance Baseline (0.8.0rc4)

Measured on Apple Silicon, monorepo venv (Python 3.14), `scripts/benchmark.py`. PRD targets all PASS (3/3).

| Metric | Target | Measured | Status |
|--------|--------|----------|--------|
| BM25 search @10K docs | <100 ms | 6.4 ms | PASS |
| BM25 index build @10K docs | — | 507 ms | — |
| Embedding cache hit rate | >90% | 90.9% | PASS |
| RRF fusion (avg) | <100 ms | 0.014 ms | PASS |

**Notes**:
- BM25 search is well within target (~15× headroom).
- Cache hit rate is at threshold — production with steady-state repeat queries will exceed it; cold-cache ingest will dip below.
- RRF fusion is negligible overhead.

### 7.1 Target-environment load test (0.8.0rc4)

End-to-end HTTP throughput + latency under concurrent load, measured against a live fusion-rag (0.8.0rc4) + fusion-mlx (BGE-M3 embeddings, 1024-dim) on Apple Silicon, monorepo venv (Python 3.14). 60-doc KB, embedding cache primed by a warmup phase. Tool: `scripts/load_test.py` (httpx async, no external deps). Report artifact: `scripts/load_test_report.json`.

| Phase | Concurrency | Duration | Requests | RPS | p50 | p90 | p99 | max | Errors |
|-------|-------------|----------|----------|-----|-----|-----|-----|-----|--------|
| Hybrid search (embed + BM25 + RRF) | 8 | 30s | 9047 | 301.6 | 23.6 ms | 37.5 ms | 69.0 ms | 155 ms | 0 |
| Keyword search (BM25 only) | 8 | 30s | 7250 | 241.7 | 27.8 ms | 49.1 ms | 104.9 ms | 486 ms | 0 |

**Reproduce**:
```bash
# 1. fusion-mlx serving BGE-M3 (e.g. port 11434, api key set)
# 2. start fusion-rag pointed at it:
FUSION_MLX_URL=http://127.0.0.1:11434/v1 FUSION_MLX_API_KEY=<key> \
FUSION_RAG_EMBED=BAAI/bge-m3 ./start.sh start
# 3. run the load test (seeds a KB, primes cache, fires concurrent search):
python scripts/load_test.py --docs 60 --concurrency 8 --duration 30
```

**Reading the numbers**:
- **0 errors across 16K+ requests** — no failures, no timeouts under sustained 8-way concurrency.
- **Hybrid p99 = 69 ms**, well inside a sub-200ms SLA. The 155 ms max is a single embedding-cache-miss tail (first occurrence of a query); steady-state is cache-hot.
- **Hybrid RPS (301) > keyword RPS (241)** looks inverted but is real: after warmup the query set's embeddings are cached, so hybrid is embed-cache-hit + BM25 + cheap RRF, while keyword pays jieba tokenization every call with no cache. The keyword p99 (105 ms) and 486 ms max reflect jieba first-load + GC jitter.
- **Not yet measured**: `/ask` (LLM generation, model-bound — add `--ask` for a chat-latency phase), memory steady-state under long runs, and higher concurrency (16/32). Run with higher `--concurrency` on your target hardware before committing to a p99 SLA above 8-way.

---

## 8. Operational Knobs (RuntimeConfig, env-driven)

| Variable | Default | Purpose |
|----------|---------|---------|
| `FUSION_RAG_SCAN_MAX_FILES` | 1000 | Max files per scan_directory |
| `FUSION_RAG_EMBED_CACHE_TTL` | 604800 | Embedding cache TTL (s), default 7d |
| `FUSION_RAG_EMBED_CACHE_MAX_ENTRIES` | 100000 | Max cache rows before LRU eviction |
| `FUSION_RAG_TOKEN_BUDGET` | 8192 | Multi-turn RAG token budget |
| `FUSION_RAG_MAX_HISTORY_TURNS` | 10 | Max RAG history turns kept |
| `FUSION_RAG_FETCH_K_MULTIPLIER` | 4 | Over-fetch factor for filtered search |
| `FUSION_RAG_WATCH_CAP` | 16 | Max concurrent directory watches per process |
| `FUSION_RAG_TRAJECTORY_MAX_MB` | 100 | Trajectory file MB before rotation |
| `FUSION_RAG_AUDIT_RETENTION_DAYS` | 30 | Audit log retention (0 = forever) |

Read-only view: `GET /kb/config`.

---

## 9. Commercial-Release Gaps (status)

| Gap | Status in 0.8.0rc4 |
|-----|--------------------|
| Single-process constraint documented | ✅ §1.1 |
| Cold-start window documented | ✅ §2 |
| Performance baseline (PRD engine metrics) | ✅ §7 |
| Env-specific throughput/p99 load test | ✅ §7.1 (0.8.0rc4, 8-way, 0 errors, hybrid p99 69ms) |
| Higher concurrency + /ask + memory steady-state | ⏳ Run `--concurrency 16/32 --ask` on target hardware |
| Version: rc → stable | ⏳ 0.8.0rc4 — still release candidate |

**To reach stable 0.8.0**: (1) confirm no rc-blocking findings in a pilot deploy, (2) optionally run a higher-concurrency load test, (3) bump `0.8.0rc4` → `0.8.0` + tag `v0.8.0`.
