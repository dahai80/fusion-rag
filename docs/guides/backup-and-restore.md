---
doc_id: fusion-rag/guide-backup-restore
product: fusion-rag
class: platform
entry: service
doc_type: guide
title: {zh: "指南：快照安全的备份与恢复", en: "Guide: snapshot-safe backup and restore"}
audience: engineer
status: draft
version: 0.8.21
updated: 2026-11-07
owner: @dahai
source_of_truth:
  - README.md
  - deploy/docker-compose.yml
  - deploy/fusion-rag.service
---

# 指南：快照安全的备份与恢复 / Snapshot-safe backup and restore

## 为什么不能直接 tar / why a raw tar is unsafe

每 KB 的 store 目录（默认 `~/.fusion-rag/stores/{kb_id}/`，实际路径由 `FUSION_RAG_STORES_DIR` / `FUSION_RAG_DATA_DIR` 决定）混合了 SQLite WAL 库（metadata.db、bm25_index.db、versions.db 等）与 LanceDB `vectors/`。写入中途直接 tar/rsync 会拿到不一致的 `.db` 加过期 `-wal`——恢复时被重放或丢弃，数据静默损坏（O-P2-1）。

A per-KB stores dir mixes SQLite WAL databases with LanceDB files; a raw tar or rsync mid-write captures an inconsistent db plus a stale WAL that a restore would replay or drop, corrupting data silently.

## 备份步骤 / backup procedure

先 checkpoint，再打包：

```bash
curl -X POST -H "Authorization: Bearer $FUSION_RAG_API_KEY" \
  http://127.0.0.1:11436/kb/bases/{kb_id}/checkpoint

tar -czf kb-snapshot.tgz -C ~/.fusion-rag/stores {kb_id}
```

`POST /kb/bases/{kb_id}/checkpoint` 对该 KB 的每个 SQLite store 执行 `PRAGMA wal_checkpoint(TRUNCATE)`，并对 LanceDB 表执行 `table.optimize()`；完成后再打快照。多库就逐库执行（该接口是按 KB 的）。

The checkpoint endpoint runs a WAL truncate-checkpoint on every SQLite store for the KB and optimizes the LanceDB table; snapshot afterwards, per KB.

## 版本快照与备份的关系 / version snapshots vs backups

`POST /kb/bases/{kb_id}/versions` 创建的硬链接快照是"逻辑回滚点"（检索层可 `rollback`），不替代文件级备份——它仍在同一磁盘上。两者配合：版本快照用于误删/误改恢复，文件备份用于磁盘级故障。

Version snapshots are logical rollback points on the same disk; they complement but do not replace file-level backups.

## 恢复 / restore

1. `./start.sh stop`（SIGTERM，最多 35s 排水；systemd 单元 `TimeoutStopSec=40`）。
2. 解包覆盖 store 目录后 `./start.sh start`。
3. 验证：`GET /kb/bases/{kb_id}/stats` 的文档/chunk 计数与备份前一致；抽查一次 `POST .../search`。

Stop the service, restore the directory, restart, then verify counts via the stats endpoint and spot-check a search.

## 相关约束 / related constraints

- 单进程约束（H3）：备份期间不要另起实例写同一 stores 目录；checkpoint + tar 足够快，常规做法是低峰执行。
- `FUSION_RAG_AUDIT_RETENTION_DAYS`（默认 30）只影响审计库；备份不含已过保留期的审计行。
- 容器部署：数据目录按 `deploy/docker-compose.yml` 的挂载为准，备份对象是宿主机挂载点。

Single-writer discipline applies during backup (no second instance against the stores dir); audit retention bounds what a backup can contain; for container deploys back up the host mount from deploy/docker-compose.yml.
