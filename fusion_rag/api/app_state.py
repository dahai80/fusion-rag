"""硬伤1 per-app state on app.state, read via contextvar.

Before this, 6 route files held module-level globals (_kb_manager /
_embed_client / _tasks / _watches / _kb_locks / _project_kb_map) that a
`set_kb_context` setter mutated at startup. Problems: (a) under
`uvicorn --workers N` the per-process dicts (tasks/watches/project map)
diverge — worker A's watch is invisible to worker B; (b) on reload the
module globals persist while the app is rebuilt → stale references; (c)
the setter write had no lock. FastAPI DI exists precisely for this.

Fix: all per-app state lives on `app.state`, populated once by
`init_app_state` in the lifespan. A middleware binds `app.state` to a
contextvar each request, so the no-arg accessors below read the *current
request's* app state without threading `Request` through ~40 endpoint
signatures. Reload rebuilds app.state fresh (no stale module globals).

H3 — DEPLOYMENT CONSTRAINT (single-process only):
    The per-app dicts (tasks/watches/kb_locks) and the SqliteBase
    connection model (one in-process threading.RLock guarding one shared
    sqlite connection per store) are single-process safe ONLY.
    `uvicorn --workers N > 1` or a multi-node deployment sharing one
    `~/.fusion-rag/stores` dir is UNSUPPORTED and will corrupt data:
    each worker opens its own sqlite connection to the same metadata.db
    / bm25_index.db / versions.db / permissions.db, the RLock does NOT
    serialize across processes, and concurrent writes hit
    `database is locked` (5s busy_timeout) or interleave writes and
    corrupt the index. `_atomic_write_json` uses fcntl.flock for
    kb_meta.json only — the sqlite stores have no cross-process lock.
    Run fusion-rag as a SINGLE worker on a SINGLE node. Multi-node /
    multi-worker requires moving these stores to a real cross-process
    backend (PostgreSQL or an external sqlite daemon). Do NOT claim
    multi-node availability without that.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

from fastapi import HTTPException, Request

from ..embed.client import EmbeddingClient
from ..engine.knowledge_base import KnowledgeBaseManager

logger = logging.getLogger(__name__)

_current_app: contextvars.ContextVar[Any] = contextvars.ContextVar("fusion_rag_current_app")

# R3: per-KB watch concurrency cap. Without it a client loop-calling /watch
# spawned an unbounded number of _watch_loop background tasks (os.stat polling)
# → FD/CPU exhaustion. Env-overridable; default 16.
_DEFAULT_WATCH_CAP = 16


def _watch_cap() -> int:
    raw = os.environ.get("FUSION_RAG_WATCH_CAP", "").strip()
    if not raw:
        return _DEFAULT_WATCH_CAP
    try:
        val = int(raw)
        return val if val > 0 else _DEFAULT_WATCH_CAP
    except ValueError:
        logger.warning("invalid FUSION_RAG_WATCH_CAP=%r, defaulting to %d", raw, _DEFAULT_WATCH_CAP)
        return _DEFAULT_WATCH_CAP


def _watch_registry_path() -> Path:
    # R3: persisted under the stores root so a restart restores watches instead
    # of silently dropping them (operator thought monitoring was on; it wasn't).
    base = os.environ.get("FUSION_RAG_STORES_DIR", "")
    if not base:
        base = str(Path.home() / ".fusion-rag")
    return Path(base) / "watch_registry.json"


def _persist_watches(watches: dict[str, dict[str, Any]]) -> None:
    # R3: write the registry without the runtime-only fields (_task, hashes).
    # hashes rebuild on restore (re-hash the files); _task is recreated.
    serializable = {}
    for wid, w in watches.items():
        if not w.get("active"):
            continue
        serializable[wid] = {
            "watch_id": w["watch_id"],
            "kb_id": w["kb_id"],
            "file_paths": list(w["file_paths"]),
            "poll_interval": w.get("poll_interval", 30),
            "changes_detected": w.get("changes_detected", 0),
        }
    path = _watch_registry_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
        logger.debug("watch registry persisted: %d watches", len(serializable))
    except Exception as e:
        logger.warning("watch registry persist failed: %s", e)


def _load_watch_registry() -> list[dict[str, Any]]:
    path = _watch_registry_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return list(data.values()) if isinstance(data, dict) else []
    except Exception as e:
        logger.warning("watch registry load failed: %s", e)
        return []


def init_app_state(app: Any, kb_manager: KnowledgeBaseManager, embed_client: EmbeddingClient) -> None:
    """Populate app.state with shared services + per-app mutable dicts.

    Called once from the lifespan startup. Idempotent: re-running on reload
    rebuilds the mutable dicts fresh (no carry-over from the old app).
    """
    app.state.kb_manager = kb_manager
    app.state.embed_client = embed_client
    app.state.tasks: dict[str, dict[str, Any]] = {}
    app.state.watches: dict[str, dict[str, Any]] = {}
    app.state.kb_locks: dict[str, asyncio.Lock] = {}
    # A-P2-2: guard kb_locks creation. _kb_lock(kb_id) did check-then-act
    # (if kb_id not in locks: locks[kb_id] = Lock()) — two concurrent ingests
    # on a fresh KB both missed, each built a Lock, one was orphaned while the
    # other raced unguarded. A threading.Lock serializes the dict mutation; the
    # asyncio.Lock itself is created under it so exactly one wins.
    app.state.kb_locks_lock = threading.Lock()
    app.state.project_kb_map: dict[str, str] = {}
    # 硬伤A/P0-3/P2-1: per-app VectorStore singleton pool. One backend handle
    # per vector_path, reused across requests — kills the per-request
    # construction that (a) leaked LanceDB/FusionStore handles (never closed),
    # (b) reopened the fusion_store env each call → RuntimeError EnvAlreadyOpened
    # once a second handle hit the same lmdb dir, (c) reloaded the BM25 index
    # cold every request. Lock guards the dict; refcount lets shutdown close all.
    app.state.vec_store_pool: dict[str, Any] = {}
    app.state.vec_store_pool_lock = threading.Lock()
    # P2-8: per-KB admin manager pool. routes_admin built a fresh
    # VersionManager / SearchTemplateManager / PermissionManager / AuditLogger /
    # BenchRunner PER REQUEST — each opened a new sqlite connection (SqliteBase
    # lazy conn) and re-ran _init_db/_seed_builtins/_create_table. On a KB hit
    # repeatedly that is N redundant conn opens + N redundant CREATE TABLE runs.
    # One manager instance per (kb_storage_path, kind), reused; closed on
    # shutdown. Keyed by a string tag so the pool is one dict, lock-guarded.
    app.state.admin_pool: dict[str, Any] = {}
    app.state.admin_pool_lock = threading.Lock()
    # A-P1-1: per-KB MetadataStore pool. _get_meta_store(kb_id) built a fresh
    # MetadataStore(kb.metadata_path) per request — each opened a new sqlite
    # connection (check_same_thread=False, WAL) and never closed it. On a KB hit
    # repeatedly that is one leaked conn per request; across a long run the FDs
    # accumulate. One instance per metadata_path, reused; closed on shutdown.
    app.state.meta_pool: dict[str, Any] = {}
    app.state.meta_pool_lock = threading.Lock()
    # A-P2-1: TrajectoryWriter singleton. routes_search built a fresh
    # TrajectoryWriter() per search call — each re-read env + reopened the same
    # JSONL file, and _maybe_rotate ran unguarded (two rotations could race on
    # concurrent searches). One writer per process, lock-guarded rotation.
    app.state.trajectory_writer: Any = None
    app.state.trajectory_writer_lock = threading.Lock()
    # R3: restore persisted watches so a restart does not silently drop every
    # directory monitor (operator believes monitoring is live; without restore
    # it was gone with no signal). Rebuild hashes from the files on disk + spawn
    # a fresh _watch_loop per persisted watch. Files that no longer exist are
    # skipped (logged). The watch cap is honored: if the registry exceeds the
    # cap, restore up to the cap and log the rest dropped.
    _restore_watches(app.state)
    logger.info("app.state initialized: kb_manager=%d bases", kb_manager.count)


def _restore_watches(state: Any) -> None:
    registry = _load_watch_registry()
    if not registry:
        return
    cap = _watch_cap()
    restored = 0
    # late import: _watch_loop lives in routes_docs; avoid a cycle at import time
    try:
        from .routes_docs import _file_hash, _watch_loop
    except Exception as e:
        logger.warning("watch restore: cannot import _watch_loop: %s", e)
        return
    per_kb: dict[str, int] = {}
    for w in registry[:cap]:
        kb_id = w.get("kb_id", "")
        if not kb_id:
            continue
        if per_kb.get(kb_id, 0) >= cap:
            logger.warning("watch restore: KB %s at cap %d, dropping watch %s", kb_id, cap, w.get("watch_id"))
            continue
        file_paths = [fp for fp in w.get("file_paths", []) if os.path.isfile(fp)]
        if not file_paths:
            logger.info("watch restore: watch %s has no existing files, skipping", w.get("watch_id"))
            continue
        hashes = {}
        for fp in file_paths:
            h = _file_hash(fp)
            if h:
                hashes[fp] = h
        watch_id = w.get("watch_id") or f"restored-{restored}"
        watch = {
            "watch_id": watch_id,
            "kb_id": kb_id,
            "file_paths": file_paths,
            "poll_interval": w.get("poll_interval", 30),
            "hashes": hashes,
            "active": True,
            "changes_detected": w.get("changes_detected", 0),
        }
        state.watches[watch_id] = watch
        watch["_task"] = asyncio.create_task(_watch_loop(watch_id))
        per_kb[kb_id] = per_kb.get(kb_id, 0) + 1
        restored += 1
        logger.info("watch restore: restored watch %s (kb=%s, %d files)", watch_id, kb_id, len(file_paths))
    if restored:
        logger.info("watch restore: %d watches re-spawned on startup", restored)


async def bind_app_state(request: Request, call_next):
    """Middleware: bind the request's app.state to a contextvar for the call.

    Lets no-arg accessors below resolve the current app's state without a
    `Request` parameter on every endpoint.
    """
    token = _current_app.set(request.app)
    try:
        return await call_next(request)
    finally:
        _current_app.reset(token)


def _state() -> Any:
    app = _current_app.get(None)
    if app is None:
        raise HTTPException(503, "application state not bound to request context")
    return app.state


def get_kb_manager() -> KnowledgeBaseManager:
    mgr = getattr(_state(), "kb_manager", None)
    if mgr is None:
        raise HTTPException(503, "Knowledge base manager not initialized")
    return mgr


def get_embed_client() -> EmbeddingClient:
    client = getattr(_state(), "embed_client", None)
    if client is None:
        raise HTTPException(503, "Embedding client not initialized")
    return client


def get_tasks() -> dict[str, dict[str, Any]]:
    return _state().tasks


def get_watches() -> dict[str, dict[str, Any]]:
    return _state().watches


def get_kb_locks() -> dict[str, asyncio.Lock]:
    return _state().kb_locks


def get_project_kb_map() -> dict[str, str]:
    return _state().project_kb_map


def get_vec_store_pool() -> dict[str, Any]:
    return _state().vec_store_pool


def get_vec_store_pool_lock() -> threading.Lock:
    return _state().vec_store_pool_lock


def get_or_create_vec_store(vector_path: str, backend_type: str) -> Any:
    """硬伤A: return the cached VectorStore for vector_path, build on first use.

    One backend handle per KB vector dir, shared across every request that hits
    that KB. Reuses the LanceDB table / fusion_store env / in-process BM25Index
    instead of opening (and never closing) a fresh one per request. Thread-safe
    via the pool lock; fusion_store reopen (EnvAlreadyOpened) can no longer race.
    """
    from ..store.vector_store import VectorStore

    pool = get_vec_store_pool()
    lock = get_vec_store_pool_lock()
    cached = pool.get(vector_path)
    if cached is not None:
        return cached
    with lock:
        cached = pool.get(vector_path)
        if cached is not None:
            return cached
        vec_store = VectorStore(vector_path, backend_type=backend_type)
        pool[vector_path] = vec_store
        logger.info("vec_store pool: opened backend=%s path=%s", backend_type, vector_path)
        return vec_store


async def close_vec_store_pool() -> None:
    """Lifespan shutdown: close every pooled backend handle, best-effort.

    LocalBackend.close nulls handles; FusionStoreBackend.close checkpoints the
    lmdb env. Both are sync — run off-loop. Called from the lifespan finally.
    """
    from fastapi.concurrency import run_in_threadpool

    try:
        pool = get_vec_store_pool()
    except HTTPException:
        return
    lock = get_vec_store_pool_lock()
    with lock:
        items = list(pool.items())
        pool.clear()
    for vpath, vs in items:
        try:
            await run_in_threadpool(vs.close)
            logger.info("vec_store pool: closed %s", vpath)
        except Exception as e:
            logger.warning("vec_store pool: close failed for %s: %s", vpath, e)


def get_meta_store(metadata_path: str):
    """A-P1-1: return the cached MetadataStore for metadata_path, build once.

    _get_meta_store(kb_id) previously built a fresh MetadataStore per request —
    a new sqlite conn opened and never closed each call (FD leak across a long
    run). One instance per metadata_path, reused; thread-safe via the pool lock.
    Closed from the lifespan finally via close_meta_pool.
    """
    from ..store.metadata_store import MetadataStore

    pool = _state().meta_pool
    lock = _state().meta_pool_lock
    cached = pool.get(metadata_path)
    if cached is not None:
        return cached
    with lock:
        cached = pool.get(metadata_path)
        if cached is not None:
            return cached
        ms = MetadataStore(metadata_path)
        pool[metadata_path] = ms
        logger.info("meta pool: opened %s", metadata_path)
        return ms


async def close_meta_pool() -> None:
    """Lifespan shutdown: close every pooled MetadataStore's sqlite conn.

    MetadataStore.close nulls the shared conn. Sync close — run off-loop.
    """
    from fastapi.concurrency import run_in_threadpool

    try:
        pool = _state().meta_pool
    except HTTPException:
        return
    lock = _state().meta_pool_lock
    with lock:
        items = list(pool.items())
        pool.clear()
    for mpath, ms in items:
        try:
            await run_in_threadpool(ms.close)
            logger.info("meta pool: closed %s", mpath)
        except Exception as e:
            logger.warning("meta pool: close failed for %s: %s", mpath, e)


def get_trajectory_writer():
    """A-P2-1: return the process-singleton TrajectoryWriter, build once.

    routes_search built a fresh TrajectoryWriter() per search call — each
    re-read env + reopened the same JSONL file, and _maybe_rotate ran
    unguarded. One writer per process, reused; created under a lock so two
    concurrent searches don't race to build two. Returns None if disabled.
    """
    from ..engine.trajectory_writer import TrajectoryWriter

    state = _state()
    if state.trajectory_writer is not None:
        return state.trajectory_writer
    with state.trajectory_writer_lock:
        if state.trajectory_writer is not None:
            return state.trajectory_writer
        writer = TrajectoryWriter()
        state.trajectory_writer = writer
        logger.info("trajectory writer singleton initialized")
        return writer


async def close_trajectory_writer() -> None:
    """Lifespan shutdown: best-effort flush/close of the singleton writer."""
    from fastapi.concurrency import run_in_threadpool

    try:
        state = _state()
    except HTTPException:
        return
    writer = state.trajectory_writer
    if writer is None:
        return
    close = getattr(writer, "close", None)
    if close is None:
        return
    try:
        await run_in_threadpool(close)
        logger.info("trajectory writer singleton closed")
    except Exception as e:
        logger.warning("trajectory writer close failed: %s", e)


def _get_or_create_admin(key: str, factory):
    # P2-8: shared admin-manager pool. Each admin manager is a SqliteBase
    # (VersionManager / SearchTemplateManager / PermissionManager / AuditLogger /
    # BenchRunner) — one instance per (kb_storage_path, kind), reused across
    # requests. Lock guards the dict; factory builds on first miss.
    pool = _state().admin_pool
    lock = _state().admin_pool_lock
    cached = pool.get(key)
    if cached is not None:
        return cached
    with lock:
        cached = pool.get(key)
        if cached is not None:
            return cached
        instance = factory()
        pool[key] = instance
        logger.info("admin_pool: created %s", key)
        return instance


def get_version_manager(kb_storage_path: str):
    from ..engine.version_manager import VersionManager

    return _get_or_create_admin(
        f"vm:{kb_storage_path}/versions.db",
        lambda: VersionManager(f"{kb_storage_path}/versions.db"),
    )


def get_template_manager(kb_storage_path: str):
    from ..engine.search_template import SearchTemplateManager

    return _get_or_create_admin(
        f"tm:{kb_storage_path}/templates.db",
        lambda: SearchTemplateManager(f"{kb_storage_path}/templates.db"),
    )


def get_permission_manager(kb_storage_path: str):
    from ..permissions import PermissionManager

    return _get_or_create_admin(
        f"pm:{kb_storage_path}/permissions.db",
        lambda: PermissionManager(f"{kb_storage_path}/permissions.db"),
    )


def get_audit_logger(kb_storage_path: str):
    from ..engine.audit_logger import AuditLogger

    return _get_or_create_admin(
        f"al:{kb_storage_path}/audit.db",
        lambda: AuditLogger(f"{kb_storage_path}/audit.db"),
    )


def get_bench_runner(kb_storage_path: str):
    from ..engine.bench import BenchRunner

    return _get_or_create_admin(
        f"br:{kb_storage_path}/bench.db",
        lambda: BenchRunner(f"{kb_storage_path}/bench.db"),
    )


async def close_admin_pool() -> None:
    """Lifespan shutdown: close every pooled admin manager's sqlite conn.

    All admin managers inherit SqliteBase (or hold a sqlite conn) and expose
    _close_conn. Sync close — run off-loop. Called from the lifespan finally.
    """
    from fastapi.concurrency import run_in_threadpool

    try:
        pool = _state().admin_pool
    except HTTPException:
        return
    lock = _state().admin_pool_lock
    with lock:
        items = list(pool.items())
        pool.clear()
    for key, mgr in items:
        close = getattr(mgr, "_close_conn", None)
        if close is None:
            continue
        try:
            await run_in_threadpool(close)
            logger.info("admin_pool: closed %s", key)
        except Exception as e:
            logger.warning("admin_pool: close failed for %s: %s", key, e)
