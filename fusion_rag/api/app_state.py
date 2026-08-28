"""硬伤1 per-app state on app.state, read via contextvar (multi-worker safe).

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
signatures. Multi-worker: each worker's app has its own app.state (correct
— tasks/watches are inherently per-process), and reload rebuilds app.state
fresh (no stale module globals).
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import threading
from typing import Any

from fastapi import HTTPException, Request

from ..embed.client import EmbeddingClient
from ..engine.knowledge_base import KnowledgeBaseManager

logger = logging.getLogger(__name__)

_current_app: contextvars.ContextVar[Any] = contextvars.ContextVar("fusion_rag_current_app")


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
    logger.info("app.state initialized: kb_manager=%d bases", kb_manager.count)


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
