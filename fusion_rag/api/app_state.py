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
