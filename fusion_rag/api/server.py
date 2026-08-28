"""Fusion-RAG FastAPI server — wires together all routes and services."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from ..embed.client import EmbeddingClient
from ..engine.knowledge_base import KnowledgeBaseManager
from .app_state import bind_app_state, init_app_state
from .mcp_server import router as mcp_router
from .routes import router as kb_router
from .routes_auth import router as auth_router

logger = logging.getLogger(__name__)


def create_app(
    kb_storage_dir: str = "",
    mlx_base_url: str = "http://127.0.0.1:11432/v1",
    embedding_model: str = "BGE-M3",
    mlx_api_key: str = "",
    fallback_url: str = "",
    fallback_api_key: str = "",
) -> FastAPI:
    kb_manager = KnowledgeBaseManager(storage_dir=kb_storage_dir)
    embed_client = EmbeddingClient(
        base_url=mlx_base_url,
        model=embedding_model,
        api_key=mlx_api_key,
        fallback_url=fallback_url,
        fallback_api_key=fallback_api_key,
    )

    # A1/A6: own the resource lifecycle. Without a lifespan the pooled
    # httpx clients (fusion_core.http_client) and EmbeddingClient's own
    # clients are never aclose'd on shutdown/reload — FDs leak across reloads
    # and the LRU pool's eviction tasks never run if the loop is exiting.
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 硬伤1: populate per-app state on app.state (shared services + the
        # per-app mutable dicts tasks/watches/kb_locks/project_kb_map). The
        # contextvar-binding middleware is registered on the app below (must
        # happen before the middleware stack is built, not inside lifespan).
        init_app_state(app, kb_manager, embed_client)
        logger.info("lifespan: startup complete, kb_manager=%d bases", kb_manager.count)
        try:
            yield
        finally:
            logger.info("lifespan: shutdown — closing clients and pooled connections")
            # P1-9: cancel every watch task so a reload doesn't orphan background
            # loops holding KB handles. Previously the tasks dict was dropped on
            # shutdown while the asyncio.create_task loops kept running.
            try:
                from .app_state import get_watches

                for watch in get_watches().values():
                    watch["active"] = False
                    task = watch.get("_task")
                    if task is not None and not task.done():
                        task.cancel()
                get_watches().clear()
            except Exception as e:
                logger.warning("lifespan: watch cancel failed: %s", e)
            try:
                await embed_client.close()
            except Exception as e:
                logger.warning("lifespan: embed_client close failed: %s", e)
            # 硬伤A/P0-3: close every pooled VectorStore backend handle (LanceDB
            # table refs / fusion_store lmdb envs) so no FD/env leaks across
            # reload.
            try:
                from .app_state import close_vec_store_pool

                await close_vec_store_pool()
            except Exception as e:
                logger.warning("lifespan: vec_store pool close failed: %s", e)
            # P2-8: close every pooled admin manager's sqlite conn (VersionManager
            # / SearchTemplateManager / PermissionManager / AuditLogger / BenchRunner)
            # so no FD/lock leaks across reload.
            try:
                from .app_state import close_admin_pool

                await close_admin_pool()
            except Exception as e:
                logger.warning("lifespan: admin pool close failed: %s", e)
            try:
                from fusion_core.http_client import close_all

                await close_all()
            except Exception as e:
                logger.warning("lifespan: fusion_core close_all failed: %s", e)

    app = FastAPI(
        title="Fusion-RAG",
        description="Apple Silicon native offline vector knowledge base backend",
        version="0.6.0",
        lifespan=lifespan,
    )

    # 硬伤1: bind each request's app.state to a contextvar so the no-arg
    # accessors (get_kb_manager / get_embed_client / ...) resolve the current
    # request's app without a Request parameter. Registered on the app object,
    # not in the lifespan — middleware cannot be added after the stack starts.
    app.middleware("http")(bind_app_state)

    app.include_router(kb_router)
    app.include_router(mcp_router)
    app.include_router(auth_router)

    @app.get("/health")
    async def health():
        return {"status": "ok", "service": "fusion-rag"}

    return app


def run_server(
    host: str = "127.0.0.1",
    port: int = 11436,
    kb_storage_dir: str = "",
    mlx_base_url: str = "http://127.0.0.1:11432/v1",
    embedding_model: str = "BGE-M3",
    mlx_api_key: str = "",
    log_level: str = "INFO",
) -> None:
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO))
    fallback_url = os.environ.get("FUSION_RAG_FALLBACK_URL", "")
    fallback_api_key = os.environ.get("FUSION_RAG_FALLBACK_API_KEY", "")
    app = create_app(
        kb_storage_dir=kb_storage_dir,
        mlx_base_url=mlx_base_url,
        embedding_model=embedding_model,
        mlx_api_key=mlx_api_key,
        fallback_url=fallback_url,
        fallback_api_key=fallback_api_key,
    )
    uvicorn.run(app, host=host, port=port, log_level=log_level.lower())


if __name__ == "__main__":
    host = os.environ.get("FUSION_RAG_HOST", "127.0.0.1")
    port = int(os.environ.get("FUSION_RAG_PORT", "11436"))
    mlx_url = os.environ.get("FUSION_MLX_URL", "http://127.0.0.1:11432/v1")
    embed_model = os.environ.get("FUSION_RAG_EMBED", "BGE-M3")
    mlx_api_key = os.environ.get("FUSION_MLX_API_KEY", "")
    if not mlx_api_key:
        try:
            import json as _json
            _settings_path = os.path.expanduser("~/.fusion-mlx/settings.json")
            if os.path.exists(_settings_path):
                with open(_settings_path) as _f:
                    _settings = _json.load(_f)
                mlx_api_key = _settings.get("auth", {}).get("api_key", "")
                if mlx_api_key:
                    logger.info("Auto-detected MLX api_key from ~/.fusion-mlx/settings.json")
        except Exception as _e:
            logger.warning("Failed to auto-detect MLX api_key: %s", _e)
    log_level = os.environ.get("FUSION_RAG_LOG_LEVEL", "INFO")
    run_server(
        host=host,
        port=port,
        mlx_base_url=mlx_url,
        embedding_model=embed_model,
        mlx_api_key=mlx_api_key,
        log_level=log_level,
    )
