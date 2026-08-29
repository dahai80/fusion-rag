"""Fusion-RAG FastAPI server — wires together all routes and services."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_v

import uvicorn
from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse

from ..embed.client import EmbeddingClient
from ..engine.knowledge_base import KnowledgeBaseManager
from .app_state import bind_app_state, init_app_state
from .auth import verify_api_key
from .mcp_server import router as mcp_router
from .routes import router as kb_router
from .routes_auth import router as auth_router

logger = logging.getLogger(__name__)


def _pkg_version() -> str:
    # O-P1-4: prior `version="0.6.0"` was a hardcoded string that drifted from
    # pyproject.toml — by audit the app advertised 0.6.0 while the package was
    # 0.7.2. Read the installed package metadata (importlib.metadata is stdlib,
    # works whether installed editable or from a wheel) so /health/openapi never
    # lie about the running version. Fall back only if the package is not
    # importable (running from source without install).
    try:
        return _pkg_v("fusion-rag")
    except PackageNotFoundError:
        return "0.0.0+unknown"


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
            # A-P1-1: close every pooled MetadataStore conn (one sqlite conn per
            # KB metadata.db, opened lazily by get_meta_store). A-P2-1: flush the
            # TrajectoryWriter singleton. Both were constructed-per-request before
            # and never closed — FD leak across a long run / reload.
            try:
                from .app_state import close_meta_pool, close_trajectory_writer

                await close_meta_pool()
                await close_trajectory_writer()
            except Exception as e:
                logger.warning("lifespan: meta/trajectory close failed: %s", e)
            try:
                from fusion_core.http_client import close_all

                await close_all()
            except Exception as e:
                logger.warning("lifespan: fusion_core close_all failed: %s", e)

    app = FastAPI(
        title="Fusion-RAG",
        description="Apple Silicon native offline vector knowledge base backend",
        version=_pkg_version(),
        lifespan=lifespan,
    )

    # 硬伤1: bind each request's app.state to a contextvar so the no-arg
    # accessors (get_kb_manager / get_embed_client / ...) resolve the current
    # request's app without a Request parameter. Registered on the app object,
    # not in the lifespan — middleware cannot be added after the stack starts.
    app.middleware("http")(bind_app_state)

    # O-P2-2: request-id correlation. Registered after bind_app_state so it runs
    # innermost — the id is set BEFORE the route handler logs, and the metrics
    # middleware (registered next, outermost) can read the id from the
    # contextvar if it needs to tag a RED metric. Echoes X-Request-ID on the
    # response so a client/gateway can trace a call end-to-end.
    from .logging_setup import request_id_middleware

    app.middleware("http")(request_id_middleware)

    # R5: RED metrics — record request count / latency / errors per
    # endpoint+kb_id. Registered after bind_app_state (order matters: this runs
    # last in the stack, so route params + final status are available).
    from ..engine.metrics import get_metrics, metrics_middleware

    app.middleware("http")(metrics_middleware)

    app.include_router(kb_router)
    app.include_router(mcp_router)
    app.include_router(auth_router)

    @app.get("/health")
    async def health():
        # O-P1-1: /health is LIVENESS only — a cheap "the process is up and the
        # event loop turns" probe. It must NOT check downstream deps (embed
        # health, store writability) or a transient MLX hiccup evicts the pod
        # from a load balancer and cascades restarts. Deps live on /ready.
        return {"status": "ok", "service": "fusion-rag", "version": _pkg_version()}

    @app.get("/ready")
    async def ready():
        # O-P1-1: /ready is READINESS — is the service actually able to serve a
        # real request right now? Checks the embedding backend (fusion-mlx
        # reachable + model loadable) and the per-KB store root writability.
        # A scrape/k8s readinessProbe points here so a not-yet-ready instance
        # is pulled from rotation WITHOUT being killed (liveness stays /health).
        # Failures surface as 503 + a `ready: false` body with the failing check.
        from .app_state import get_kb_manager

        checks: dict[str, str] = {}
        ready = True
        try:
            embed_client.health()
            checks["embedding"] = "ok"
        except Exception as e:
            ready = False
            checks["embedding"] = f"unavailable: {e}"
        try:
            import os as _os
            import tempfile

            stores_dir = get_kb_manager().storage_dir
            _os.makedirs(stores_dir, exist_ok=True)
            with tempfile.TemporaryFile(dir=stores_dir):
                pass
            checks["store"] = "ok"
        except Exception as e:
            ready = False
            checks["store"] = f"not writable: {e}"
        # O-P2-3: surface whether the startup health gate passed. run_server
        # calls EmbeddingClient.health() before serving; if MLX was down at
        # boot the gate logs + the operator sees it here too.
        status_code = 200 if ready else 503
        return JSONResponse(
            {"ready": ready, "checks": checks, "service": "fusion-rag", "version": _pkg_version()},
            status_code=status_code,
        )

    @app.get("/metrics", dependencies=[Depends(verify_api_key)])
    async def metrics() -> str:
        # R5: Prometheus text exposition. S-P2-2: auth — /metrics exposes
        # aggregate counters (no per-user PII), but on an authenticated instance
        # an unauthenticated scrape leaks volume/latency signal. Depends(
        # verify_api_key) gates it the same way write endpoints are: NoAuth
        # backend (no admin key) -> verify returns None -> open (local-first
        # single-user box); ApiKey backend -> missing/wrong key -> 401. No
        # separate env flag — the auth backend itself is the switch.
        from starlette.responses import PlainTextResponse

        return PlainTextResponse(
            get_metrics().render_prometheus(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

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
    # O-P1-2 + O-P2-2: structured logging — RotatingFileHandler (10MB x 5) so a
    # long run no longer fills the disk, optional JSON formatter
    # (FUSION_RAG_LOG_FORMAT=json) for an aggregator, and a request-id contextvar
    # the middleware below tags every log line with. Replaces the prior
    # basicConfig (single unbounded StreamHandler).
    from .logging_setup import configure_logging

    configure_logging(log_level)
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
    # O-P1-5: graceful drain. Default uvicorn timeout_graceful_shutdown is ~5s
    # (or 0 = immediate). A fusion-rag request may hold an in-flight LLM call
    # (rerank/contextualize/RAG generation) whose retry deadline can run tens
    # of seconds. A SIGTERM during one cut it off mid-response. Give in-flight
    # requests 30s to drain before the loop hard-stops, matching the LLM retry
    # deadline so a deploy/restart finishes pending work instead of aborting.
    # O-P1-3: silence uvicorn's per-request access log — it logs the full query
    # string (PII: the search query) at INFO. Keep app logs (which we control
    # and have already downgraded PII fields to DEBUG); drop uvicorn.access so
    # the raw request line never hits the log sink.
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_level.lower(),
        timeout_graceful_shutdown=30,
        access_log=False,
    )


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
    # O-P2-3: startup health gate. CLAUDE.md states "fusion-mlx must be running"
    # but nothing enforced it — a fusion-rag started before fusion-mlx served
    # /health=ok while every embed call 401'd until MLX came up (false healthy).
    # Probe the embedding backend once at boot; if unreachable, log loudly and
    # exit non-zero so a supervisor/process manager does not route traffic to a
    # service that cannot answer a real request. FUSION_RAG_SKIP_STARTUP_PROBE=1
    # opts out (e.g. MLX is known to come up later on a constrained box).
    if os.environ.get("FUSION_RAG_SKIP_STARTUP_PROBE", "").strip() not in ("1", "true", "yes"):
        import sys

        probe = EmbeddingClient(base_url=mlx_url, model=embed_model, api_key=mlx_api_key)
        try:
            probe.health()
            logger.info("startup probe: fusion-mlx embedding backend reachable at %s", mlx_url)
        except Exception as e:
            logger.error(
                "startup probe FAILED: fusion-mlx not reachable at %s (%s). "
                "Start fusion-mlx first, or set FUSION_RAG_SKIP_STARTUP_PROBE=1 to bypass. Exiting.",
                mlx_url,
                e,
            )
            sys.exit(1)
    run_server(
        host=host,
        port=port,
        mlx_base_url=mlx_url,
        embedding_model=embed_model,
        mlx_api_key=mlx_api_key,
        log_level=log_level,
    )
