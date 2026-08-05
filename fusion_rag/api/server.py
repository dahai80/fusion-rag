"""Fusion-RAG FastAPI server — wires together all routes and services."""

from __future__ import annotations

import logging
import os

import uvicorn
from fastapi import FastAPI

from ..embed.client import EmbeddingClient
from ..engine.knowledge_base import KnowledgeBaseManager
from .mcp_server import router as mcp_router
from .routes import router as kb_router
from .routes import set_kb_context
from .routes_auth import router as auth_router

logger = logging.getLogger(__name__)


def create_app(
    kb_storage_dir: str = "",
    mlx_base_url: str = "http://127.0.0.1:11434/v1",
    embedding_model: str = "BGE-M3",
    mlx_api_key: str = "",
    fallback_url: str = "",
    fallback_api_key: str = "",
) -> FastAPI:
    app = FastAPI(
        title="Fusion-RAG",
        description="Apple Silicon native offline vector knowledge base backend",
        version="0.6.0",
    )

    kb_manager = KnowledgeBaseManager(storage_dir=kb_storage_dir)
    embed_client = EmbeddingClient(
        base_url=mlx_base_url,
        model=embedding_model,
        api_key=mlx_api_key,
        fallback_url=fallback_url,
        fallback_api_key=fallback_api_key,
    )

    set_kb_context(kb_manager, embed_client)

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
    mlx_base_url: str = "http://127.0.0.1:11434/v1",
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
    mlx_url = os.environ.get("FUSION_MLX_URL", "http://127.0.0.1:11434/v1")
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
