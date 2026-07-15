"""Fusion-KB FastAPI server — wires together all routes and services."""
from __future__ import annotations

import logging
from pathlib import Path

import uvicorn
from fastapi import FastAPI

from .routes import router as kb_router
from .routes import set_kb_context
from ..engine.knowledge_base import KnowledgeBaseManager
from ..embed.client import EmbeddingClient

logger = logging.getLogger(__name__)


def create_app(
    kb_storage_dir: str = "",
    mlx_base_url: str = "http://localhost:11434/v1",
    embedding_model: str = "BGE-M3",
) -> FastAPI:
    """Create and configure the Fusion-KB FastAPI application."""
    app = FastAPI(
        title="Fusion-KB",
        description="Apple Silicon native offline vector knowledge base backend",
        version="0.1.0",
    )

    # Initialize services
    kb_manager = KnowledgeBaseManager(storage_dir=kb_storage_dir)
    embed_client = EmbeddingClient(
        base_url=mlx_base_url,
        model=embedding_model,
    )

    # Inject context
    set_kb_context(kb_manager, embed_client)

    # Register routes
    app.include_router(kb_router)

    # Health check
    @app.get("/health")
    async def health():
        return {"status": "ok", "service": "fusion-kb"}

    return app


def run_server(
    host: str = "127.0.0.1",
    port: int = 11436,
    kb_storage_dir: str = "",
    mlx_base_url: str = "http://localhost:11434/v1",
    embedding_model: str = "BGE-M3",
    log_level: str = "INFO",
) -> None:
    """Run the Fusion-KB server."""
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO))
    app = create_app(
        kb_storage_dir=kb_storage_dir,
        mlx_base_url=mlx_base_url,
        embedding_model=embedding_model,
    )
    uvicorn.run(app, host=host, port=port, log_level=log_level.lower())