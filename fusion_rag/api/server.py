"""Fusion-RAG FastAPI server — wires together all routes and services."""
from __future__ import annotations

import logging
from pathlib import Path

import uvicorn
from fastapi import FastAPI

# callers: create_app() mounts both routers; run_server() calls create_app
# API: app.include_router(mcp_router) adds /mcp/* endpoints
# schema: MCP JSON-RPC 2.0 protocol (see mcp_server.py)
# user instruction: "按照你的方案和计划落地所有phase阶段的需求"
from .routes import router as kb_router
from .routes import set_kb_context
from .mcp_server import router as mcp_router
from ..engine.knowledge_base import KnowledgeBaseManager
from ..embed.client import EmbeddingClient

logger = logging.getLogger(__name__)


def create_app(
    kb_storage_dir: str = "",
    mlx_base_url: str = "http://localhost:11434/v1",
    embedding_model: str = "BGE-M3",
) -> FastAPI:
    """Create and configure the Fusion-RAG FastAPI application."""
    app = FastAPI(
        title="Fusion-RAG",
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
    app.include_router(mcp_router)

    # Health check
    @app.get("/health")
    async def health():
        return {"status": "ok", "service": "fusion-rag"}

    return app


def run_server(
    host: str = "127.0.0.1",
    port: int = 11436,
    kb_storage_dir: str = "",
    mlx_base_url: str = "http://localhost:11434/v1",
    embedding_model: str = "BGE-M3",
    log_level: str = "INFO",
) -> None:
    """Run the Fusion-RAG server."""
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO))
    app = create_app(
        kb_storage_dir=kb_storage_dir,
        mlx_base_url=mlx_base_url,
        embedding_model=embedding_model,
    )
    uvicorn.run(app, host=host, port=port, log_level=log_level.lower())

if __name__ == "__main__":
    import os
    host = os.environ.get("FUSION_RAG_HOST", "127.0.0.1")
    port = int(os.environ.get("FUSION_RAG_PORT", "11436"))
    mlx_url = os.environ.get("FUSION_MLX_URL", "http://localhost:11434/v1")
    embed_model = os.environ.get("FUSION_RAG_EMBED", "BGE-M3")
    log_level = os.environ.get("FUSION_RAG_LOG_LEVEL", "INFO")
    run_server(
        host=host,
        port=port,
        mlx_base_url=mlx_url,
        embedding_model=embed_model,
        log_level=log_level,
    )
