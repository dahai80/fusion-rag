"""Fusion-RAG — Apple Silicon native offline vector knowledge base backend.

All model inference (embeddings, chat) goes through fusion-mlx HTTP API.
Never imports MLX, mlx-lm, or any engine code directly.
"""

from .embed.client import EmbeddingClient
from .engine.chunker import Chunker
from .engine.document import DocumentParser, DocumentType
from .engine.knowledge_base import KnowledgeBase, KnowledgeBaseConfig
from .store.metadata_store import MetadataStore

__all__ = [
    "Chunker",
    "DocumentParser",
    "DocumentType",
    "EmbeddingClient",
    "KnowledgeBase",
    "KnowledgeBaseConfig",
    "MetadataStore",
]
