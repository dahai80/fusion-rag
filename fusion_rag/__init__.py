"""Fusion-RAG — Apple Silicon native offline vector knowledge base backend.

All model inference (embeddings, chat) goes through fusion-mlx HTTP API.
Never imports MLX, mlx-lm, or any engine code directly.
"""

from .engine.knowledge_base import KnowledgeBase, KnowledgeBaseConfig
from .engine.document import DocumentParser, DocumentType
from .engine.chunker import Chunker
from .embed.client import EmbeddingClient
from .store.metadata_store import MetadataStore

__all__ = [
    "KnowledgeBase", "KnowledgeBaseConfig",
    "DocumentParser", "DocumentType",
    "Chunker",
    "EmbeddingClient",
    "MetadataStore",
]