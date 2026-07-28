"""Knowledge base management — create, configure, list, and delete knowledge bases.

All model calls go through fusion-mlx HTTP API (/v1/embeddings, /v1/chat/completions).
No direct MLX imports.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeBaseConfig:
    """Configuration for a single knowledge base."""

    name: str
    description: str = ""
    chunk_strategy: str = "semantic"  # "semantic", "code", "fixed"
    chunk_size: int = 512
    chunk_overlap: int = 64
    embedding_model: str = "BGE-M3"  # Model name for fusion-mlx
    max_results: int = 10
    similarity_threshold: float = 0.6
    language: str = "zh"  # "zh", "en", "auto"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "chunk_strategy": self.chunk_strategy,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedding_model": self.embedding_model,
            "max_results": self.max_results,
            "similarity_threshold": self.similarity_threshold,
            "language": self.language,
        }

    @classmethod
    def from_dict(cls, data: dict) -> KnowledgeBaseConfig:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class KnowledgeBase:
    """A single knowledge base with isolated storage and vector index."""

    id: str = ""
    config: KnowledgeBaseConfig = field(default_factory=KnowledgeBaseConfig)
    storage_path: str = ""
    storage_dir: str = ""
    file_count: int = 0
    chunk_count: int = 0
    created_at: float = 0.0
    updated_at: float = 0.0

    def __post_init__(self):
        if not self.id:
            self.id = uuid.uuid4().hex[:16]
        if not self.storage_path:
            base = Path(self.storage_dir or Path.home() / ".fusion-rag" / "stores") / self.id
            self.storage_path = str(base)
        now = time.time()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now

    @property
    def vector_path(self) -> str:
        return str(Path(self.storage_path) / "vectors")

    @property
    def metadata_path(self) -> str:
        return str(Path(self.storage_path) / "metadata.db")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            **self.config.to_dict(),
            "storage_path": self.storage_path,
            "file_count": self.file_count,
            "chunk_count": self.chunk_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> KnowledgeBase:
        cfg_data = {k: data[k] for k in KnowledgeBaseConfig.__dataclass_fields__ if k in data}
        config = KnowledgeBaseConfig(**cfg_data)
        return cls(
            id=data.get("id", ""),
            config=config,
            storage_path=data.get("storage_path", ""),
            storage_dir=data.get("storage_dir", ""),
            file_count=data.get("file_count", 0),
            chunk_count=data.get("chunk_count", 0),
            created_at=data.get("created_at", 0.0),
            updated_at=data.get("updated_at", 0.0),
        )


class KnowledgeBaseManager:
    """Manages multiple knowledge bases with CRUD operations."""

    def __init__(self, storage_dir: str = ""):
        if not storage_dir:
            storage_dir = str(Path.home() / ".fusion-rag" / "stores")
        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._bases: dict[str, KnowledgeBase] = {}
        self._load_all()

    def _load_all(self) -> None:
        """Load all knowledge bases from disk."""
        for kb_dir in self._storage_dir.iterdir():
            if kb_dir.is_dir():
                meta_file = kb_dir / "kb_meta.json"
                if meta_file.exists():
                    import json
                    try:
                        data = json.loads(meta_file.read_text(encoding="utf-8"))
                        kb = KnowledgeBase.from_dict(data)
                        self._bases[kb.id] = kb
                    except Exception as e:
                        logger.warning("Failed to load KB meta from %s: %s", meta_file, e)

    def _save_meta(self, kb: KnowledgeBase) -> None:
        """Save knowledge base metadata to disk."""
        import json
        path = Path(kb.storage_path)
        path.mkdir(parents=True, exist_ok=True)
        meta_file = path / "kb_meta.json"
        meta_file.write_text(json.dumps(kb.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    def create(self, name: str, description: str = "",
               chunk_strategy: str = "semantic", embedding_model: str = "BGE-M3") -> KnowledgeBase:
        """Create a new knowledge base."""
        config = KnowledgeBaseConfig(
            name=name, description=description,
            chunk_strategy=chunk_strategy, embedding_model=embedding_model,
        )
        kb = KnowledgeBase(config=config, storage_dir=str(self._storage_dir))
        self._bases[kb.id] = kb
        self._save_meta(kb)
        return kb

    def get(self, kb_id: str) -> KnowledgeBase:
        """Get a knowledge base by ID."""
        if kb_id not in self._bases:
            raise KeyError(f"Knowledge base '{kb_id}' not found")
        return self._bases[kb_id]

    def list(self) -> list[dict[str, Any]]:
        """List all knowledge bases."""
        return [kb.to_dict() for kb in self._bases.values()]

    def delete(self, kb_id: str) -> bool:
        """Delete a knowledge base and all its data."""
        if kb_id not in self._bases:
            return False
        kb = self._bases[kb_id]
        import shutil
        shutil.rmtree(Path(kb.storage_path), ignore_errors=True)
        del self._bases[kb_id]
        return True

    def update(self, kb_id: str, **kwargs) -> KnowledgeBase:
        """Update knowledge base configuration."""
        kb = self.get(kb_id)
        for key, value in kwargs.items():
            if hasattr(kb.config, key):
                setattr(kb.config, key, value)
        kb.updated_at = time.time()
        self._save_meta(kb)
        return kb

    @property
    def count(self) -> int:
        return len(self._bases)
