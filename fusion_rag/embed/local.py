"""Local embedding engine — fallback using sentence-transformers when MLX embedding unavailable."""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_MODEL_ALIASES: dict[str, str] = {
    "BGE-M3": "BAAI/bge-m3",
    "bge-m3": "BAAI/bge-m3",
    "bge_m3": "BAAI/bge-m3",
}

_local_model: Any = None
_local_model_name: str = ""


def _resolve_model_name(model_name: str) -> str:
    return _MODEL_ALIASES.get(model_name, model_name)


def get_local_model(model_name: str = "BAAI/bge-m3") -> Any:
    global _local_model, _local_model_name
    resolved = _resolve_model_name(model_name)
    if _local_model is not None and _local_model_name == resolved:
        return _local_model
    try:
        from sentence_transformers import SentenceTransformer

        cache_dir = os.path.expanduser("~/.fusion-mlx/models")
        logger.info("Loading local embedding model: %s -> %s (cache=%s)", model_name, resolved, cache_dir)
        _local_model = SentenceTransformer(resolved, cache_folder=cache_dir)
        _local_model_name = resolved
        try:
            dim = _local_model.get_embedding_dimension()
        except AttributeError:
            dim = _local_model.get_sentence_embedding_dimension()
        logger.info("Local embedding model loaded: %s, dim=%d", resolved, dim)
        return _local_model
    except ImportError:
        logger.warning("sentence-transformers not installed, local embedding unavailable")
        return None
    except Exception as e:
        logger.error("Failed to load local embedding model: %s", e)
        return None


def embed_local(texts: list[str], model_name: str = "BAAI/bge-m3") -> list[list[float]]:
    model = get_local_model(model_name)
    if model is None:
        return [[0.0] * 1024 for _ in texts]
    try:
        embeddings = model.encode(texts, normalize_embeddings=True)
        return [emb.tolist() for emb in embeddings]
    except Exception as e:
        logger.error("Local embedding encode failed: %s", e)
        return [[0.0] * 1024 for _ in texts]
