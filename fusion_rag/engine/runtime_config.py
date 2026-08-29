"""D4: RuntimeConfig — env-driven operator knobs, single source of truth.

Before this, runtime tuning (scan max_files, embedding-cache TTL/size, RAG
token budget / max history, search fetch_k multiplier) was hardcoded across
8+ files. An operator could not raise the cache size or shrink the token
budget without a code change + redeploy. RuntimeConfig centralizes those
knobs, reads each from env at startup, and exposes a read-only view via
/admin/config. Defaults match the prior hardcoded values (no behavior change
unless an env is set)."""
from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        val = int(raw)
    except ValueError:
        logger.warning("invalid %s=%r, using default %d", name, raw, default)
        return default
    if val < minimum:
        logger.warning("%s=%d below minimum %d, using default %d", name, val, minimum, default)
        return default
    return val


@dataclass(frozen=True)
class RuntimeConfig:
    # D4: every field is env-overridable. Defaults are the prior hardcoded
    # values so a deployment that sets no envs behaves exactly as before.
    scan_max_files: int = 1000
    embedding_cache_ttl_seconds: int = 86400 * 7
    embedding_cache_max_entries: int = 100_000
    rag_token_budget: int = 8192
    rag_max_history_turns: int = 10
    search_fetch_k_multiplier: int = 4
    max_content_chars: int = 2_000_000
    max_batch_files: int = 200

    @classmethod
    def from_env(cls) -> RuntimeConfig:
        return cls(
            scan_max_files=_env_int("FUSION_RAG_SCAN_MAX_FILES", 1000, minimum=1),
            embedding_cache_ttl_seconds=_env_int(
                "FUSION_RAG_EMBED_CACHE_TTL", 86400 * 7, minimum=0
            ),
            embedding_cache_max_entries=_env_int(
                "FUSION_RAG_EMBED_CACHE_MAX_ENTRIES", 100_000, minimum=1
            ),
            rag_token_budget=_env_int("FUSION_RAG_TOKEN_BUDGET", 8192, minimum=1),
            rag_max_history_turns=_env_int("FUSION_RAG_MAX_HISTORY_TURNS", 10, minimum=1),
            search_fetch_k_multiplier=_env_int(
                "FUSION_RAG_FETCH_K_MULTIPLIER", 4, minimum=1
            ),
            max_content_chars=_env_int(
                "FUSION_RAG_MAX_CONTENT_CHARS", 2_000_000, minimum=1
            ),
            max_batch_files=_env_int("FUSION_RAG_MAX_BATCH_FILES", 200, minimum=1),
        )

    def to_dict(self) -> dict:
        return asdict(self)


_config: RuntimeConfig | None = None


def get_runtime_config() -> RuntimeConfig:
    global _config
    if _config is None:
        _config = RuntimeConfig.from_env()
        logger.info("RuntimeConfig loaded: %s", _config.to_dict())
    return _config


def reset_runtime_config() -> None:
    # tests: force the next get_runtime_config() to re-read env.
    global _config
    _config = None
