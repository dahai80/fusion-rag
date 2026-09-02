"""Knowledge base management — create, configure, list, and delete knowledge bases.

All model calls go through fusion-mlx HTTP API (/v1/embeddings, /v1/chat/completions).
No direct MLX imports.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _atomic_write_json(path: Path, data: dict) -> None:
    # P0-11: write_text truncates then writes — a crash/power loss mid-write
    # leaves kb_meta.json empty or partial, and the next _load_all swallows the
    # JSONDecodeError, silently dropping the KB. Atomic pattern: write a tmp
    # file, fsync, os.replace (atomic rename on POSIX). A cross-process flock
    # on a sidecar lockfile serializes concurrent workers (uvicorn --workers N)
    # so two processes never interleave a read-modify-write of the same KB meta.
    payload = json.dumps(data, indent=2, ensure_ascii=False)
    lock_path = path.with_name(path.name + ".lock")
    tmp_path = path.with_name(path.name + ".tmp")
    with open(lock_path, "w") as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        try:
            with open(tmp_path, "w", encoding="utf-8") as tmp_fd:
                tmp_fd.write(payload)
                tmp_fd.flush()
                os.fsync(tmp_fd.fileno())
            os.replace(tmp_path, path)
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)


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
    similarity_threshold: float = 0.3
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
        # P4-7: surface unknown keys instead of silently dropping them. A typo
        # in a persisted field (e.g. "chunk_strategie") used to vanish here and
        # the config silently reverted to the default — undetectable. Log a
        # warning naming the dropped keys so the drift is visible, not silent.
        known = cls.__dataclass_fields__
        unknown = [k for k in data if k not in known]
        if unknown:
            logger.warning(
                "KnowledgeBaseConfig.from_dict ignoring unknown keys: %s", unknown
            )
        return cls(**{k: v for k, v in data.items() if k in known})


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
    # Issue #61: the authoritative tenant this KB belongs to. None = no tenant
    # isolation (single-tenant local dev, or a KB created before tenant
    # scoping). Set at create time from the gateway-stamped X-Fusion-Tenant.
    # KBs with tenant=None are visible to every caller (backward compat); a
    # caller whose request tenant is set only sees KBs whose tenant matches.
    tenant: str | None = None

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
            "tenant": self.tenant,
        }

    @classmethod
    def from_dict(cls, data: dict) -> KnowledgeBase:
        # P4-7: this is the LOAD path (reading persisted kb_meta.json). An empty
        # id here means the meta file is corrupt/truncated, not a fresh create.
        # The prior code let __post_init__ regenerate a uuid4 — so the next
        # save wrote kb_meta.json under a NEW id while the orphaned on-disk
        # vectors/metadata sat under the old (now unknown) path forever, and
        # any caller holding the old id 404'd. That is silent data loss.
        # Distinguish the two cases: fresh create goes through KnowledgeBase()
        # directly (id="" → __post_init__ regenerates, which is correct); load
        # goes through from_dict, where an empty id must fail loudly so the
        # operator recovers the meta file rather than writing a fresh one.
        kb_id = data.get("id", "")
        if not kb_id:
            raise ValueError(
                "KnowledgeBase.from_dict: persisted meta has empty/missing 'id' — "
                "refusing to regenerate (would orphan existing storage). "
                f"Fix or delete the corrupt meta file. data keys={list(data.keys())}"
            )
        # storage_path empty on load is the same class of silent-orphan risk:
        # __post_init__ would derive one from storage_dir + the id, masking the
        # fact that the persisted path was lost. Warn + let __post_init__ rebuild
        # only when storage_dir is present; if BOTH are empty the derived path
        # lands under ~/.fusion-rag/stores/{id} which is the documented default,
        # so that specific case is recoverable, not a silent orphan.
        if not data.get("storage_path") and not data.get("storage_dir"):
            logger.warning(
                "KnowledgeBase.from_dict: id=%s has no storage_path nor storage_dir; "
                "falling back to default ~/.fusion-rag/stores/%s",
                kb_id,
                kb_id,
            )
        cfg_data = {k: data[k] for k in KnowledgeBaseConfig.__dataclass_fields__ if k in data}
        config = KnowledgeBaseConfig(**cfg_data)
        return cls(
            id=kb_id,
            config=config,
            storage_path=data.get("storage_path", ""),
            storage_dir=data.get("storage_dir", ""),
            file_count=data.get("file_count", 0),
            chunk_count=data.get("chunk_count", 0),
            created_at=data.get("created_at", 0.0),
            updated_at=data.get("updated_at", 0.0),
            tenant=data.get("tenant"),
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

    @property
    def storage_dir(self) -> str:
        # O-P2-3: public read accessor for the stores root — /ready's writability
        # probe + server.py read it. Prior code reached for the private
        # _storage_dir at the call site, but the probe used the missing public
        # name and crashed the readiness check (503 on a healthy store).
        return str(self._storage_dir)

    def _load_all(self) -> None:
        """Load all knowledge bases from disk."""
        for kb_dir in self._storage_dir.iterdir():
            if kb_dir.is_dir():
                meta_file = kb_dir / "kb_meta.json"
                if meta_file.exists():
                    self._load_meta_file(meta_file)

    def _load_meta_file(self, meta_file: Path) -> None:
        # P0-11: was `except Exception: logger.warning(...)` — a corrupted
        # kb_meta.json (truncated by a non-atomic write crash) was swallowed and
        # the KB silently dropped, invisible to list/get. Fail visibly now: a
        # JSON decode error is a data-loss event, not a warning. We still keep
        # iterating (one bad KB must not abort loading the rest), but log at
        # ERROR and surface the corrupt path so the operator can recover it.
        try:
            data = json.loads(meta_file.read_text(encoding="utf-8"))
            kb = KnowledgeBase.from_dict(data)
            self._bases[kb.id] = kb
        except json.JSONDecodeError as e:
            logger.error("CORRUPT kb_meta.json at %s (will not load KB): %s", meta_file, e)
        except Exception as e:
            logger.error("Failed to load KB meta from %s: %s", meta_file, e)

    def _try_reload_from_disk(self, kb_id: str) -> KnowledgeBase | None:
        # P0-11: cross-worker reconciliation. With uvicorn --workers N each
        # worker owns its own in-memory _bases; a KB created by worker A is
        # invisible to worker B, so B's get(kb_id) 404s on a KB that provably
        # exists on disk. On a get miss, re-read that one KB's meta file from
        # disk (cheap, one file) instead of failing. This is eventually
        # consistent: deletes are still best-effort cross-worker (a worker may
        # serve a stale cached KB), but the common create-then-search flow works.
        meta_file = self._storage_dir / kb_id / "kb_meta.json"
        if meta_file.exists():
            self._load_meta_file(meta_file)
            return self._bases.get(kb_id)
        return None

    def _save_meta(self, kb: KnowledgeBase) -> None:
        """Save knowledge base metadata to disk (atomically)."""
        path = Path(kb.storage_path)
        path.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(path / "kb_meta.json", kb.to_dict())

    def create(
        self,
        name: str,
        description: str = "",
        chunk_strategy: str = "semantic",
        embedding_model: str = "BGE-M3",
        kb_id: str = "",
        tenant: str | None = None,
    ) -> KnowledgeBase:
        """Create a new knowledge base. If kb_id given and already exists, return existing (idempotent).

        Issue #61: `tenant` stamps the authoritative tenant on the new KB. None
        = no tenant isolation (single-tenant local dev). The idempotent return
        path checks the tenant matches when scoping is in effect — a caller
        whose tenant differs from an existing kb_id's tenant does NOT get a
        silent cross-tenant return; it creates a fresh KB (or, if kb_id is
        caller-supplied and collides across tenants, the validate_identifier
        gate upstream prevents the collision in practice).
        """
        if kb_id and kb_id in self._bases:
            existing = self._bases[kb_id]
            # Issue #61: if tenant isolation is in effect and the existing KB
            # belongs to a different tenant, do NOT return it — that would leak
            # another tenant's KB id/handle. Treat as not-found so the caller
            # creates a distinct KB. The None==None case (both unscoped) and
            # the matching-tenant case both return the existing KB.
            if existing.tenant == tenant:
                logger.info("KB '%s' already exists, returning existing", kb_id)
                return existing
            logger.warning(
                "create: kb_id=%s exists under tenant=%r but caller tenant=%r — not returning cross-tenant",
                kb_id,
                existing.tenant,
                tenant,
            )
        config = KnowledgeBaseConfig(
            name=name,
            description=description,
            chunk_strategy=chunk_strategy,
            embedding_model=embedding_model,
        )
        kb = KnowledgeBase(id=kb_id or "", config=config, storage_dir=str(self._storage_dir), tenant=tenant)
        self._bases[kb.id] = kb
        self._save_meta(kb)
        logger.info("Created KB '%s' (id=%s, tenant=%s)", name, kb.id, tenant)
        return kb

    def get(self, kb_id: str, tenant: str | None = None, require_tenant_match: bool = False) -> KnowledgeBase:
        """Get a knowledge base by ID.

        Issue #61: when `tenant` is not None and `require_tenant_match` is True,
        a KB whose tenant does not match raises KeyError (404) — a tenant-A
        caller cannot address tenant-B's KB by id. When `tenant` is None (no
        isolation in effect) or `require_tenant_match` is False, no filtering
        happens (backward compat for internal/admin callers that list/get
        without a request tenant).
        """
        if kb_id not in self._bases:
            # P0-11: cross-worker reconciliation — re-read from disk on miss.
            reloaded = self._try_reload_from_disk(kb_id)
            if reloaded is None:
                raise KeyError(f"Knowledge base '{kb_id}' not found")
            kb = reloaded
        else:
            kb = self._bases[kb_id]
        if require_tenant_match and tenant is not None and kb.tenant != tenant:
            logger.warning(
                "get: kb=%s tenant=%r denied to caller tenant=%r — 404",
                kb_id,
                kb.tenant,
                tenant,
            )
            raise KeyError(f"Knowledge base '{kb_id}' not found")
        return kb

    def list(self, tenant: str | None = None, require_tenant_match: bool = False) -> list[dict[str, Any]]:
        """List all knowledge bases.

        Issue #61: when `tenant` is not None and `require_tenant_match` is True,
        only KBs whose tenant matches are returned (plus KBs with tenant=None
        are excluded — a tenant-scoped caller must not see unscoped legacy
        KBs). When `tenant` is None, all KBs are returned (backward compat).
        """
        if require_tenant_match and tenant is not None:
            return [kb.to_dict() for kb in self._bases.values() if kb.tenant == tenant]
        return [kb.to_dict() for kb in self._bases.values()]

    def delete(self, kb_id: str) -> bool:
        """Delete a knowledge base and all its data."""
        if kb_id not in self._bases:
            return False
        kb = self._bases[kb_id]
        import shutil

        # R4: evict the pooled VectorStore handle BEFORE rmtree. A pooled
        # VectorStore holds an open LanceDB/HNSW handle on the KB's vectors dir.
        # Rmtree under it leaves a stale handle; if the KB id is later reused
        # (create with same kb_id), the pool returns the stale handle pointing
        # at the now-recreated dir → ENOENT / corrupt reads. Drop + close first.
        self._evict_vec_store_pool(kb.storage_path)
        shutil.rmtree(Path(kb.storage_path), ignore_errors=True)
        del self._bases[kb_id]
        return True

    @staticmethod
    def _evict_vec_store_pool(storage_path: str) -> None:
        # R4: best-effort pool eviction. No-op outside the server (no app.state
        # bound, e.g. unit tests / direct manager calls). Lazy import avoids a
        # cycle (app_state imports KnowledgeBaseManager).
        try:
            from ..api.app_state import get_vec_store_pool, get_vec_store_pool_lock

            pool = get_vec_store_pool()
        except Exception:
            return
        lock = get_vec_store_pool_lock()
        vector_path = str(Path(storage_path) / "vectors")
        with lock:
            vs = pool.pop(vector_path, None)
        if vs is not None:
            try:
                vs.close()
            except Exception as e:
                logger.warning("delete: pooled vec_store close failed for %s: %s", vector_path, e)
            logger.info("delete: evicted pooled vec_store for %s", vector_path)

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
