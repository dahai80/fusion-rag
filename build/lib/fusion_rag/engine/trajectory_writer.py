from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_DIR = os.path.expanduser("~/.fusion/trajectories/rag")
# R6: unbounded append blew up the disk on a high-QPS deployment (~17GB/day at
# 100 QPS). Cap + rotate instead of writing one file forever. Both knobs env-
# overridable; defaults keep a bounded tail without operator action.
_DEFAULT_MAX_MB = 100
_DEFAULT_KEEP = 5


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        val = int(raw)
        return val if val > 0 else default
    except ValueError:
        logger.warning("invalid %s=%r, using default %d", name, raw, default)
        return default


class TrajectoryWriter:
    def __init__(self, dir_path: str | None = None):
        self._dir = Path(dir_path or os.environ.get("FUSION_TRAJECTORY_DIR", _DEFAULT_DIR))
        self._dir.mkdir(parents=True, exist_ok=True)
        self._file = self._dir / "rag_trajectories.jsonl"
        self._max_bytes = _env_int("FUSION_RAG_TRAJECTORY_MAX_MB", _DEFAULT_MAX_MB) * 1024 * 1024
        self._keep = _env_int("FUSION_RAG_TRAJECTORY_KEEP", _DEFAULT_KEEP)
        logger.info(
            "TrajectoryWriter initialized: %s max_mb=%d keep=%d",
            self._file,
            self._max_bytes // (1024 * 1024),
            self._keep,
        )

    def _maybe_rotate(self) -> None:
        # R6: rotate when the active file crosses the size cap. Keep the last
        # _keep rotated files (.jsonl.1 .. .jsonl.<keep>). Drop the oldest
        # first, shift the rest up high-index-first, then active -> .1.
        # Best-effort; a rotation failure logs + continues (next write retries).
        try:
            if not self._file.exists() or self._file.stat().st_size < self._max_bytes:
                return
        except OSError as e:
            logger.warning("trajectory rotate stat failed: %s", e)
            return
        oldest = self._file.with_suffix(f".jsonl.{self._keep}")
        if oldest.exists():
            oldest.unlink(missing_ok=True)
        for i in range(self._keep - 1, 0, -1):
            src = self._file.with_suffix(f".jsonl.{i}")
            if src.exists():
                try:
                    os.replace(src, self._file.with_suffix(f".jsonl.{i + 1}"))
                except OSError as e:
                    logger.warning("trajectory rotate shift .%d failed: %s", i, e)
        try:
            os.replace(self._file, self._file.with_suffix(".jsonl.1"))
        except OSError as e:
            logger.warning("trajectory rotate active->.1 failed: %s", e)

    def write(
        self,
        kb_id: str,
        query: str,
        caller: str,
        results_count: int,
        top_sources: list[dict],
        latency_ms: float,
        metadata: dict | None = None,
    ) -> None:
        record = {
            "ts": time.time(),
            "kb_id": kb_id,
            "query": query,
            "caller": caller,
            "results_count": results_count,
            "top_sources": top_sources,
            "latency_ms": latency_ms,
            "empty": results_count == 0,
            "metadata": metadata or {},
        }
        try:
            self._maybe_rotate()
            with open(self._file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            logger.debug(
                "trajectory written: kb_id=%s caller=%s results=%d latency=%.1fms empty=%s",
                kb_id,
                caller,
                results_count,
                latency_ms,
                record["empty"],
            )
        except Exception as e:
            logger.warning("Failed to write trajectory: %s", e)
