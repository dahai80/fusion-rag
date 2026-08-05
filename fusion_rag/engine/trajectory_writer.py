from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_DIR = os.path.expanduser("~/.fusion/trajectories/rag")


class TrajectoryWriter:
    def __init__(self, dir_path: str | None = None):
        self._dir = Path(dir_path or os.environ.get("FUSION_TRAJECTORY_DIR", _DEFAULT_DIR))
        self._dir.mkdir(parents=True, exist_ok=True)
        self._file = self._dir / "rag_trajectories.jsonl"
        logger.info("TrajectoryWriter initialized: %s", self._file)

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
