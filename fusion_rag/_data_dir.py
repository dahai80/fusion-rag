# Issue #96: single-point data-dir resolution via fusion_core.config.
# fusion-core's convention passes the SHORT app name ("rag" — see
# fusion-core tests/test_config_conventions.py), so the env var read is
# FUSION_RAG_DATA_DIR; passing the full "fusion-rag" name would double the
# FUSION_ prefix on the env lookup. The bare-app default (~/.rag) would
# orphan every existing store, so the pre-#96 location ~/.fusion-rag is
# passed as the explicit default. Call sites stay one-line: data_dir().

from __future__ import annotations

import logging
from pathlib import Path

from fusion_core.config import service_data_dir

logger = logging.getLogger(__name__)


def data_dir() -> Path:
    resolved = service_data_dir("rag", default=Path.home() / ".fusion-rag")
    logger.debug("fusion-rag data dir resolved: %s", resolved)
    return resolved
