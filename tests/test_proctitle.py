"""Tests for issue #92: the server process title must show the product name
(fusion-rag serve :PORT) instead of a bare Python."""

from __future__ import annotations

import sys
import types
from unittest.mock import patch

from fusion_rag.api.server import _apply_process_title


class TestApplyProcessTitle:
    def test_sets_title_when_setproctitle_available(self):
        calls = []
        fake = types.SimpleNamespace(setproctitle=lambda t: calls.append(t))
        with patch.dict(sys.modules, {"setproctitle": fake}):
            _apply_process_title(11436)
        assert calls == ["fusion-rag serve :11436"]

    def test_no_crash_when_setproctitle_missing(self):
        with patch.dict(sys.modules, {"setproctitle": None}):
            _apply_process_title(11436)

    def test_no_crash_when_setproctitle_raises(self):
        def _boom(title):
            raise RuntimeError("no argv on this platform")

        fake = types.SimpleNamespace(setproctitle=_boom)
        with patch.dict(sys.modules, {"setproctitle": fake}):
            _apply_process_title(11436)
