"""Issue #104 — O-P2-3 startup probe must actually run (was a no-op coroutine).

EmbeddingClient.health() is async; the __main__ startup gate called it bare,
so the probe never executed and a fusion-rag booted with fusion-mlx down
passed unconditionally (plus a "never awaited" RuntimeWarning per boot).
_run_startup_probe is the extracted gate: it probes fusion-mlx /models
directly (NOT via health(), which accepts the local sentence-transformers
fallback and would mask a down MLX) and must raise when MLX is unreachable
so the __main__ handler logs the loud failure and exits non-zero.
"""

from __future__ import annotations

import warnings

import httpx
import pytest

from fusion_rag.api import server


class _FakeResponse:
    def __init__(self, status_code: int):
        self.status_code = status_code


class _FakeHttpClient:
    ok_status = 200

    async def get(self, path: str, timeout: float = 2.0) -> _FakeResponse:
        if type(self).ok_status == -1:
            # Simulate connection refused — the real httpx client raises.
            raise httpx.ConnectError("connection refused")
        return _FakeResponse(type(self).ok_status)


class _FakeEmbeddingClient:
    def __init__(self, **_):
        self.client = _FakeHttpClient()


def test_probe_raises_when_mlx_down(monkeypatch):
    monkeypatch.setattr(_FakeHttpClient, "ok_status", -1)
    monkeypatch.setattr(server, "EmbeddingClient", _FakeEmbeddingClient)
    with warnings.catch_warnings():
        # The original bug: a bare async call emits this warning and the probe
        # passes unconditionally. Turn it into an error so it can never regress.
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(httpx.ConnectError):
            server._run_startup_probe("http://127.0.0.1:1/v1", "BGE-M3", "key")


def test_probe_raises_on_non_200(monkeypatch):
    monkeypatch.setattr(_FakeHttpClient, "ok_status", 503)
    monkeypatch.setattr(server, "EmbeddingClient", _FakeEmbeddingClient)
    with pytest.raises(RuntimeError, match="non-200"):
        server._run_startup_probe("http://127.0.0.1:1/v1", "BGE-M3", "key")


def test_probe_passes_when_mlx_up(monkeypatch):
    monkeypatch.setattr(_FakeHttpClient, "ok_status", 200)
    monkeypatch.setattr(server, "EmbeddingClient", _FakeEmbeddingClient)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        server._run_startup_probe("http://127.0.0.1:11432/v1", "BGE-M3", "key")


def test_probe_request_actually_awaited(monkeypatch):
    # If the probe is ever called without awaiting, the request never runs and
    # this flag stays empty — the exact no-op shape of issue #104.
    calls = []

    class _TrackingClient:
        def __init__(self, **_):
            pass

        class client:  # accessed via instance, kept flat for the fake
            @staticmethod
            async def get(path: str, timeout: float = 2.0):
                calls.append(path)
                return _FakeResponse(200)

    monkeypatch.setattr(server, "EmbeddingClient", _TrackingClient)
    server._run_startup_probe("http://127.0.0.1:11432/v1", "BGE-M3", "key")
    assert calls == ["/models"]
