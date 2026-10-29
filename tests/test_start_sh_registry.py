"""Issue #98: start.sh register-then-start self-check (config Phase 3.3).

check_port_registry() must refuse to start on a port missing from
port-registry.yaml, pass when the entry is present, allow a
FUSION_SKIP_REGISTRY_CHECK=1 bypass, and warn+continue when the registry
file itself is absent (standalone deploy). Sourcing start.sh runs the
default `status` dispatch, which is read-only and exits 0 — safe in tests.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

_REGISTRY_WITH_ENTRY = """services:
  fusion-rag:
    port: 11437
    purpose: test stub
"""

_REGISTRY_WITHOUT_ENTRY = """services:
  some-other-service:
    port: 11437
    purpose: test stub
"""


def _run_check(registry: Path | None, port: str, skip: str = ""):
    env = dict(os.environ)
    env["FUSION_REGISTRY"] = str(registry) if registry is not None else "/nonexistent/port-registry.yaml"
    env["FUSION_RAG_PORT"] = port
    env.pop("FUSION_SKIP_REGISTRY_CHECK", None)
    if skip:
        env["FUSION_SKIP_REGISTRY_CHECK"] = skip
    return subprocess.run(
        ["/bin/bash", "-c", "source ./start.sh && check_port_registry"],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_refuses_unregistered_port(tmp_path):
    reg = tmp_path / "port-registry.yaml"
    reg.write_text(_REGISTRY_WITHOUT_ENTRY)
    result = _run_check(reg, "11437")
    assert result.returncode == 1
    assert "refusing to start" in result.stderr


def test_passes_with_registered_port(tmp_path):
    reg = tmp_path / "port-registry.yaml"
    reg.write_text(_REGISTRY_WITH_ENTRY)
    result = _run_check(reg, "11437")
    assert result.returncode == 0


def test_bypass_env_allows_unregistered_port(tmp_path):
    reg = tmp_path / "port-registry.yaml"
    reg.write_text(_REGISTRY_WITHOUT_ENTRY)
    result = _run_check(reg, "11437", skip="1")
    assert result.returncode == 0
    assert "bypassed" in result.stderr


def test_missing_registry_warns_and_continues():
    result = _run_check(None, "11437")
    assert result.returncode == 0
    assert "skipped" in result.stderr
