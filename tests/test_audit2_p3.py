"""P3 audit fixes: R5 metrics, D3 dead ResultCache, D4 config env, D5 dup map.

Raw pytest (rtk proxy) — rtk masks errors. Order-independent across seeds.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

# ── D3: ResultCache deleted from streaming ──


class TestD3ResultCacheRemoved:
    def test_import_resultcache_raises(self):
        # D3: ResultCache was dead code (never instantiated). Deletion verified
        # by import failing — it must no longer exist on the module.
        from fusion_rag.engine import streaming

        assert not hasattr(streaming, "ResultCache"), "ResultCache should be deleted"

    def test_import_name_fails_directly(self):
        # importing the deleted name raises ImportError, not AttributeError-ish
        with pytest.raises(ImportError):
            from fusion_rag.engine.streaming import ResultCache  # noqa: F401


# ── D5: CONTENT_TYPE_MAP single source of truth ──


class TestD5ContentTypeMap:
    def test_map_exists_on_document(self):
        from fusion_rag.engine.document import CONTENT_TYPE_MAP, DocumentType

        assert CONTENT_TYPE_MAP["markdown"] is DocumentType.MARKDOWN
        assert CONTENT_TYPE_MAP["html"] is DocumentType.HTML
        assert CONTENT_TYPE_MAP["text"] is DocumentType.TXT

    def test_routes_docs_uses_shared_map(self):
        # D5: routes_docs must import the shared CONTENT_TYPE_MAP, not keep a
        # divergent inline copy. Inspect the module source for the import.
        import inspect

        from fusion_rag.api import routes_docs

        src = inspect.getsource(routes_docs)
        assert "from ..engine.document import" in src
        assert "CONTENT_TYPE_MAP" in src
        # no inline dict literal redefining content-type → DocumentType
        assert '"markdown": DocumentType' not in src.replace(" ", "")


# ── D4: RuntimeConfig env overrides ──


class TestD4RuntimeConfig:
    def test_defaults_match_hardcoded(self):
        from fusion_rag.engine.runtime_config import get_runtime_config, reset_runtime_config

        reset_runtime_config()
        cfg = get_runtime_config()
        assert cfg.scan_max_files == 1000
        assert cfg.embedding_cache_ttl_seconds == 86400 * 7
        assert cfg.embedding_cache_max_entries == 100_000
        assert cfg.rag_token_budget == 8192
        assert cfg.rag_max_history_turns == 10
        assert cfg.search_fetch_k_multiplier == 4
        reset_runtime_config()

    def test_env_override(self, monkeypatch):
        from fusion_rag.engine.runtime_config import get_runtime_config, reset_runtime_config

        reset_runtime_config()
        monkeypatch.setenv("FUSION_RAG_SCAN_MAX_FILES", "50")
        monkeypatch.setenv("FUSION_RAG_TOKEN_BUDGET", "2048")
        monkeypatch.setenv("FUSION_RAG_FETCH_K_MULTIPLIER", "8")
        cfg = get_runtime_config()
        assert cfg.scan_max_files == 50
        assert cfg.rag_token_budget == 2048
        assert cfg.search_fetch_k_multiplier == 8
        reset_runtime_config()

    def test_env_invalid_falls_back(self, monkeypatch):
        from fusion_rag.engine.runtime_config import get_runtime_config, reset_runtime_config

        reset_runtime_config()
        monkeypatch.setenv("FUSION_RAG_SCAN_MAX_FILES", "not-a-number")
        cfg = get_runtime_config()
        assert cfg.scan_max_files == 1000  # default on invalid
        reset_runtime_config()

    def test_env_below_minimum_falls_back(self, monkeypatch):
        from fusion_rag.engine.runtime_config import get_runtime_config, reset_runtime_config

        reset_runtime_config()
        monkeypatch.setenv("FUSION_RAG_SCAN_MAX_FILES", "0")
        cfg = get_runtime_config()
        assert cfg.scan_max_files == 1000  # minimum=1, 0 rejected
        reset_runtime_config()

    def test_to_dict_roundtrip(self):
        from fusion_rag.engine.runtime_config import get_runtime_config, reset_runtime_config

        reset_runtime_config()
        d = get_runtime_config().to_dict()
        assert d["scan_max_files"] == 1000
        reset_runtime_config()


# ── D4: /admin/config endpoint ──


class TestD4ConfigEndpoint:
    def test_config_endpoint_returns_runtime(self, tmp_path):
        from fusion_rag.api import auth as auth_mod
        from fusion_rag.api import server
        from fusion_rag.api.auth import ApiKeyBackend

        auth_mod._auth_backend = ApiKeyBackend(admin_key="admin-key", db_path=str(tmp_path / "auth.db"))
        app = server.create_app(kb_storage_dir=str(tmp_path / "kb"))
        client = TestClient(app)
        resp = client.get("/kb/config", headers={"X-API-Key": "admin-key"})
        assert resp.status_code == 200
        body = resp.json()
        assert "runtime" in body
        assert body["runtime"]["scan_max_files"] == 1000
        auth_mod._auth_backend = None


# ── R5: metrics recording + /metrics endpoint ──


class TestR5Metrics:
    def test_record_and_snapshot(self):
        from fusion_rag.engine.metrics import get_metrics

        m = get_metrics()
        m.reset()
        m.record("/search", "kb1", 200, 12.5)
        m.record("/search", "kb1", 200, 30.0)
        m.record("/search", "kb1", 500, 1500.0)
        snap = m.snapshot()
        assert snap["requests"]["/search|kb1|2xx"] == 2
        assert snap["requests"]["/search|kb1|5xx"] == 1
        assert snap["errors"]["/search|kb1|5xx"] == 1
        lat = snap["latency"]["/search|kb1"]
        assert lat["count"] == 3
        assert lat["sum"] == pytest.approx(1542.5)
        m.reset()

    def test_render_prometheus_format(self):
        from fusion_rag.engine.metrics import get_metrics

        m = get_metrics()
        m.reset()
        m.record("/kb/bases/{kb_id}/search", "kb1", 200, 25.0)
        text = m.render_prometheus()
        assert "# TYPE fusion_rag_requests_total counter" in text
        assert "# TYPE fusion_rag_request_latency_ms histogram" in text
        assert "fusion_rag_requests_total" in text
        assert 'endpoint="/kb/bases/{kb_id}/search"' in text
        assert 'kb_id="kb1"' in text
        assert "fusion_rag_request_latency_ms_bucket" in text
        assert 'le="+Inf"' in text
        assert "fusion_rag_request_latency_ms_count" in text
        m.reset()

    def test_metrics_endpoint_exposes_text(self, tmp_path):
        from fusion_rag.api import auth as auth_mod
        from fusion_rag.api import server
        from fusion_rag.api.auth import ApiKeyBackend

        auth_mod._auth_backend = ApiKeyBackend(admin_key="admin-key", db_path=str(tmp_path / "auth.db"))
        app = server.create_app(kb_storage_dir=str(tmp_path / "kb"))
        client = TestClient(app)
        # hit health (skipped from metrics) then metrics (skipped from metrics)
        client.get("/health")
        # S-P2-2: /metrics now gates on verify_api_key (same as write endpoints).
        # NoAuth -> open; ApiKey backend -> requires a valid key. Present the
        # admin key so the authed scrape succeeds; a missing key now 401s.
        resp = client.get("/metrics", headers={"X-API-Key": "admin-key"})
        assert resp.status_code == 200
        assert "fusion_rag_requests_total" in resp.text or "fusion_rag_request_latency_ms" in resp.text
        auth_mod._auth_backend = None

    def test_metrics_recorded_on_request(self, tmp_path):
        from fusion_rag.api import auth as auth_mod
        from fusion_rag.api import server
        from fusion_rag.api.auth import ApiKeyBackend
        from fusion_rag.engine.metrics import get_metrics

        m = get_metrics()
        m.reset()
        auth_mod._auth_backend = ApiKeyBackend(admin_key="admin-key", db_path=str(tmp_path / "auth.db"))
        app = server.create_app(kb_storage_dir=str(tmp_path / "kb"))
        client = TestClient(app)
        client.get("/health")
        snap = m.snapshot()
        # /health is skipped, so no request should be recorded for it
        health_keys = [k for k in snap["requests"] if "health" in k]
        assert health_keys == []
        m.reset()
        auth_mod._auth_backend = None
