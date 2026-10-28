"""Issue #96: single-point config resolution via fusion_core.config.

Every resolver (client key alias chain, service port, service data dir) lives
in fusion-core; this repo keeps one-line call sites. These tests pin the
contract at the fusion-rag call sites: EmbeddingClient / CrossEncoderReranker
resolve the key via client_key(), the server port via service_port(), and the
stores root via service_data_dir().
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fusion_core.config import client_key, resolve_api_key, service_port

from fusion_rag.embed.client import EmbeddingClient

_KEY_ENVS = ("FUSION_CLIENT_KEY", "FUSION_MLX_API_KEY", "MLX_API_KEY")


def _clear_key_envs(monkeypatch):
    for env in _KEY_ENVS:
        monkeypatch.delenv(env, raising=False)
    monkeypatch.delenv("FUSION_PROFILE", raising=False)


def test_client_key_primary_env(monkeypatch):
    _clear_key_envs(monkeypatch)
    monkeypatch.setenv("FUSION_CLIENT_KEY", "primary-key")
    assert client_key() == "primary-key"
    assert EmbeddingClient(cache_enabled=False).api_key == "primary-key"


def test_client_key_legacy_alias_fusion_mlx_api_key(monkeypatch):
    _clear_key_envs(monkeypatch)
    monkeypatch.setenv("FUSION_MLX_API_KEY", "legacy-key")
    assert client_key() == "legacy-key"
    assert EmbeddingClient(cache_enabled=False).api_key == "legacy-key"


def test_client_key_legacy_alias_mlx_api_key(monkeypatch):
    _clear_key_envs(monkeypatch)
    monkeypatch.setenv("MLX_API_KEY", "bare-key")
    assert client_key() == "bare-key"


def test_client_key_missing_dev_is_empty(monkeypatch):
    _clear_key_envs(monkeypatch)
    assert client_key() == ""
    assert EmbeddingClient(cache_enabled=False).api_key == ""


def test_client_key_missing_prod_raises(monkeypatch):
    _clear_key_envs(monkeypatch)
    monkeypatch.setenv("FUSION_PROFILE", "prod")
    with pytest.raises(RuntimeError):
        client_key()
    with pytest.raises(RuntimeError):
        EmbeddingClient(cache_enabled=False)


def test_explicit_api_key_beats_env(monkeypatch):
    _clear_key_envs(monkeypatch)
    monkeypatch.setenv("FUSION_CLIENT_KEY", "env-key")
    assert EmbeddingClient(api_key="explicit-key", cache_enabled=False).api_key == "explicit-key"


def test_cross_encoder_reranker_resolves_key(monkeypatch):
    _clear_key_envs(monkeypatch)
    monkeypatch.setenv("FUSION_CLIENT_KEY", "rerank-key")
    from fusion_rag.engine.cross_encoder_reranker import CrossEncoderReranker

    reranker = CrossEncoderReranker(mlx_base_url="http://127.0.0.1:11434", model="bge-reranker-v2-m3")
    assert reranker._api_key == "rerank-key"


def test_resolve_api_key_reads_settings_json(monkeypatch, tmp_path):
    # resolve_api_key is the local-first ~/.fusion-mlx/settings.json fallback
    # (client_key covers the env chain only). server __main__ chains both:
    # client_key() or resolve_api_key().
    import json

    _clear_key_envs(monkeypatch)
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"auth": {"api_key": "settings-key"}}))
    assert resolve_api_key(settings_path=settings) == "settings-key"


def test_service_port_env(monkeypatch):
    # fusion-core convention: short app name ("rag") — reads FUSION_RAG_PORT.
    monkeypatch.delenv("FUSION_RAG_PORT", raising=False)
    assert service_port("rag", default=11436) == 11436
    monkeypatch.setenv("FUSION_RAG_PORT", "12345")
    assert service_port("rag", default=11436) == 12345


def test_data_dir_default_and_override(monkeypatch, tmp_path):
    # The repo-wide helper keeps the pre-#96 default ~/.fusion-rag (an
    # explicit default — fusion-core's bare-app default would be ~/.rag and
    # orphan every existing store) while FUSION_RAG_DATA_DIR still relocates.
    from fusion_rag._data_dir import data_dir

    monkeypatch.delenv("FUSION_RAG_DATA_DIR", raising=False)
    assert data_dir() == Path.home() / ".fusion-rag"
    monkeypatch.setenv("FUSION_RAG_DATA_DIR", str(tmp_path))
    assert data_dir() == tmp_path


def test_kb_manager_storage_dir_follows_data_dir(monkeypatch, tmp_path):
    # Issue #96: the default stores root derives from service_data_dir, so
    # FUSION_RAG_DATA_DIR relocates the whole store tree (kb_meta + vectors).
    monkeypatch.setenv("FUSION_RAG_DATA_DIR", str(tmp_path))
    from fusion_rag.engine.knowledge_base import KnowledgeBaseManager

    manager = KnowledgeBaseManager()
    assert Path(manager.storage_dir) == tmp_path / "stores"
    assert (tmp_path / "stores").is_dir()
