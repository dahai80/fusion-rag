"""Audit-2 P2 regression tests: R3 watch cap+persist+restore, R6 trajectory
rotation + audit retention env, H3 single-process doc, D6 BM25 degraded,
D7 embedding_model mismatch guard. Each asserts real behavior the first audit
cycle's tests did not exercise."""
from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from fusion_rag.api.server import create_app
from fusion_rag.engine.bm25_index import BM25Index
from fusion_rag.engine.trajectory_writer import TrajectoryWriter


async def _fake_embed_batch(self, texts):
    return [[0.01] * 1024 for _ in texts]


_SENTINEL = object()


@pytest.fixture(autouse=True)
def _isolate_auth_singleton():
    import os

    from fusion_rag.api import auth as auth_mod

    saved_backend = auth_mod._auth_backend
    saved_env = os.environ.get("FUSION_RAG_API_KEY", _SENTINEL)
    yield
    if saved_env is _SENTINEL:
        os.environ.pop("FUSION_RAG_API_KEY", None)
    else:
        os.environ["FUSION_RAG_API_KEY"] = saved_env
    auth_mod._auth_backend = saved_backend


# ── D6: BM25 load failure marks degraded, search refuses ──


class TestD6BM25Degraded:
    def test_malformed_stats_marks_degraded_and_search_refuses(self, tmp_path):
        # D6: a load failure (here: corrupt stats JSON in bm25_meta) used to
        # warn + silently start an empty index — keyword recall vanished with
        # no signal. Now load marks _degraded + search() refuses (returns []
        # with a warning), surfacing the break instead of serving empty results
        # as if real. Build a valid SQLite file whose bm25_meta row is unparseable.
        db = tmp_path / "bm25_index.db"
        conn = sqlite3.connect(str(db))
        conn.execute("CREATE TABLE bm25_meta (key TEXT PRIMARY KEY, value TEXT)")
        conn.execute("CREATE TABLE bm25_docs (doc_id TEXT, text TEXT, doc_len INTEGER, doc_path TEXT)")
        conn.execute("CREATE TABLE bm25_inverted (token TEXT, doc_id TEXT, tf INTEGER)")
        conn.execute("INSERT INTO bm25_meta (key, value) VALUES ('stats', 'NOT_JSON{')")
        conn.commit()
        conn.close()
        idx = BM25Index(str(db))
        assert idx._degraded is True, "malformed stats must set _degraded"
        assert idx._degraded_reason, "degraded reason must be recorded"
        # search refuses on a degraded index (does not silently return [])
        results = idx.search("anything", top_k=5)
        assert results == [], "degraded search returns empty (refuses, not fresh-start)"

    def test_healthy_db_not_degraded(self, tmp_path):
        idx = BM25Index(str(tmp_path / "ok.db"))
        assert idx._degraded is False
        assert idx.search("x", top_k=5) == []  # empty corpus, not degraded


# ── H3: app_state documents single-process-only deployment ──


class TestH3SingleProcessDoc:
    def test_app_state_docstring_states_single_process_constraint(self):
        # H3: the module docstring claimed "per-process app.state 保证多 worker
        # 正确" which was false — SqliteBase uses an in-process RLock, not a
        # cross-process lock; multi-worker/multi-node sharing one stores dir
        # hits `database is locked` or corrupts. The fix documents the real
        # constraint explicitly. Operators must NOT deploy multi-worker without
        # a cross-process backend.
        from fusion_rag.api import app_state

        doc = app_state.__doc__ or ""
        assert "single-process" in doc.lower(), "docstring must state single-process constraint"
        assert "workers" in doc.lower(), "docstring must warn about multi-worker"
        assert "database is locked" in doc.lower(), "docstring must name the failure mode"


# ── R6: trajectory rotation respects size cap + keep N ──


class TestR6TrajectoryRotation:
    def _writer(self, tmp_path, max_mb=1, keep=3):
        # tiny cap so a few writes trigger rotation without writing MBs.
        with patch.dict(
            os.environ,
            {"FUSION_RAG_TRAJECTORY_MAX_MB": str(max_mb), "FUSION_RAG_TRAJECTORY_KEEP": str(keep)},
        ):
            return TrajectoryWriter(dir_path=str(tmp_path))

    def test_rotation_creates_keep_files_when_cap_exceeded(self, tmp_path):
        # R6: a single file appended forever blew up disk. Now it rotates at the
        # size cap, keeping the last `keep` rotated files (.jsonl.1 .. .jsonl.N).
        w = self._writer(tmp_path, max_mb=0, keep=3)
        # max_mb=0 → _env_int rejects 0 (val>0) → default. Use a tiny real cap.
        w._max_bytes = 60  # force a small cap deterministically
        for i in range(20):
            w.write("kb", f"query {i}", "caller", 0, [], 1.0, {})
        rotated = sorted(tmp_path.glob("rag_trajectories.jsonl.*"))
        assert rotated, "rotation must produce at least one rotated file"
        # at most `keep` rotated files (oldest dropped)
        assert len(rotated) <= 3, f"keep=3 caps rotated files, got {len(rotated)}"
        assert (tmp_path / "rag_trajectories.jsonl").exists(), "active file still present"

    def test_env_keep_zero_falls_back_to_default(self, tmp_path):
        # invalid env (0/negative) must fall back to default, not 0 (which would
        # delete every rotated file and keep nothing).
        with patch.dict(os.environ, {"FUSION_RAG_TRAJECTORY_KEEP": "0"}):
            w = TrajectoryWriter(dir_path=str(tmp_path))
            assert w._keep > 0, "keep=0 invalid → default, not 0"

    def test_rotation_drops_oldest_beyond_keep(self, tmp_path):
        # write enough to rotate past `keep`; the highest suffix must not exceed keep.
        w = self._writer(tmp_path, max_mb=1, keep=2)
        w._max_bytes = 40
        for i in range(30):
            w.write("kb", f"q{i}", "c", 0, [], 1.0, {})
        rotated = list(tmp_path.glob("rag_trajectories.jsonl.*"))
        suffixes = sorted(int(p.suffix.split(".")[-1]) for p in rotated)
        assert max(suffixes) <= 2, f"keep=2 → no suffix >2, got {suffixes}"


# ── R6: audit retention via env + periodic reaper ──


class TestR6AuditRetention:
    def test_retention_zero_means_forever(self, tmp_path):
        # R6: FUSION_RAG_AUDIT_RETENTION_DAYS=0 disables pruning (keep forever).
        with patch.dict(os.environ, {"FUSION_RAG_AUDIT_RETENTION_DAYS": "0"}):
            from fusion_rag.engine.audit_logger import AuditLogger

            log = AuditLogger(str(tmp_path / "audit.db"))
            assert log.retention_seconds == 0, "0 days → retention off (forever)"

    def test_retention_env_read_when_arg_none(self, tmp_path):
        with patch.dict(os.environ, {"FUSION_RAG_AUDIT_RETENTION_DAYS": "7"}):
            from fusion_rag.engine.audit_logger import AuditLogger

            log = AuditLogger(str(tmp_path / "audit.db"))
            assert log.retention_seconds == 7 * 86400

    def test_periodic_reaper_prunes_after_threshold(self, tmp_path):
        # R6: prune ran once at construction; a long-running process never
        # pruned again. The reaper fires every _REAPER_EVERY inserts.
        from fusion_rag.engine.audit_logger import AuditLogger

        log = AuditLogger(str(tmp_path / "audit.db"), retention_days=1)
        log._REAPER_EVERY = 3  # small threshold for the test
        pruned = []
        log.prune = lambda: pruned.append(1) or 0
        for _ in range(3):
            log.log_search("kb", "q", "c", 0, [], 1.0, {})
        assert pruned, "reaper must fire after _REAPER_EVERY inserts"


# ── R3: watch cap rejects over-limit with 429 ──


class TestR3WatchCap:
    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        from fusion_rag.api import auth as auth_mod
        from fusion_rag.embed.client import EmbeddingClient

        backend = auth_mod.ApiKeyBackend(admin_key="admin-key", db_path=str(tmp_path / "auth.db"))
        auth_mod._auth_backend = backend
        # tiny cap + isolated stores dir so registry writes do not touch ~/.fusion-rag
        stores = tmp_path / "stores"
        monkeypatch.setenv("FUSION_RAG_WATCH_CAP", "2")
        monkeypatch.setenv("FUSION_RAG_STORES_DIR", str(stores))
        storage_dir = tempfile.mkdtemp()
        app = create_app(kb_storage_dir=storage_dir)
        with patch.object(EmbeddingClient, "embed_batch", _fake_embed_batch), TestClient(app) as tc:
            yield tc
        auth_mod._auth_backend = None

    def _make_watchable_file(self, client, kb_id, tmp_path):
        # watch requires real files under the ingest root. Use a file in tmp.
        f = tmp_path / "watched.txt"
        f.write_text("seed")
        return str(f)

    def test_watch_over_cap_returns_429(self, client, tmp_path):
        admin = {"X-API-Key": "admin-key"}
        kb_id = client.post("/kb/bases", json={"name": "wkb", "kb_id": "wkb"}, headers=admin).json()["id"]
        f = self._make_watchable_file(client, kb_id, tmp_path)
        # cap=2 → first two succeed, third 429s
        for i in range(2):
            r = client.post(
                f"/kb/bases/{kb_id}/watch",
                json={"file_paths": [f], "poll_interval": 30},
                headers=admin,
            )
            assert r.status_code == 200, f"watch {i} should succeed: {r.text}"
        r3 = client.post(
            f"/kb/bases/{kb_id}/watch",
            json={"file_paths": [f], "poll_interval": 30},
            headers=admin,
        )
        assert r3.status_code == 429, f"third watch over cap must 429: {r3.status_code} {r3.text}"

    def test_watch_registry_persisted(self, client, tmp_path):
        admin = {"X-API-Key": "admin-key"}
        kb_id = client.post("/kb/bases", json={"name": "wkb2", "kb_id": "wkb2"}, headers=admin).json()["id"]
        f = self._make_watchable_file(client, kb_id, tmp_path)
        client.post(f"/kb/bases/{kb_id}/watch", json={"file_paths": [f]}, headers=admin)
        # registry written under the stores dir set by the fixture
        import json

        reg = Path(os.environ["FUSION_RAG_STORES_DIR"]) / "watch_registry.json"
        assert reg.exists(), "watch registry must be persisted"
        data = json.loads(reg.read_text())
        assert len(data) >= 1, "at least one watch persisted"
        any_watch = next(iter(data.values()))
        assert any_watch["kb_id"] == kb_id
        assert f in any_watch["file_paths"]


# ── R3: watches restore on startup ──


class TestR3WatchRestore:
    def test_persisted_watch_restored_on_new_app(self, tmp_path):
        # R3: a restart used to silently drop every directory monitor. The
        # registry persists active watches; a fresh app re-spawns _watch_loop
        # for each persisted watch whose files still exist.
        from fusion_rag.embed.client import EmbeddingClient

        stores = tmp_path / "stores"
        stores.mkdir()
        watched = tmp_path / "live.txt"
        watched.write_text("seed")
        # write a registry the restore path will read
        import json

        reg = stores / "watch_registry.json"
        reg.write_text(
            json.dumps(
                {
                    "wid1": {
                        "watch_id": "wid1",
                        "kb_id": "restorekb",
                        "file_paths": [str(watched)],
                        "poll_interval": 30,
                        "changes_detected": 0,
                    }
                }
            )
        )
        os.environ["FUSION_RAG_STORES_DIR"] = str(stores)
        try:
            app = create_app(kb_storage_dir=str(tmp_path / "kbstore"))
            with patch.object(EmbeddingClient, "embed_batch", _fake_embed_batch), TestClient(app) as tc:
                # init_app_state ran in lifespan → restore populated app.watches
                watches = tc.app.state.watches
                assert "wid1" in watches, "persisted watch must be restored on startup"
                assert watches["wid1"]["active"] is True
                assert str(watched) in watches["wid1"]["file_paths"]
        finally:
            os.environ.pop("FUSION_RAG_STORES_DIR", None)

    def test_restore_skips_watch_with_no_existing_files(self, tmp_path):
        # a watch whose files were deleted since persist must NOT be restored
        # (re-indexing a missing file would error forever).
        from fusion_rag.embed.client import EmbeddingClient

        stores = tmp_path / "stores"
        stores.mkdir()
        import json

        (stores / "watch_registry.json").write_text(
            json.dumps(
                {
                    "gone": {
                        "watch_id": "gone",
                        "kb_id": "xkb",
                        "file_paths": [str(tmp_path / "deleted.txt")],
                        "poll_interval": 30,
                        "changes_detected": 0,
                    }
                }
            )
        )
        os.environ["FUSION_RAG_STORES_DIR"] = str(stores)
        try:
            app = create_app(kb_storage_dir=str(tmp_path / "kbstore2"))
            with patch.object(EmbeddingClient, "embed_batch", _fake_embed_batch), TestClient(app) as tc:
                assert "gone" not in tc.app.state.watches, "watch with no existing files must be skipped"
        finally:
            os.environ.pop("FUSION_RAG_STORES_DIR", None)


# ── D7: embedding_model mismatch rejects ingest ──


class TestD7EmbedModelGuard:
    @pytest.fixture
    def client(self, tmp_path):
        from fusion_rag.api import auth as auth_mod
        from fusion_rag.embed.client import EmbeddingClient

        backend = auth_mod.ApiKeyBackend(admin_key="admin-key", db_path=str(tmp_path / "auth.db"))
        auth_mod._auth_backend = backend
        storage_dir = tempfile.mkdtemp()
        app = create_app(kb_storage_dir=storage_dir)
        with patch.object(EmbeddingClient, "embed_batch", _fake_embed_batch), TestClient(app) as tc:
            tc.kb_storage_dir = storage_dir
            yield tc
        auth_mod._auth_backend = None

    def test_ingest_rejected_when_kb_model_differs(self, client):
        # D7: a KB configured with embedding_model != service-wide model must
        # be rejected at ingest (would persist cross-model vectors → mixed-dim
        # recall corruption). The service runs a single model (BGE-M3 default).
        admin = {"X-API-Key": "admin-key"}
        create = client.post(
            "/kb/bases",
            json={"name": "mkb", "kb_id": "mkb", "embedding_model": "text-embedding-3-small"},
            headers=admin,
        ).json()
        kb_id = create["id"]
        r = client.post(
            f"/kb/bases/{kb_id}/documents/ingest",
            json={"content": "body", "content_type": "text", "contextualize": False},
            headers=admin,
        )
        assert r.status_code == 400, f"mismatched model must 400: {r.status_code} {r.text}"
        assert "mismatch" in r.text.lower()

    def test_ingest_allowed_when_kb_model_matches(self, client):
        admin = {"X-API-Key": "admin-key"}
        create = client.post(
            "/kb/bases",
            json={"name": "okkb", "kb_id": "okkb", "embedding_model": "BGE-M3"},
            headers=admin,
        ).json()
        kb_id = create["id"]
        r = client.post(
            f"/kb/bases/{kb_id}/documents/ingest",
            json={"content": "body", "content_type": "text", "contextualize": False},
            headers=admin,
        )
        assert r.status_code == 200, f"matching model must ingest: {r.status_code} {r.text}"

    def test_ingest_allowed_when_kb_model_unset(self, client):
        # default KB (no embedding_model set) must still ingest — the guard
        # only fires when kb.config.embedding_model is truthy AND differs.
        admin = {"X-API-Key": "admin-key"}
        kb_id = client.post(
            "/kb/bases", json={"name": "def", "kb_id": "def"}, headers=admin
        ).json()["id"]
        r = client.post(
            f"/kb/bases/{kb_id}/documents/ingest",
            json={"content": "body", "content_type": "text", "contextualize": False},
            headers=admin,
        )
        assert r.status_code == 200, f"default model must ingest: {r.status_code} {r.text}"

    def test_scan_rejected_when_kb_model_differs(self, client, tmp_path):
        admin = {"X-API-Key": "admin-key"}
        kb_id = client.post(
            "/kb/bases",
            json={"name": "skb", "kb_id": "skb", "embedding_model": "wrong-model"},
            headers=admin,
        ).json()["id"]
        d = tmp_path / "scanroot"
        d.mkdir()
        (d / "a.txt").write_text("hello")
        r = client.post(
            f"/kb/bases/{kb_id}/scan",
            json={"dir_path": str(d), "contextualize": False},
            headers=admin,
        )
        assert r.status_code == 400, f"scan with mismatched model must 400: {r.status_code} {r.text}"
