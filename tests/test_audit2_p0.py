"""Audit-2 P0 regression tests: H1 rollback data-survival, D1 template filter,
D2 AST auto-detect, H4 path-scoped ACL. Each asserts real behavior the unit
suite missed (the first audit cycle's tests passed but never exercised these
defects)."""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from fusion_rag.api.server import create_app
from fusion_rag.engine.chunker import Chunker
from fusion_rag.engine.document import DocumentParser, DocumentType, ParseResult
from fusion_rag.engine.search_template import SearchTemplateManager
from fusion_rag.engine.version_manager import VersionManager


async def _fake_embed_batch(self, texts):
    # fusion-mlx has no MLX embedding model (BGE-M3 is pytorch, not safetensors),
    # so the real embed_batch 502s. These tests exercise ACL / store paths, not
    # embedding — return deterministic 1024-dim vectors (BGE-M3 dim) so the
    # store write path runs for real. Patched in via `new=` so `self` (the
    # EmbeddingClient instance) arrives as the first arg.
    return [[0.01] * 1024 for _ in texts]


_SENTINEL = object()


@pytest.fixture(autouse=True)
def _isolate_auth_singleton():
    # auth._auth_backend is a module-global cached singleton shared across the
    # whole test session. The H4 tests below inject a temp ApiKeyBackend; if a
    # test fails mid-body the teardown reset to None is skipped, and a poisoned
    # enabled-backend leaks into later modules (test_coverage_final POSTs with
    # no key → 401 → KeyError 'id'). Snapshot + always-restore so the singleton
    # + env revert no matter how the test exits. pytest-randomly would otherwise
    # surface order-dependent flakiness.
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


# ── H1: rollback must NOT destroy data ──


class TestH1RollbackDataSurvival:
    def _seed_kb(self, storage: Path, doc_count: int) -> None:
        storage.mkdir(parents=True, exist_ok=True)
        vectors = storage / "vectors"
        vectors.mkdir(exist_ok=True)
        (vectors / "data.lance").write_text(f"vec-{doc_count}")
        (storage / "metadata.db").write_text(f"meta-{doc_count}")
        (storage / "bm25_index.db").write_text(f"bm25-{doc_count}")

    def test_rollback_restores_snapshot_data_in_place(self, tmp_path):
        # The H1 bug: rollback returned success:True but left the live dir
        # empty (snapshot was nested inside the moved live dir, so copy-back
        # read from a moved path that no longer existed, then rmtree'd the
        # backup holding both live + snapshot). Assert data survives.
        storage = tmp_path / "kb_h1"
        self._seed_kb(storage, doc_count=5)
        vm = VersionManager(str(tmp_path / "versions.db"))
        snap = vm.create_snapshot("kb_h1", str(storage), "baseline")

        # Mutate live data after snapshot. vectors/ is snapshotted via hard
        # links (os.link), so an in-place truncate would corrupt the snapshot
        # inode too — real LanceDB never truncates in place (append-only: it
        # writes new fragment files). Mimic that: unlink+rewrite breaks the
        # hard link, leaving the snapshot's copy intact, then diverges live.
        (storage / "vectors" / "data.lance").unlink()
        (storage / "vectors" / "data.lance").write_text("vec-MUTATED")
        # metadata.db is a full copy (copy2), so in-place rewrite is fine.
        (storage / "metadata.db").write_text("meta-MUTATED")

        result = vm.rollback("kb_h1", str(storage), snap["version_id"])

        assert result["success"] is True
        # Snapshot artifacts restored to live dir.
        assert (storage / "vectors" / "data.lance").read_text() == "vec-5"
        assert (storage / "metadata.db").read_text() == "meta-5"
        assert (storage / "bm25_index.db").read_text() == "bm25-5"
        # Snapshot itself survives (rollback must not delete it).
        assert Path(snap["snapshot_path"]).exists()

    def test_rollback_does_not_move_or_delete_live_dir(self, tmp_path):
        # H1 root cause was shutil.move(kb_path, backup) then rmtree(backup).
        # The live dir must stay on disk throughout; no _backup_ sibling may
        # appear or persist.
        storage = tmp_path / "kb_h1b"
        self._seed_kb(storage, doc_count=1)
        vm = VersionManager(str(tmp_path / "versions.db"))
        snap = vm.create_snapshot("kb_h1b", str(storage), "s1")

        vm.rollback("kb_h1b", str(storage), snap["version_id"])

        assert storage.exists(), "live kb dir must survive rollback"
        backups = list(storage.parent.glob("*_backup_*"))
        assert backups == [], f"no backup dir must persist, found: {backups}"

    def test_rollback_preserves_admin_dbs_not_in_snapshot(self, tmp_path):
        # versions.db / permissions.db / templates.db live in kb_storage_path
        # but are NOT snapshotted. The old move-live-dir pattern wiped them
        # too. In-place restore must leave them intact.
        storage = tmp_path / "kb_h1c"
        self._seed_kb(storage, doc_count=2)
        (storage / "permissions.db").write_text("perms-LIVE")
        (storage / "templates.db").write_text("tpl-LIVE")
        vm = VersionManager(str(tmp_path / "versions.db"))
        snap = vm.create_snapshot("kb_h1c", str(storage), "s1")

        vm.rollback("kb_h1c", str(storage), snap["version_id"])

        assert (storage / "permissions.db").read_text() == "perms-LIVE"
        assert (storage / "templates.db").read_text() == "tpl-LIVE"

    def test_snapshot_lives_outside_live_dir(self, tmp_path):
        # H1 fix invariant: snapshot MUST be outside kb_storage_path.
        storage = tmp_path / "kb_h1d"
        self._seed_kb(storage, doc_count=1)
        vm = VersionManager(str(tmp_path / "versions.db"))
        snap = vm.create_snapshot("kb_h1d", str(storage), "s1")
        snap_path = Path(snap["snapshot_path"])
        assert storage not in snap_path.parents, "snapshot must NOT nest inside kb_storage_path"
        assert snap_path.exists()


# ── D1: builtin templates must filter (lowercase enum values) ──


class TestD1TemplateFilterMatches:
    def test_code_builtin_filter_contains_lowercase_values(self, tmp_path):
        mgr = SearchTemplateManager(str(tmp_path / "templates.db"))
        tpl = mgr.get_template("", "code")
        assert tpl is not None, "builtin 'code' template must exist"
        # DocumentType.CODE_PYTHON.value == "code_python" (lowercase). The bug
        # stored uppercase "CODE_PYTHON" which never matched a stored doc_type.
        for v in tpl["doc_type_filter"]:
            assert v == v.lower(), f"builtin filter value must be lowercase: {v}"

    def test_design_builtin_filter_matches_stored_doc_types(self, tmp_path):
        mgr = SearchTemplateManager(str(tmp_path / "templates.db"))
        tpl = mgr.get_template("", "design")
        assert tpl is not None
        # Stored doc_type is DocumentType.MARKDOWN.value = "markdown" (routes
        # persist result.doc_type.value). The bug stored "MD"/"HTML".
        assert "markdown" in tpl["doc_type_filter"]
        assert "html" in tpl["doc_type_filter"]
        # A doc stored with doc_type="markdown" must pass the filter.
        assert "markdown" in tpl["doc_type_filter"]
        sample_doc_type = DocumentType.MARKDOWN.value
        assert sample_doc_type in tpl["doc_type_filter"], (
            "stored doc_type must match builtin filter (case was the bug)"
        )

    def test_existing_uppercase_rows_migrated_on_startup(self, tmp_path):
        # _seed_builtins UPSERTs builtins on every construction. An install that
        # previously seeded uppercase rows (pre-fix) must be rewritten to
        # lowercase on the next manager open. Simulate a stale uppercase row.
        import json
        import sqlite3

        db = tmp_path / "templates.db"
        conn = sqlite3.connect(str(db))
        conn.execute(
            "CREATE TABLE search_templates (name TEXT, kb_id TEXT, description TEXT, "
            "alpha REAL, rerank INTEGER, top_k INTEGER, threshold REAL, "
            "rewrite_mode TEXT, doc_type_filter TEXT, is_builtin INTEGER, "
            "created_at REAL, PRIMARY KEY (kb_id, name))"
        )
        conn.execute(
            "CREATE TABLE template_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT INTO search_templates (name, kb_id, doc_type_filter, is_builtin, "
            "alpha, rerank, top_k, threshold, rewrite_mode, description, created_at) "
            "VALUES ('code', '', ?, 1, 0.4, 1, 15, 0.3, 'expand', 'old', 0)",
            (json.dumps(["CODE_PYTHON", "CODE_OTHER"]),),
        )
        conn.execute("INSERT INTO template_meta (key, value) VALUES ('builtins_seeded', '1')")
        conn.commit()
        conn.close()

        # Reopen — UPSERT rewrites the builtin row in place.
        mgr = SearchTemplateManager(str(db))
        tpl = mgr.get_template("", "code")
        assert all(v == v.lower() for v in tpl["doc_type_filter"])
        assert "code_python" in tpl["doc_type_filter"]


# ── D2: AST auto-detect must fire for .py files ──


class TestD2AstAutoDetect:
    def test_should_use_ast_true_for_python(self):
        chunker = Chunker(strategy="semantic", chunk_size=512)
        result = ParseResult(
            file_path="x.py",
            file_name="x.py",
            content="x = 1\n",
            chars=6,
            doc_type=DocumentType.CODE_PYTHON,
            metadata={},
        )
        # The bug: compared dtype == "CODE_PYTHON" but dtype is the .value
        # "code_python" → always False → AST path dead unless strategy="ast".
        assert chunker._should_use_ast(result) is True

    def test_should_use_ast_false_for_non_python(self):
        chunker = Chunker(strategy="semantic", chunk_size=512)
        result = ParseResult(
            file_path="x.md",
            file_name="x.md",
            content="# hi\n",
            chars=5,
            doc_type=DocumentType.MARKDOWN,
            metadata={},
        )
        assert chunker._should_use_ast(result) is False

    def test_should_use_ast_false_when_strategy_is_code(self):
        # Explicit strategy="code" skips AST (caller chose generic code path).
        chunker = Chunker(strategy="code", chunk_size=512)
        result = ParseResult(
            file_path="x.py",
            file_name="x.py",
            content="x = 1\n",
            chars=6,
            doc_type=DocumentType.CODE_PYTHON,
            metadata={},
        )
        assert chunker._should_use_ast(result) is False

    def test_python_file_chunks_via_ast_auto_detect(self, tmp_path):
        # End-to-end: a .py file with strategy="semantic" must use the AST
        # chunker (auto-detected), not the semantic paragraph splitter. AST
        # chunks split on top-level defs/classes; semantic splits on headings.
        py_file = tmp_path / "mod.py"
        py_file.write_text(
            "def alpha():\n    return 1\n\n\ndef beta():\n    return 2\n\n\nclass Gamma:\n    pass\n"
        )
        parser = DocumentParser()
        result = asyncio.run(parser.parse(str(py_file)))
        chunker = Chunker(strategy="semantic", chunk_size=512)
        chunks = asyncio.run(chunker.chunk(result))
        # AST chunker yields per top-level node — at least 3 chunks here.
        assert len(chunks) >= 3, f"AST auto-detect expected >=3 chunks, got {len(chunks)}"
        texts = [c.text for c in chunks]
        assert any("def alpha" in t for t in texts)
        assert any("class Gamma" in t for t in texts)


# ── H4: path-scoped ACL must deny out-of-scope doc ──


class TestH4PathScopedACL:
    @pytest.fixture
    def client(self, tmp_path):
        # create_app has no admin_api_key param; the admin key comes from the
        # auth backend. Inject a fresh ApiKeyBackend with a temp auth.db so the
        # subject keys added below do NOT pollute the real ~/.fusion-rag/auth.db.
        from fusion_rag.api import auth as auth_mod
        from fusion_rag.embed.client import EmbeddingClient

        backend = auth_mod.ApiKeyBackend(admin_key="admin-key", db_path=str(tmp_path / "auth.db"))
        auth_mod._auth_backend = backend
        storage_dir = tempfile.mkdtemp()
        app = create_app(kb_storage_dir=storage_dir)
        # fusion-mlx has no embed model -> patch embed_batch so ingest lands docs
        # for real (the store + metadata write runs; only the LLM/embed step is
        # faked). The ACL path-scoping is what these tests assert.
        with patch.object(EmbeddingClient, "embed_batch", _fake_embed_batch), TestClient(app) as tc:
            tc.kb_storage_dir = storage_dir
            yield tc
        auth_mod._auth_backend = None  # reset singleton for the rest of the suite

    def test_subtree_rule_denies_out_of_scope_doc(self, client):
        # H4 bug: resource_path hardcoded "/" made every path-prefixed rule
        # match, so a subject restricted to /restricted/* could read/delete a
        # doc under /public/*. With the fix, a rule scoped to /restricted/
        # must DENY a doc whose file_path is /public/x.
        admin = {"X-API-Key": "admin-key"}
        create = client.post("/kb/bases", json={"name": "aclkb", "kb_id": "aclkb"}, headers=admin).json()
        kb_id = create["id"]
        # Ingest two inline docs under distinct synthetic paths by writing
        # metadata directly via the admin key, then create an ACL rule.
        # Use the ingest endpoint with distinct doc_name to set doc_path.
        # inline:// doc_path = "inline://<doc_name>".
        for name in ("restricted_doc", "public_doc"):
            r = client.post(
                f"/kb/bases/{kb_id}/documents/ingest",
                json={"content": f"body {name}", "content_type": "text", "doc_name": name,
                      "contextualize": False},
                headers=admin,
            )
            assert r.status_code == 200, r.text

        # Register an API key for subject "alice", scope her to /restricted.
        # The ingest path is inline://restricted_doc / inline://public_doc.
        client.post(
            f"/kb/bases/{kb_id}/permissions",
            json={
                "subject": "alice",
                "resource_type": "document",
                "resource_path": "inline://restricted_doc",
                "actions": ["read", "delete"],
            },
            headers=admin,
        )

        # List docs (admin) to get doc_ids. list_documents returns the documents
        # table rows (PK column is `id`, with file_path/file_name/doc_type).
        docs = client.get(f"/kb/bases/{kb_id}/documents", headers=admin).json()
        assert len(docs) == 2
        by_path = {d.get("file_path"): d for d in docs}
        public_doc_id = by_path["inline://public_doc"]["id"]
        restricted_doc_id = by_path["inline://restricted_doc"]["id"]

        # Register alice's key directly in the auth backend (no admin route to
        # add API keys exists), then exercise path-scoped ACL.
        from fusion_rag.api.auth import get_auth_backend

        backend = get_auth_backend()
        backend.auth_config.add_key("alice-key", "alice")

        # Alice CAN read her scoped doc's status.
        r = client.get(
            f"/kb/bases/{kb_id}/documents/{restricted_doc_id}/status",
            headers={"X-API-Key": "alice-key"},
        )
        assert r.status_code == 200, f"alice should read in-scope doc: {r.status_code} {r.text}"

        # Alice CANNOT read the out-of-scope public doc — H4 fix enforces path.
        r = client.get(
            f"/kb/bases/{kb_id}/documents/{public_doc_id}/status",
            headers={"X-API-Key": "alice-key"},
        )
        assert r.status_code == 403, f"alice must be denied out-of-scope doc: {r.status_code} {r.text}"

    def test_subtree_rule_allows_in_scope_delete(self, client):
        admin = {"X-API-Key": "admin-key"}
        create = client.post("/kb/bases", json={"name": "aclkb2", "kb_id": "aclkb2"}, headers=admin).json()
        kb_id = create["id"]
        client.post(
            f"/kb/bases/{kb_id}/documents/ingest",
            json={"content": "scoped body", "content_type": "text", "doc_name": "keep",
                  "contextualize": False},
            headers=admin,
        )
        client.post(
            f"/kb/bases/{kb_id}/permissions",
            json={
                "subject": "bob",
                "resource_type": "document",
                "resource_path": "inline://keep",
                "actions": ["delete"],
            },
            headers=admin,
        )
        from fusion_rag.api.auth import get_auth_backend

        backend = get_auth_backend()
        backend.auth_config.add_key("bob-key", "bob")

        docs = client.get(f"/kb/bases/{kb_id}/documents", headers=admin).json()
        doc_id = docs[0]["id"]
        r = client.delete(
            f"/kb/bases/{kb_id}/documents/{doc_id}",
            headers={"X-API-Key": "bob-key"},
        )
        assert r.status_code == 200, f"bob should delete in-scope doc: {r.status_code} {r.text}"
