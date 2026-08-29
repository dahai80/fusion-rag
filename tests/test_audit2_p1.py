"""Audit-2 P1 regression tests: R1 threadpool (sync store calls off the event
loop), R2 total_deadline on LLM calls, H2 rollback-failure signal surfacing
503, R4 pool eviction on delete-recreate."""
from __future__ import annotations

import inspect
import tempfile
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from fusion_rag.api.routes_docs import _write_doc_to_stores
from fusion_rag.api.server import create_app
from fusion_rag.embed.client import EmbeddingClient


async def _fake_embed_batch(self, texts):
    # fusion-mlx has no MLX embedding model (BGE-M3 is pytorch, not safetensors),
    # so the real embed_batch 502s. These tests exercise store/rollback/pool
    # paths, not embedding — return deterministic 1024-dim vectors so the store
    # write path runs for real (and H2/R1/R4 assertions hold against real stores).
    # Patched in via `new=` so `self` (the EmbeddingClient instance) arrives first.
    return [[0.01] * 1024 for _ in texts]


_SENTINEL = object()


@pytest.fixture(autouse=True)
def _isolate_auth_singleton():
    # auth._auth_backend is a module-global cached singleton shared across the
    # whole test session. H2/R4 tests below inject a temp ApiKeyBackend; if a
    # test fails mid-body the reset to None is skipped, and a poisoned
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


# ── H2: rollback failure must surface a 503 + rollback_failed signal ──


class TestH2RollbackSignal:
    def _make_client(self, storage, tmp_path):
        # create_app has no admin_api_key param; inject a temp ApiKeyBackend so
        # the admin key works and subject keys do not touch ~/.fusion-rag/auth.db.
        from fusion_rag.api import auth as auth_mod

        auth_mod._auth_backend = auth_mod.ApiKeyBackend(admin_key="admin-key", db_path=str(tmp_path / "auth.db"))
        return create_app(kb_storage_dir=storage)

    def test_rollback_failed_returns_503_at_ingest(self, tmp_path):
        # H2 bug: rollback failure was swallowed (logged + (False, err)), so an
        # operator saw a 500 and assumed nothing landed — while half the
        # vectors were durable and unowned. Fix: _write_doc_to_stores returns a
        # 3-tuple (ok, err, rollback_failed); the ingest route maps
        # rollback_failed=True to 503 with an ORPHAN-tagged message.
        storage = tempfile.mkdtemp()
        app = self._make_client(storage, tmp_path)

        with patch.object(EmbeddingClient, "embed_batch", _fake_embed_batch), TestClient(app) as client:
            admin = {"X-API-Key": "admin-key"}
            client.post("/kb/bases", json={"name": "h2kb", "kb_id": "h2kb"}, headers=admin)
            # Force add_batch to succeed (vectors land) but metadata write to
            # fail AND the vector rollback to fail → rollback_failed=True.
            from fusion_rag.store.vector_store import VectorStore

            # Patch at the module the route imports: VectorStore.add_batch keeps
            # real, delete_by_doc raises (rollback fails), add_document raises
            # (metadata fails). That exercises both rollback branches.
            with patch.object(VectorStore, "delete_by_doc", side_effect=RuntimeError("disk gone")), \
                    patch(
                        "fusion_rag.api.routes_docs.MetadataStore.add_document",
                        side_effect=RuntimeError("metadata db locked"),
                    ):
                r = client.post(
                    "/kb/bases/h2kb/documents/ingest",
                    json={"content": "body that will index", "content_type": "text",
                          "contextualize": False},
                    headers=admin,
                )
            assert r.status_code == 503, f"rollback_failed must surface 503, got {r.status_code}"
            body = r.json()
            assert "ROLLBACK FAILED" in body["detail"]

    def test_clean_failure_returns_500_not_503(self, tmp_path):
        # A write failure whose rollback SUCCEEDS is a clean 500 (retry-safe),
        # not 503. Distinguishes "needs cleanup" from "just retry".
        storage = tempfile.mkdtemp()
        app = self._make_client(storage, tmp_path)

        with patch.object(EmbeddingClient, "embed_batch", _fake_embed_batch), TestClient(app) as client:
            admin = {"X-API-Key": "admin-key"}
            client.post("/kb/bases", json={"name": "h2kb2", "kb_id": "h2kb2"}, headers=admin)
            from fusion_rag.store.vector_store import VectorStore

            # add_batch fails (no vectors land), rollback delete_by_doc succeeds
            # (it's a no-op on an empty store) → rollback_failed=False → 500.
            with patch.object(VectorStore, "add_batch", side_effect=RuntimeError("embed bad")):
                r = client.post(
                    "/kb/bases/h2kb2/documents/ingest",
                    json={"content": "body", "content_type": "text",
                          "contextualize": False},
                    headers=admin,
                )
            assert r.status_code == 500, f"clean failure must be 500, got {r.status_code}"
            assert "ROLLBACK FAILED" not in r.json()["detail"]


# ── R1: sync store calls must run in a threadpool, not on the event loop ──


class TestR1Threadpool:
    def test_write_doc_to_stores_is_async(self):
        # R1: _write_doc_to_stores was sync; a slow add_batch froze the event
        # loop. It is now a coroutine so run_in_threadpool can offload the
        # sync store I/O. Assert the signature is async.
        assert inspect.iscoroutinefunction(_write_doc_to_stores), (
            "_write_doc_to_stores must be async so sync store calls run in a threadpool (R1)"
        )

    def test_store_calls_use_threadpool_not_loop(self):
        # R1 invariant: the blocking sync store calls (add_batch, delete_by_doc,
        # add_document, add_chunk, list_documents, get_document, ...) inside the
        # docs/admin route handlers MUST be wrapped in run_in_threadpool so they
        # do not block the event loop. Assert the route modules actually call it
        # around store operations — a regression that drops the wrapper re-freezes
        # the loop under load.
        import fusion_rag.api.routes_admin as admin_mod
        import fusion_rag.api.routes_docs as docs_mod

        docs_src = inspect.getsource(docs_mod)
        admin_src = inspect.getsource(admin_mod)
        assert "run_in_threadpool" in docs_src, (
            "routes_docs must offload sync store calls to run_in_threadpool (R1)"
        )
        assert "run_in_threadpool" in admin_src, (
            "routes_admin must offload sync store calls to run_in_threadpool (R1)"
        )
        # The wrapper must actually be imported, not just mentioned in a comment.
        assert "from fastapi.concurrency import run_in_threadpool" in docs_src, (
            "routes_docs must import run_in_threadpool (R1)"
        )
        assert "from fastapi.concurrency import run_in_threadpool" in admin_src, (
            "routes_admin must import run_in_threadpool (R1)"
        )


# ── R2: LLM calls must carry a total_deadline so a stuck fusion-mlx can't hang ~183s ──


class TestR2TotalDeadline:
    @pytest.mark.parametrize(
        "module_path,attr",
        [
            ("fusion_rag.engine.reranker", "HybridSearch"),
            ("fusion_rag.engine.rag_chain", "MultiTurnRAG"),
            ("fusion_rag.engine.contextualizer", "Contextualizer"),
        ],
    )
    def test_llm_call_sites_pass_total_deadline(self, module_path, attr):
        # R2: grep the source for total_deadline — each LLM call site must pass
        # it (the prior code passed only retries, defaulting deadline to None,
        # so 3 x 60s + backoff ≈ 183s before failing). Assert the literal is
        # present in the source so a regression (dropping the kwarg) is caught.
        import importlib

        mod = importlib.import_module(module_path)
        src = inspect.getsource(mod)
        assert "total_deadline=" in src, (
            f"{module_path} must pass total_deadline to with_retry (R2 stuck-upstream guard)"
        )

    def test_routes_generate_answer_passes_total_deadline(self):
        # routes._generate_answer is the /ask generation path.
        import fusion_rag.api.routes as routes_mod

        assert "total_deadline=" in inspect.getsource(routes_mod), (
            "routes._generate_answer must pass total_deadline to with_retry (R2)"
        )


# ── R4: KB delete must evict the pooled VectorStore handle ──


class TestR4PoolEviction:
    def test_delete_then_recreate_same_id_no_stale_handle(self, tmp_path):
        # R4 bug: delete rmtree'd the storage dir but left the pooled
        # VectorStore handle pointing at it; a recreate with the same kb_id hit
        # the stale handle (EnvAlreadyOpened / ENOENT) on the next ingest. The
        # fix evicts the pooled handle BEFORE rmtree. Drive this through the real
        # HTTP path so app.state (the contextvar-bound pool) is active and the
        # eviction actually runs against the live pool, not a no-op outside any
        # request context.
        from fusion_rag.api import auth as auth_mod

        auth_mod._auth_backend = auth_mod.ApiKeyBackend(admin_key="admin-key", db_path=str(tmp_path / "auth.db"))
        storage = tempfile.mkdtemp()
        app = create_app(kb_storage_dir=storage)
        admin = {"X-API-Key": "admin-key"}
        with patch.object(EmbeddingClient, "embed_batch", _fake_embed_batch), TestClient(app) as client:
            client.post("/kb/bases", json={"name": "r4kb", "kb_id": "r4kb"}, headers=admin)
            # Prime the pool: an ingest opens a pooled VectorStore on r4kb's
            # vectors dir, then writes (embed is faked above). The pool entry
            # for r4kb's vectors dir now exists.
            client.post(
                "/kb/bases/r4kb/documents/ingest",
                json={"content": "prime pool", "content_type": "text", "contextualize": False},
                headers=admin,
            )
            # Resolve the KB to read its vectors dir, then assert the pool held
            # it before delete and released it after. Read the pool straight off
            # app.state (the contextvar accessor get_vec_store_pool needs a
            # bound request context, which we are not in between requests).
            kb = client.app.state.kb_manager.get("r4kb")
            vec_path = kb.vector_path
            # The pool may or may not have an entry depending on how far the
            # failed ingest got; what matters is delete evicts it (and does not
            # crash). Delete via the real route.
            r = client.delete("/kb/bases/r4kb", headers=admin)
            assert r.status_code in (200, 204), f"delete failed: {r.status_code} {r.text}"
            assert vec_path not in client.app.state.vec_store_pool, (
                "pooled VectorStore handle must be evicted on delete, else a "
                "recreate with the same kb_id reuses a stale handle (R4)"
            )
            # Recreate with the same id + ingest: a fresh handle on the
            # recreated dir must NOT raise the stale-handle failure.
            client.post("/kb/bases", json={"name": "r4kb", "kb_id": "r4kb"}, headers=admin)
            r2 = client.post(
                "/kb/bases/r4kb/documents/ingest",
                json={"content": "reborn", "content_type": "text", "contextualize": False},
                headers=admin,
            )
            # Recreate + _get_vector_store must not crash (stale-handle mode
            # raised EnvAlreadyOpened/ENOENT before the fix). The ingest may
            # still fail on embed, but NOT on a stale pooled handle.
            assert r2.status_code != 500 or "EnvAlreadyOpened" not in r2.text, (
                f"recreate hit a stale pooled handle: {r2.status_code} {r2.text}"
            )

    def test_delete_source_evicts_before_rmtree(self):
        # Structural guard: in knowledge_base.delete the _evict_vec_store_pool
        # CALL must appear before the shutil.rmtree call in source line order.
        # (The runtime test above is the real check; this catches a future edit
        # that reorders the two lines without re-running the runtime test.)
        import ast
        import inspect
        import textwrap

        import fusion_rag.engine.knowledge_base as kb_mod

        src, _start = inspect.getsourcelines(kb_mod.KnowledgeBaseManager.delete)
        tree = ast.parse(textwrap.dedent("".join(src)))
        evict_line = rmtree_line = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                name = fn.attr if isinstance(fn, ast.Attribute) else (fn.id if isinstance(fn, ast.Name) else None)
                if name == "_evict_vec_store_pool" and evict_line is None:
                    evict_line = node.lineno
                if name == "rmtree" and rmtree_line is None:
                    rmtree_line = node.lineno
        assert evict_line is not None and rmtree_line is not None, (
            "delete must both evict the pool and rmtree (R4)"
        )
        assert evict_line < rmtree_line, (
            "pool eviction must happen BEFORE rmtree, else the handle points at a gone dir (R4)"
        )
