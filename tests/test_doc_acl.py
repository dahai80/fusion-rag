"""Issue #133 (R11): document ACL validation, authorization matrix, search filter."""

from __future__ import annotations

import pytest

from fusion_rag.api.tenant import _request_tenant, _request_user, reset_request_tenant
from fusion_rag.engine.doc_acl import acl_allows, validate_acl


@pytest.fixture(autouse=True)
def _clear_identity():
    reset_request_tenant()
    yield
    reset_request_tenant()


def test_validate_public_and_none():
    assert validate_acl(None) is None
    assert validate_acl({}) == {"visibility": "public"}
    assert validate_acl({"visibility": "public"}) == {"visibility": "public"}


def test_validate_public_rejects_scoping():
    with pytest.raises(ValueError, match="must not carry"):
        validate_acl({"visibility": "public", "users": ["a"]})


def test_validate_tenant():
    assert validate_acl({"visibility": "tenant", "tenant": "acme"}) == {
        "visibility": "tenant",
        "tenant": "acme",
    }
    with pytest.raises(ValueError, match=r"requires a non-empty acl\.tenant"):
        validate_acl({"visibility": "tenant"})


def test_validate_private():
    acl = validate_acl({"visibility": "private", "users": ["b", "a", "a"]})
    assert acl == {"visibility": "private", "users": ["a", "b"]}
    with pytest.raises(ValueError, match="non-empty list"):
        validate_acl({"visibility": "private", "users": []})
    with pytest.raises(ValueError, match="unknown keys"):
        validate_acl({"visibility": "private", "users": ["a"], "role": "admin"})


def test_acl_allows_matrix():
    tenant_acl = {"visibility": "tenant", "tenant": "acme"}
    private_acl = {"visibility": "private", "users": ["alice"], "tenant": "acme"}
    # public / absent => everyone
    assert acl_allows(None, None, None)
    assert acl_allows({"visibility": "public"}, None, None)
    # tenant visibility
    assert acl_allows(tenant_acl, "acme", None)
    assert not acl_allows(tenant_acl, "other", "alice")
    assert not acl_allows(tenant_acl, None, "alice")
    # private visibility
    assert acl_allows(private_acl, "acme", None)
    assert acl_allows(private_acl, None, "alice")
    assert acl_allows(private_acl, "acme", "alice")
    assert not acl_allows(private_acl, "other", "bob")
    # identity-less caller sees public only
    assert not acl_allows(private_acl, None, None)


def test_public_doc_untouched_without_acl(tmp_path):
    from fusion_rag.store.metadata_store import MetadataStore

    db = tmp_path / "meta.db"
    store = MetadataStore(str(db))
    store.add_document("d1", "/x.md", "x.md", "markdown", metadata={})
    assert store.get_document_acl_map(["d1"]) == {"d1": None}


def test_acl_roundtrip_and_revocation(tmp_path):
    from fusion_rag.store.metadata_store import MetadataStore

    db = tmp_path / "meta.db"
    store = MetadataStore(str(db))
    store.add_document("d1", "/x.md", "x.md", "markdown", metadata={"acl": {"visibility": "tenant", "tenant": "acme"}})
    assert store.get_document_acl_map(["d1"])["d1"] == {"visibility": "tenant", "tenant": "acme"}
    # revoke without reindex: merge-patch to public
    store.update_document_metadata("d1", {"acl": {"visibility": "public"}})
    assert store.get_document_acl_map(["d1"])["d1"] == {"visibility": "public"}
    assert acl_allows(store.get_document_acl_map(["d1"])["d1"], None, None)


def test_contextvars_user_capture():
    _request_tenant.set("acme")
    _request_user.set("alice")
    from fusion_rag.api.tenant import get_request_tenant, get_request_user

    assert get_request_tenant() == "acme"
    assert get_request_user() == "alice"
    reset_request_tenant()
    assert get_request_user() is None
