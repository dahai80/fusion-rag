"""P0-5/P0-6: KB-level access control dependency.

Wires the existing permission module (ACL + PermissionManager) into the REST
surface — before this, permission rules were CRUD-able but never enforced,
which is worse than no permissions: admins believed access was restricted when
it was not.

Semantics:
- NoAuth (FUSION_RAG_AUTH_BACKEND=none or no admin key): allow. Local-first
  default — a single-user offline box must not start 401-ing.
- Authenticated subject == "admin": allow any action (ACL bypass).
- KB has NO permission rules: allow. An open KB keeps working; enforcement
  only engages once an admin has actually written a rule for that KB. This is
  the backward-compat path — existing KBs are not silently locked out.
- KB HAS rules for the subject: fail-closed. PermissionManager.check_permission
  decides; a denied check is 403, a backend error is 403 (never allow-on-error).

This dependency runs the same kb_id validation + KB lookup as _get_base, so
endpoints can keep their own _get_base call (it is a cheap dict get) OR rely
on the dependency's resolution. To keep the diff surgical, endpoints keep their
existing _get_base; the dependency resolves independently and its 404/400
fires before the handler body runs.
"""

from __future__ import annotations

import logging

from fastapi import Depends, HTTPException

from .._validators import validate_identifier
from .app_state import get_kb_manager
from .auth import verify_api_key

logger = logging.getLogger(__name__)


def _resolve_base(kb_id: str):
    # Mirror of each router's _get_base: confine kb_id then fetch. Duplicated
    # deliberately (Rule 3) so access.py has no import edge into any router
    # module — routers import access.py, never the reverse.
    try:
        validate_identifier(kb_id, field="kb_id")
    except ValueError:
        raise HTTPException(400, f"Invalid kb_id: {kb_id}")
    try:
        return get_kb_manager().get(kb_id)
    except KeyError:
        raise HTTPException(404, f"Knowledge base '{kb_id}' not found")


def _check_kb_access(kb_id: str, subject: str | None, action: str, resource_path: str = "/") -> None:
    # NoAuth -> subject is None -> open (local-first default).
    if subject is None:
        return
    # Admin key authenticates as "admin" (auth.py verify) -> ACL bypass.
    if subject == "admin":
        return
    kb = _resolve_base(kb_id)
    storage_path = kb.vector_path.rsplit("/vectors", 1)[0] if "/vectors" in kb.vector_path else kb.vector_path
    from ..permissions import PermissionManager

    pm = PermissionManager(f"{storage_path}/permissions.db")
    # Open-KB path: if no rules exist for this KB at all, treat it as open.
    # Enforcement only engages once a rule is written (backward compat).
    try:
        all_rules = pm.list_rules(kb_id)
    except Exception as e:
        # Fail-closed on a permission-store read error — never allow-on-error.
        logger.error("permission store read failed for kb=%s, denying (fail-closed): %s", kb_id, e)
        raise HTTPException(403, "Access denied (permission store unavailable)")
    if not all_rules:
        logger.debug("access: no permission rules for kb=%s -> open KB, allow", kb_id)
        return
    # H4 fix: pass the REAL resource_path, not a hardcoded "/". The prior code
    # always checked "/" — and every path-prefixed rule matches "/" (rule root
    # is a prefix of "/"), so path-level ACL was never enforced: a subject
    # restricted to "/docs/restricted/*" could read "/docs/public/*" too.
    # KB-scoped endpoints pass "/" (the KB is the resource); doc-specific
    # endpoints resolve doc_id -> file_path so a rule scoped to a subtree only
    # permits that subtree.
    try:
        allowed = pm.check_permission(kb_id, subject, action, resource_path)
    except Exception as e:
        logger.error("permission check failed for kb=%s subject=%s, denying: %s", kb_id, subject, e)
        raise HTTPException(403, "Access denied")
    if not allowed:
        logger.warning("access denied: kb=%s subject=%s action=%s path=%s", kb_id, subject, action, resource_path)
        raise HTTPException(403, "Access denied")
    logger.debug("access granted: kb=%s subject=%s action=%s path=%s", kb_id, subject, action, resource_path)


def require_kb_action(action: str, resolve_doc_path: bool = False):
    """Build a FastAPI dependency enforcing `action` on the path's kb_id.

    Usage: `dependencies=[Depends(require_kb_action("read"))]` on a route that
    has a {kb_id} path param, or `subject = Depends(require_kb_action("read"))`
    as a handler param to also receive the subject string.

    resolve_doc_path: for doc-specific routes (/bases/{kb_id}/documents/{doc_id})
    set True — the dependency resolves doc_id -> the doc's stored file_path and
    passes it as the ACL resource_path, so a rule scoped to a path subtree only
    permits that subtree. KB-scoped routes leave it False (resource = "/" = KB).
    """

    def _dep(
        kb_id: str,
        doc_id: str | None = None,
        subject: str | None = Depends(verify_api_key),
    ) -> str | None:
        resource_path = "/"
        if resolve_doc_path and doc_id:
            resource_path = _resolve_doc_resource_path(kb_id, doc_id)
        _check_kb_access(kb_id, subject, action, resource_path)
        return subject

    return _dep


def _resolve_doc_resource_path(kb_id: str, doc_id: str) -> str:
    # H4: map a path-param doc_id to the stored file_path so ACL rules scoped to
    # a path subtree can actually enforce at the doc level. If the doc is absent
    # (404) or the store read errors, fall back to "/" — the handler will 404
    # on the missing doc anyway, and failing open to "/" here would let a
    # subtree-restricted subject delete a doc outside its subtree. Instead, on
    # a store error we deny (fail-closed) by returning a path unlikely to match
    # any rule, so ACL.check's no-match -> deny fires. A missing doc returns "/"
    # because the subsequent handler 404s regardless.
    try:
        kb = _resolve_base(kb_id)
    except HTTPException:
        return "/"
    storage_path = kb.vector_path.rsplit("/vectors", 1)[0] if "/vectors" in kb.vector_path else kb.vector_path
    try:
        from ..store.metadata_store import MetadataStore

        ms = MetadataStore(f"{storage_path}/metadata.db")
        doc = ms.get_document(doc_id)
        if not doc:
            return "/"
        return doc.get("file_path", "/") or "/"
    except Exception as e:
        logger.warning("access: doc resource_path resolve failed for kb=%s doc=%s: %s", kb_id, doc_id, e)
        return "/__acl_unresolved__"
