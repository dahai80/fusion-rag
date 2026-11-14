"""Issue #133 (R11): document-level ACL metadata + caller authorization.

Permission-aware ingestion: connectors/ingest capture a per-document `acl`
dict alongside content; search resolves the AUTHORITATIVE acl from the
documents table at query time (same pattern as #86 status — a PATCH/revocation
propagates to results without re-indexing) and drops documents the caller
cannot see.

ACL shape (stored under documents.metadata["acl"]):
    {"visibility": "public" | "tenant" | "private",
     "tenant": "<owner-tenant>",      # required for visibility=tenant
     "users": ["user-a", ...]}        # required (non-empty) for private

Caller identity: tenant = the #61/#68-resolved request tenant contextvar;
user = the X-Fusion-User header (defense-in-depth tier, same trust level as
the gateway tenant header — authoritative user identity needs #68 identity to
expose a sub claim, tracked separately).

Absent acl / visibility=public => visible to everyone (backward compat:
existing docs and single-tenant local-first dev are unchanged). A caller with
NO resolved identity (tenant=None, user=None) sees only public documents —
a restricted doc is hidden, never leaked.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_VISIBILITIES = ("public", "tenant", "private")
_USER_RE = re.compile(r"^[A-Za-z0-9_\-:.@]{1,128}$")


def validate_acl(acl: dict | None) -> dict | None:
    """Validate + normalize an ingest/PATCH acl dict. Raises ValueError."""
    if acl is None:
        return None
    if not isinstance(acl, dict):
        raise ValueError("acl must be an object")
    unknown = set(acl) - {"visibility", "tenant", "users"}
    if unknown:
        raise ValueError(f"acl has unknown keys: {sorted(unknown)}")
    visibility = acl.get("visibility", "public")
    if visibility not in _VISIBILITIES:
        raise ValueError(f"acl.visibility must be one of {list(_VISIBILITIES)}, got {visibility!r}")
    if visibility == "public":
        if acl.get("tenant") or acl.get("users"):
            raise ValueError("acl.visibility=public must not carry tenant/users")
        return {"visibility": "public"}
    tenant = acl.get("tenant")
    if visibility == "tenant":
        if not tenant or not isinstance(tenant, str):
            raise ValueError("acl.visibility=tenant requires a non-empty acl.tenant string")
        return {"visibility": "tenant", "tenant": tenant}
    # private
    users = acl.get("users")
    if not isinstance(users, list) or not users or not all(
        isinstance(u, str) and u and _USER_RE.match(u) for u in users
    ):
        raise ValueError("acl.visibility=private requires acl.users: non-empty list of user ids")
    out: dict = {"visibility": "private", "users": sorted(set(users))}
    if tenant:
        if not isinstance(tenant, str):
            raise ValueError("acl.tenant must be a string")
        out["tenant"] = tenant
    return out


def acl_allows(acl: dict | None, caller_tenant: str | None, caller_user: str | None) -> bool:
    """Whether the caller identity may see a document carrying this acl.

    None/absent acl = public. An identity-less caller sees public docs only.
    """
    if not acl:
        return True
    visibility = acl.get("visibility", "public")
    if visibility == "public":
        return True
    # A document owned by the caller's tenant is always visible to that
    # tenant (owner override for private docs with a tenant stamp).
    if caller_tenant and acl.get("tenant") and caller_tenant == acl["tenant"]:
        return True
    if visibility == "tenant":
        return bool(caller_tenant) and caller_tenant == acl.get("tenant")
    # private: owner tenant (checked above) or an allow-listed user.
    return bool(caller_user) and caller_user in (acl.get("users") or [])
