"""Issue #61 — gateway-origin + tenant scoping for multi-tenant isolation.

Backend-side enforcement matching fusion-gateway #150 Gap 1c. The gateway
derives an authoritative tenant from the api_key's key->team binding and
stamps it as X-Fusion-Tenant on every outbound request, plus X-Fusion-Route:
gateway-decision as the origin signal. fusion-rag, reached by the gateway on
its direct port (:11436), enforces the matching backend half:

1. Require X-Fusion-Route: gateway-decision on /kb/* when tenant isolation is
   enabled (FUSION_RAG_REQUIRE_GATEWAY=1). A request missing the header is
   rejected 403, so direct-port access cannot bypass the gateway's tenant
   derivation. Default OFF — single-tenant local-first dev keeps working.
2. Honor X-Fusion-Tenant as the authoritative tenant for the request. KB
   list/get are scoped to this tenant; a tenant-A caller never sees tenant-B's
   KBs. The client-supplied X-Space-Id is a non-authoritative passthrough and
   is ignored for scoping decisions.

Tenant scoping is the FIRST defense (list/get hide other tenants' KBs); the
existing per-KB ACL (access.py) is the second (sub-tenant path rules). When
tenant isolation is OFF (default), tenant is None and no filtering happens —
zero behavior change for existing single-tenant deployments.

callers: server.py (middleware), routes_kb.py + _get_base helpers (tenant
read via get_request_tenant), knowledge_base.py KnowledgeBaseManager (tenant
filter param)
"""

from __future__ import annotations

import logging
import os
import re
from contextvars import ContextVar

from fastapi import Request
from fastapi.responses import JSONResponse

from .identity import identity_enabled

logger = logging.getLogger(__name__)


def _extract_bearer(raw: str | None) -> str | None:
    """Pull a bare JWT out of an `Authorization: Bearer <token>` header."""
    if not raw:
        return None
    raw = raw.strip()
    if not raw:
        return None
    low = raw.lower()
    if low.startswith("bearer "):
        return raw[7:].strip() or None
    return raw or None

# Authoritative tenant for the current request. None = no tenant isolation
# (single-tenant local dev, or the gateway header was absent and isolation is
# off). Set by tenant_middleware; read by get_request_tenant.
_request_tenant: ContextVar[str | None] = ContextVar("_request_tenant", default=None)

# The gateway origin-signal header. Present => the request transited the
# gateway (which derived + stamped the tenant). Absent => direct-port access.
ROUTE_HEADER = "X-Fusion-Route"
GATEWAY_ROUTE_VALUE = "gateway-decision"

# Authoritative tenant header (gateway-derived from key->team binding).
TENANT_HEADER = "X-Fusion-Tenant"

# Non-authoritative passthrough header — ignored for scoping (documented only).
SPACE_ID_HEADER = "X-Space-Id"

# Tenant id charset: same conservative identifier set as kb_id. A spoofed or
# malformed tenant must not reach the KB layer (it is used as a storage
# scoping key, so path separators / traversal chars are forbidden).
_TENANT_RE = re.compile(r"^[A-Za-z0-9_\-:.]{1,128}$")


def require_gateway_enabled() -> bool:
    """FUSION_RAG_REQUIRE_GATEWAY=1/true/yes => reject non-gateway requests."""
    return os.environ.get("FUSION_RAG_REQUIRE_GATEWAY", "").strip().lower() in ("1", "true", "yes")


def _normalize_tenant(raw: str | None) -> str | None:
    """Validate + return the authoritative tenant, or None if absent/invalid.

    An invalid tenant (bad charset, overlong) is treated as absent + logged at
    WARNING — we do NOT 403 on a malformed tenant alone, because a legitimate
    gateway always sends a valid one and a direct caller has no tenant at all.
    When require-gateway is ON, a missing tenant is caught by the route-header
    check (the gateway sends both); when OFF, no tenant means no filtering.
    """
    if not raw:
        return None
    raw = raw.strip()
    if not raw:
        return None
    if not _TENANT_RE.match(raw):
        logger.warning("tenant header value rejected (invalid charset): %r", raw[:32])
        return None
    return raw


def _is_exempt(path: str) -> bool:
    """Paths that must never be gated by tenant resolution.

    /health, /ready (liveness/readiness), /metrics (scrape), /mcp (MCP), and the
    /v1 + /auth login/token surfaces (they mint the tokens, cannot require one),
    plus the dynamic /store/* M2M surface (node-to-node via X-API-Key, no user
    JWT). A down gateway/identity must not make the service appear down to its
    own orchestrator.
    """
    if path in ("/health", "/ready", "/metrics", "/mcp"):
        return True
    if path.startswith("/v1") or path.startswith("/auth"):
        return True
    return "/store/" in path


async def tenant_middleware(request: Request, call_next):
    """Bind the authoritative tenant + enforce gateway/identity origin on /kb/*.

    Two tenant-resolution modes:

    1. Identity mode (FUSION_RAG_REQUIRE_IDENTITY=1, issue #68): the tenant is
       resolved AUTHORITATIVELY from a fusion-identity JWT. The caller's
       `Authorization: Bearer <jwt>` is verified via the identity service
       (revocation + tenant-status enforced there); the JWT `tid` is the
       authoritative tenant. X-Fusion-Tenant is DEMOTED to defense-in-depth:
       if present it MUST equal the JWT `tid` (mismatch => 401, logged as a
       header-forgery attempt). A request with no/invalid/revoked JWT is 401.
       This retires the blind-trust path — the header can no longer override
       the JWT. The resolved tenant feeds the SAME `_request_tenant` contextvar
       so #61/#66 scoping is unchanged.

    2. Gateway mode (FUSION_RAG_REQUIRE_GATEWAY=1, issue #61, default path): the
       tenant is the gateway-stamped X-Fusion-Tenant (the gateway derived it
       from the api_key's key->team binding). /kb/* without X-Fusion-Route:
       gateway-decision is rejected 403. This is now the defense-in-depth tier
       beneath identity; when identity is ON it is skipped (identity is the
       primary authority).

    Both modes OFF (default) => single-tenant local-first dev, zero change.

    Exempt paths (no resolution): /health, /ready, /metrics, /mcp, /v1, /auth,
    and the dynamic /store/* M2M surface.
    """
    path = request.url.path

    # --- Identity mode (#68): authoritative JWT resolution -----------------
    if identity_enabled() and path.startswith("/kb/") and not _is_exempt(path):
        client = getattr(request.app.state, "identity_client", None)
        if client is None:
            # Integration on but no client wired — operator misconfig. Fail-closed.
            logger.error("identity mode on but identity_client not wired on app.state — 401")
            return JSONResponse(
                status_code=401,
                content={"detail": "Identity resolution unavailable (service misconfigured)"},
            )
        jwt = _extract_bearer(request.headers.get("Authorization"))
        if not jwt:
            logger.info("identity mode: /kb%s request without Bearer JWT — 401", path)
            return JSONResponse(
                status_code=401,
                content={"detail": "Bearer JWT required for KB access"},
            )
        claims = await client.verify(jwt)
        if claims is None:
            logger.info("identity mode: /kb%s JWT rejected/revoked or identity unreachable — 401", path)
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or revoked token"},
            )
        jwt_tenant = _normalize_tenant(claims.get("tid"))
        if jwt_tenant is None:
            logger.warning("identity mode: JWT resolved but tid missing/invalid — 401")
            return JSONResponse(
                status_code=401,
                content={"detail": "Token has no valid tenant"},
            )
        # Defense-in-depth: X-Fusion-Tenant, if present, must agree with the JWT.
        header_tenant = _normalize_tenant(request.headers.get(TENANT_HEADER))
        if header_tenant is not None and header_tenant != jwt_tenant:
            logger.warning(
                "identity mode: X-Fusion-Tenant=%s != JWT tid=%s — forgery attempt, 401",
                header_tenant,
                jwt_tenant,
            )
            return JSONResponse(
                status_code=401,
                content={"detail": "Tenant header does not match token"},
            )
        _request_tenant.set(jwt_tenant)
        logger.debug("identity mode: resolved tenant=%s for /kb%s", jwt_tenant, path)
        return await call_next(request)

    # --- Gateway mode (#61, unchanged when identity off) -------------------
    tenant = _normalize_tenant(request.headers.get(TENANT_HEADER))
    _request_tenant.set(tenant)
    # Gateway-origin enforcement only applies to the KB surface (/kb/*). The
    # health/readiness/metrics/MCP/auth routes are intentionally exempt so the
    # service stays observable + manageable when the gateway is down.
    if (
        require_gateway_enabled()
        and path.startswith("/kb/")
        and "/store/" not in path
        and request.headers.get(ROUTE_HEADER, "") != GATEWAY_ROUTE_VALUE
    ):
        # The /store/* surface is M2M: another fusion-rag node acting as
        # RemoteBackend authenticates via X-API-Key and carries no gateway
        # headers. Exempt it from the gateway-origin gate so a correctly
        # authenticated node call is not 403'd; the router still enforces
        # verify_api_key on every /store endpoint.
        logger.warning(
            "reject non-gateway /kb%s request: missing/invalid %s — 403",
            path,
            ROUTE_HEADER,
        )
        return JSONResponse(
            status_code=403,
            content={"detail": "Gateway-origin required for KB access"},
        )
    return await call_next(request)


def get_request_tenant() -> str | None:
    """The authoritative tenant for the current request, or None.

    None means tenant isolation is not in effect for this request (default,
    single-tenant dev). KB list/get pass this to KnowledgeBaseManager, which
    filters by tenant when it is not None and leaves the set unfiltered when
    it is None — so the default path is zero-change.
    """
    return _request_tenant.get()


def tenant_scope() -> tuple[str | None, bool]:
    """Return (tenant, require_tenant_match) for KB list/get scoping.

    Scoping engages when EITHER isolation mode is on:
    - Identity mode (#68, FUSION_RAG_REQUIRE_IDENTITY=1): the tenant was resolved
      from the JWT `tid` by the middleware; every /kb/* request that reached a
      route handler has a valid JWT-resolved tenant (else it was 401'd upstream).
    - Gateway mode (#61, FUSION_RAG_REQUIRE_GATEWAY=1): the tenant is the
      gateway-stamped X-Fusion-Tenant; a direct-port caller was 403'd upstream.

    Otherwise (both off, default) (None, False) — no filtering, zero change for
    single-tenant dev.
    """
    if not (identity_enabled() or require_gateway_enabled()):
        return None, False
    tenant = get_request_tenant()
    if tenant is None:
        return None, False
    return tenant, True


def reset_request_tenant() -> None:
    """Test helper — clear the contextvar between test cases."""
    _request_tenant.set(None)
