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

logger = logging.getLogger(__name__)

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


async def tenant_middleware(request: Request, call_next):
    """Bind the authoritative tenant + enforce gateway-origin on /kb/*.

    Registered as HTTP middleware. Reads X-Fusion-Tenant (authoritative) and
    X-Fusion-Route (origin signal). When FUSION_RAG_REQUIRE_GATEWAY is on, a
    /kb/* request without X-Fusion-Route: gateway-decision is rejected 403 —
    direct-port access cannot bypass the gateway. /health, /ready, /metrics,
    /mcp and /v1 auth routes are exempt (liveness/readiness/scrape/MCP must
    not be gated behind the gateway, or a down gateway makes the service
    appear down to its own orchestrator).
    """
    path = request.url.path
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

    Scoping engages only when FUSION_RAG_REQUIRE_GATEWAY is on (isolation mode)
    AND the request carried an authoritative X-Fusion-Tenant. Otherwise
    (None, False) — no filtering, zero behavior change for single-tenant dev.
    In isolation mode every /kb/* request that passed the gateway-origin gate
    has a tenant; a direct-port caller was already 403'd by the middleware, so
    reaching here with a tenant means the gateway derived it authoritatively.
    """
    if not require_gateway_enabled():
        return None, False
    tenant = get_request_tenant()
    if tenant is None:
        return None, False
    return tenant, True


def reset_request_tenant() -> None:
    """Test helper — clear the contextvar between test cases."""
    _request_tenant.set(None)
