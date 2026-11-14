"""Issue #68 — fusion-identity integration: authoritative tenant resolution.

Replaces the forgeable X-Fusion-Tenant blind-trust (the #61 gateway path) with
authoritative tenant resolution from a fusion-identity JWT. fusion-rag calls
fusion-identity's POST /api/v1/auth/verify (service-token-gated) to resolve the
user JWT's `tid` claim, with revocation + tenant-status enforced by the identity
service. The resolved tenant feeds into the SAME `_request_tenant` contextvar
that #61/#66 scoping already reads, so KB + collection isolation is unchanged.

This module is HTTP-decoupled: fusion-rag talks to fusion-identity as a plain
HTTP service (port 11470), NOT a Python import. No `fusion_identity` dep.

Env knobs (all opt-in, default OFF — single-tenant local-first dev unaffected):
  FUSION_RAG_REQUIRE_IDENTITY=1     enable authoritative JWT resolution
  FUSION_IDENTITY_URL               identity base URL (default 127.0.0.1:11470)
  FUSION_IDENTITY_SERVICE_TOKEN     service token sent as Bearer to /verify
"""

from __future__ import annotations

import hashlib
import logging
import os
import time

logger = logging.getLogger(__name__)

_DEFAULT_IDENTITY_URL = "http://127.0.0.1:11470"
_VERIFY_TIMEOUT = 5.0
_CACHE_TTL = 15.0


def identity_enabled() -> bool:
    """FUSION_RAG_REQUIRE_IDENTITY=1/true/yes => authoritative JWT resolution on."""
    return os.environ.get("FUSION_RAG_REQUIRE_IDENTITY", "").strip().lower() in ("1", "true", "yes")


def identity_url() -> str:
    return os.environ.get("FUSION_IDENTITY_URL", _DEFAULT_IDENTITY_URL).rstrip("/")


def identity_service_token() -> str:
    return os.environ.get("FUSION_IDENTITY_SERVICE_TOKEN", "").strip()


def _bearer(raw: str | None) -> str | None:
    if not raw:
        return None
    raw = raw.strip()
    if not raw:
        return None
    low = raw.lower()
    if low.startswith("bearer "):
        return raw[7:].strip() or None
    # A bare token is also accepted (some internal clients send raw JWTs).
    return raw or None


class IdentityClient:
    """Async client for fusion-identity /verify.

    verify(token) -> claims dict (tid, role, scopes, sub, jti, tenant_status,
    revoked) on a valid active token, or None on invalid/revoked/expired token
    OR identity-unreachable (fail-closed when integration is ON). A short-TTL
    in-process cache avoids hitting identity on every request in a burst.
    """

    def __init__(self, *, url: str = "", service_token: str = "", timeout: float = _VERIFY_TIMEOUT):
        self._url = (url or identity_url()).rstrip("/")
        self._service_token = service_token or identity_service_token()
        self._timeout = timeout
        # token-hash -> (claims, expiry_epoch). Bounded by cache_ttl eviction.
        self._cache: dict[str, tuple[dict, float]] = {}

    async def verify(self, token: str) -> dict | None:
        if not token or not self._service_token:
            # No service token => cannot call identity. Fail-closed (deny) when
            # integration is ON: an operator who turned FUSION_RAG_REQUIRE_IDENTITY
            # on without a service token gets loud denials, not silent pass-through.
            logger.warning("IdentityClient.verify: missing token or service token — deny")
            return None
        key = hashlib.sha256(token.encode("utf-8")).hexdigest()
        now = time.monotonic()
        cached = self._cache.get(key)
        if cached and cached[1] > now:
            return cached[0]
        claims = await self._call_verify(token)
        if claims is not None:
            self._cache[key] = (claims, now + _CACHE_TTL)
            # Evict any expired entries opportunistically (no background task).
            self._evict(now)
        return claims

    async def _call_verify(self, token: str) -> dict | None:
        from fusion_core.http_client import get_async_client

        client = get_async_client(base_url=self._url)
        try:
            resp = await client.post(
                "/api/v1/auth/verify",
                json={"token": token},
                headers={"Authorization": f"Bearer {self._service_token}"},
                timeout=self._timeout,
            )
        except Exception as e:
            # Identity unreachable. Fail-closed when integration is ON: a down
            # identity service must not degrade to header blind-trust.
            logger.warning("IdentityClient.verify: identity unreachable (%s): %s", self._url, e)
            return None
        if resp.status_code == 200:
            try:
                data = resp.json()
            except Exception as e:
                logger.warning("IdentityClient.verify: bad JSON from identity: %s", e)
                return None
            if data.get("revoked"):
                logger.info("IdentityClient.verify: token revoked (jti=%s)", data.get("jti"))
                return None
            if data.get("tenant_status") not in (None, "", "active"):
                logger.info(
                    "IdentityClient.verify: tenant not active (tid=%s status=%s)",
                    data.get("tid"),
                    data.get("tenant_status"),
                )
                return None
            return data
        if resp.status_code == 401:
            logger.info("IdentityClient.verify: token rejected by identity (401)")
            return None
        logger.warning("IdentityClient.verify: unexpected identity status %s", resp.status_code)
        return None

    def _evict(self, now: float) -> None:
        stale = [k for k, v in self._cache.items() if v[1] <= now]
        for k in stale:
            self._cache.pop(k, None)

    def clear_cache(self) -> None:
        self._cache.clear()


async def close_identity_client(app) -> None:
    """Shutdown helper — drop the identity client off app.state (cache only).

    The pooled httpx client is owned by fusion_core.http_client and closed by
    close_all() in the lifespan; nothing to aclose here. Kept for symmetry.
    """
    client = getattr(app.state, "identity_client", None)
    if client is not None:
        client.clear_cache()
