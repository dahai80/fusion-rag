"""LLM credential health state — issue #93.

A stale/rejected FUSION_MLX_API_KEY used to surface only as a generic
"upstream LLM unavailable" 502 on /ask, indistinguishable from a real outage.
This module tracks the credential status (ok / rejected / unknown) in process
memory so /health can expose it and the startup probe can flag it loudly.
Never stores or logs the key itself — only a short blake2b hash prefix.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_AUTH_STATUS_UNKNOWN = "unknown"
_AUTH_STATUS_OK = "ok"
_AUTH_STATUS_REJECTED = "rejected"

_llm_auth_status: str = _AUTH_STATUS_UNKNOWN


def get_llm_auth_status() -> str:
    return _llm_auth_status


def set_llm_auth_status(status: str) -> None:
    # "ok" | "rejected" | "unknown". A rejected key that later starts working
    # (rotation without restart) flips back to ok on the next successful call.
    global _llm_auth_status
    if status not in (_AUTH_STATUS_OK, _AUTH_STATUS_REJECTED, _AUTH_STATUS_UNKNOWN):
        logger.warning("set_llm_auth_status: unknown status %r ignored", status)
        return
    _llm_auth_status = status


def key_hash_prefix(api_key: str) -> str:
    # Short non-reversible prefix for logs only — identifies WHICH key was
    # rejected without ever logging the key material itself.
    import hashlib

    if not api_key:
        return "(none)"
    return hashlib.blake2b(api_key.encode("utf-8"), digest_size=4).hexdigest()


def probe_llm_auth_sync(base_url: str, api_key: str) -> str:
    # Cheap credential sanity check: GET /models with the configured key.
    # Returns + records "ok" (200), "rejected" (401/403), or "unknown"
    # (unreachable / other status — do NOT claim rejection on a network error;
    # a down gateway is an outage, not a credential problem).
    import httpx

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    url = base_url.rstrip("/")
    if not url.endswith("/v1"):
        url = f"{url}/v1"
    try:
        resp = httpx.get(f"{url}/models", headers=headers, timeout=10.0)
    except Exception as e:
        logger.warning("LLM auth probe: %s unreachable (%s)", url, e)
        set_llm_auth_status(_AUTH_STATUS_UNKNOWN)
        return _AUTH_STATUS_UNKNOWN
    if resp.status_code in (401, 403):
        set_llm_auth_status(_AUTH_STATUS_REJECTED)
        return _AUTH_STATUS_REJECTED
    if resp.status_code == 200:
        set_llm_auth_status(_AUTH_STATUS_OK)
        return _AUTH_STATUS_OK
    logger.warning("LLM auth probe: unexpected status %s from %s", resp.status_code, url)
    set_llm_auth_status(_AUTH_STATUS_UNKNOWN)
    return _AUTH_STATUS_UNKNOWN
