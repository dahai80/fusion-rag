"""API key management routes — expose AuthConfig via HTTP endpoints."""

from __future__ import annotations

import logging
import os
import secrets
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from .access import require_admin
from .auth import AuthConfig

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/kb/auth", tags=["auth"])


def _scaled_mode() -> bool:
    # Issue #102 (DB conventions DB-5): in gateway/identity mode the deployment
    # is horizontally scaled (stateless LB, issue #61/#68 topology), and the
    # per-instance SQLite auth.db cannot serve as a shared key store — a key
    # provisioned on node A 401s on node B. Scaled deployments must use
    # config-plane credentials only (shared FUSION_RAG_API_KEY env /
    # gateway-issued / identity JWT); local key creation is refused.
    for var in ("FUSION_RAG_REQUIRE_GATEWAY", "FUSION_RAG_REQUIRE_IDENTITY"):
        if os.environ.get(var, "").strip().lower() in ("1", "true", "yes"):
            return True
    return False


@router.get("/keys", dependencies=[Depends(require_admin())])
async def list_api_keys() -> dict[str, Any]:
    keys = AuthConfig().list_keys()
    return {"keys": keys}


@router.post("/keys", dependencies=[Depends(require_admin())])
async def create_api_key(body: dict[str, Any]) -> dict[str, Any]:
    if _scaled_mode():
        # Issue #102 option 1: per-instance auth.db keys are unusable behind a
        # stateless LB — refuse loudly instead of minting a key that works on
        # one node and 401s on the rest.
        logger.warning("local API key creation refused: scaled mode (gateway/identity) is on")
        raise HTTPException(
            409,
            "Local API key creation is disabled in scaled mode (FUSION_RAG_REQUIRE_GATEWAY/"
            "FUSION_RAG_REQUIRE_IDENTITY): per-instance auth.db does not survive horizontal "
            "scaling. Use the shared FUSION_RAG_API_KEY env or gateway/identity-issued credentials.",
        )
    name = body.get("name", "default")
    # S-P0-1: belt-and-suspenders — require_admin blocks non-admin callers, and
    # this rejects a reserved name even if an admin mistypes one. add_key also
    # rejects it, but a clear 400 here beats a silent 500 from add_key.
    if name in ("admin",):
        raise HTTPException(400, "name 'admin' is reserved for the env admin key")
    key = f"frg_{secrets.token_hex(16)}"
    auth = AuthConfig()
    if auth.add_key(key, name):
        logger.info("API key created: name=%s", name)
        return {"key": key, "name": name, "created_at": time.time()}
    raise HTTPException(500, "Failed to create API key")


@router.delete("/keys/{key_hash_prefix}", dependencies=[Depends(require_admin())])
async def delete_api_key(key_hash_prefix: str) -> dict[str, Any]:
    keys = AuthConfig().list_keys()
    for k in keys:
        if k["key_hash"].startswith(key_hash_prefix):
            full_hash = k["key_hash"].rstrip("...")
            auth = AuthConfig()
            conn = auth._get_conn()
            try:
                cursor = conn.execute(
                    "DELETE FROM api_keys WHERE key_hash LIKE ?",
                    (full_hash + "%",),
                )
                conn.commit()
                if cursor.rowcount > 0:
                    logger.info("API key deleted: hash_prefix=%s", key_hash_prefix)
                    return {"deleted": True}
            finally:
                conn.close()
    raise HTTPException(404, f"Key with prefix {key_hash_prefix} not found")
