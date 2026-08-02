"""API key management routes — expose AuthConfig via HTTP endpoints."""

from __future__ import annotations

import logging
import secrets
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from .auth import AuthConfig, verify_api_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/kb/auth", tags=["auth"])


@router.get("/keys")
async def list_api_keys(_: str | None = Depends(verify_api_key)) -> dict[str, Any]:
    keys = AuthConfig().list_keys()
    return {"keys": keys}


@router.post("/keys")
async def create_api_key(
    body: dict[str, Any],
    _: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    name = body.get("name", "default")
    key = f"frg_{secrets.token_hex(16)}"
    auth = AuthConfig()
    if auth.add_key(key, name):
        logger.info("API key created: name=%s", name)
        return {"key": key, "name": name, "created_at": time.time()}
    raise HTTPException(500, "Failed to create API key")


@router.delete("/keys/{key_hash_prefix}")
async def delete_api_key(
    key_hash_prefix: str,
    _: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
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
