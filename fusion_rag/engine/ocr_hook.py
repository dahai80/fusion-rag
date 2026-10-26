"""Issue #84/#85: OcrHook — async client for fusion-drawing-parser (scanned-PDF OCR).

fusion-rag never imports OCR models directly. Scanned PDFs (pages present, zero
extractable text) are handed to the fusion-drawing-parser service over HTTP:
POST /v1/parse {file_path} -> {task_id}; poll GET /v1/tasks/{id} until a
terminal status; GET /v1/tasks/{id}/result -> schema-v1 result whose
text_chunks[] carry text + page + bbox + slice_id + ocr_confidence. Those
chunks feed the ingest pipeline with per-chunk OCR evidence metadata (#85).
"""
from __future__ import annotations

import asyncio
import logging
import os
import re

from fusion_core.http_client import get_async_client

logger = logging.getLogger(__name__)

TERMINAL_DONE = "done"
TERMINAL_FAILED = {"failed", "cancelled"}
DEFAULT_TIMEOUT = 300.0
POLL_INTERVAL = 1.0

# Issue #88: drawing-parser (fusion_core.tenant middleware) requires X-Tenant-Id
# on every call — missing header is a 401. Same conservative charset as the
# fusion-rag tenant id: a bad value must not reach the downstream service.
TENANT_ID_HEADER = "X-Tenant-Id"
_TENANT_RE = re.compile(r"^[A-Za-z0-9_\-:.]{1,128}$")


def _resolve_tenant_id() -> str | None:
    # 1. The tenant of the request that triggered the ingest (set by
    #    tenant_middleware, read from the contextvar) — the multi-tenant path.
    # 2. FUSION_RAG_OCR_TENANT_ID env fallback — for background ingestion
    #    (directory watch loop, no request context) and single-tenant
    #    deployments where no gateway/identity resolution is on.
    from fusion_rag.api.tenant import get_request_tenant

    tenant = get_request_tenant()
    if not tenant:
        tenant = os.environ.get("FUSION_RAG_OCR_TENANT_ID", "").strip() or None
    if tenant and not _TENANT_RE.match(tenant):
        logger.warning("OCR tenant id rejected (invalid charset): %r", tenant[:32])
        return None
    return tenant


class OcrUnavailable(Exception):
    # Raised when no OCR service is configured or the service fails a task.
    pass


def ocr_service_configured() -> bool:
    return bool(os.environ.get("FUSION_RAG_OCR_URL", "").strip())


class OcrHook:
    # Env-driven client: FUSION_RAG_OCR_URL (base, e.g. http://127.0.0.1:11491),
    # FUSION_RAG_OCR_API_KEY (Bearer token, may be empty), FUSION_RAG_OCR_TIMEOUT
    # (total poll budget in seconds, default 300).
    def __init__(self) -> None:
        self.base_url = os.environ.get("FUSION_RAG_OCR_URL", "").strip().rstrip("/")
        self.api_key = os.environ.get("FUSION_RAG_OCR_API_KEY", "").strip()
        try:
            self.timeout = float(os.environ.get("FUSION_RAG_OCR_TIMEOUT", "300").strip() or 300)
        except ValueError:
            logger.warning("invalid FUSION_RAG_OCR_TIMEOUT, using default %s", DEFAULT_TIMEOUT)
            self.timeout = DEFAULT_TIMEOUT

    def _headers(self) -> dict:
        headers: dict = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        # Issue #88: pass the calling request's tenant through — drawing-parser
        # rejects header-less calls with 401 in multi-tenant deployments.
        tenant = _resolve_tenant_id()
        if tenant:
            headers[TENANT_ID_HEADER] = tenant
        return headers

    def _client(self):
        return get_async_client(self.base_url, timeout=30.0)

    async def parse_file(self, file_path: str) -> list[dict]:
        # Submit -> poll -> fetch result. Returns text_chunks (schema v1).
        if not self.base_url:
            raise OcrUnavailable(
                "scanned PDF detected but no OCR service configured (set FUSION_RAG_OCR_URL)"
            )
        client = self._client()
        resp = await client.post(
            "/v1/parse", json={"file_path": file_path}, headers=self._headers()
        )
        if resp.status_code != 200:
            raise OcrUnavailable(
                f"OCR service rejected parse request: {resp.status_code} {resp.text[:300]}"
            )
        task_id = resp.json().get("task_id")
        if not task_id:
            raise OcrUnavailable(f"OCR service returned no task_id: {resp.text[:300]}")
        logger.info("OCR task submitted task_id=%s file=%s", task_id, file_path)

        waited = 0.0
        while True:
            task = await client.get(f"/v1/tasks/{task_id}", headers=self._headers())
            if task.status_code != 200:
                raise OcrUnavailable(f"OCR task status fetch failed: {task.status_code} {task.text[:300]}")
            status = task.json().get("status")
            if status == TERMINAL_DONE:
                break
            if status in TERMINAL_FAILED:
                err = task.json().get("error") or status
                raise OcrUnavailable(f"OCR task {task_id} failed: {err}")
            if waited >= self.timeout:
                raise OcrUnavailable(f"OCR task {task_id} timed out after {self.timeout:.0f}s (status={status})")
            await asyncio.sleep(POLL_INTERVAL)
            waited += POLL_INTERVAL

        result = await client.get(f"/v1/tasks/{task_id}/result", headers=self._headers())
        if result.status_code != 200:
            raise OcrUnavailable(f"OCR result fetch failed: {result.status_code} {result.text[:300]}")
        data = result.json()
        chunks = data.get("text_chunks") or []
        # Stamp task_id onto every chunk so ingest can build per-slice evidence
        # URLs ({ocr_base}/v1/tasks/{task_id}/slices/{slice_id}, issue #85).
        for c in chunks:
            c["task_id"] = task_id
        stats = data.get("stats") or {}
        logger.info(
            "OCR task done task_id=%s chunks=%d pages=%s low_confidence=%s",
            task_id, len(chunks), stats.get("pages"), stats.get("low_confidence"),
        )
        return chunks
