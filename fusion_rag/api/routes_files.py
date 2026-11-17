"""Issue #156: standalone attachment file service — /api/files/*.

KB-independent file objects for chat-client attachment pipelines:
upload (multipart or raw body), list, bounded byte-range segment reads,
delete, and an optional promote endpoint that runs an uploaded file through
the EXISTING KB ingest pipeline (parse -> chunk -> index) when raw segment
reads are not enough (PDF/DOCX, very large documents).

Segment reads enforce the fusion-harness file.read_segment contract
server-side: the per-call max_bytes cap (RuntimeConfig.file_segment_max_bytes,
default 256 KB) clamps oversize requests, so one call can never return a
whole large document. These routes are deliberately KB-free — no tenant
scoping, no ACL: lifecycle is the caller's own upload/delete.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

from ..engine.file_store import FileStore, is_valid_file_id
from ..engine.runtime_config import get_runtime_config
from .app_state import get_file_store
from .auth import verify_api_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/files", tags=["files"], dependencies=[Depends(verify_api_key)])


@router.post("")
async def upload_file(
    request: Request,
    filename: str = Query("", description="Filename for raw-body uploads"),
    file: UploadFile | None = File(None),
) -> dict:
    store: FileStore = get_file_store()
    if file is not None:
        data = await file.read()
        name = file.filename or filename or "attachment.bin"
        logger.info("file upload (multipart): %s (%d bytes)", name, len(data))
    else:
        data = await request.body()
        name = filename or "attachment.bin"
        logger.info("file upload (raw body): %s (%d bytes)", name, len(data))
    if not data:
        raise HTTPException(400, "empty upload")
    row = await run_in_threadpool(store.put, name, data)
    return row


@router.get("")
async def list_files() -> dict:
    store: FileStore = get_file_store()
    files = await run_in_threadpool(store.list_files)
    return {"files": files, "count": len(files)}


@router.get("/{file_id}/segments")
async def read_segment(
    file_id: str,
    offset: int = Query(0, ge=0),
    max_bytes: int = Query(0, ge=0),
) -> dict:
    store: FileStore = get_file_store()
    if not is_valid_file_id(file_id):
        raise HTTPException(404, f"file '{file_id}' not found")
    cap = get_runtime_config().file_segment_max_bytes
    # Server-side enforcement: unset/zero/negative and oversize requests all
    # CLAMP to the cap inside read_segment — never an error, never unbounded.
    try:
        segment = await run_in_threadpool(store.read_segment, file_id, offset, max_bytes, cap)
    except KeyError:
        raise HTTPException(404, f"file '{file_id}' not found")
    logger.debug(
        "segment %s offset=%d bytes=%d eof=%s", file_id, offset,
        len(segment["content"]), segment["eof"],
    )
    return segment


@router.delete("/{file_id}")
async def delete_file(file_id: str) -> dict:
    store: FileStore = get_file_store()
    if not is_valid_file_id(file_id):
        raise HTTPException(404, f"file '{file_id}' not found")
    deleted = await run_in_threadpool(store.delete, file_id)
    if not deleted:
        raise HTTPException(404, f"file '{file_id}' not found")
    return {"deleted": True, "file_id": file_id}


@router.post("/{file_id}/ingest")
async def ingest_file(file_id: str, request: Request) -> dict:
    # Promote an uploaded file into the existing KB pipeline. The blob is
    # copied to a temp file with the original extension so DocumentParser's
    # type detection (and the #84 scanned-PDF OCR path) applies — the same
    # pattern as the #90 inline-binary ingest.
    from .routes_docs import _index_document

    store: FileStore = get_file_store()
    row = store.get(file_id)
    if row is None:
        raise HTTPException(404, f"file '{file_id}' not found")
    body = await request.json()
    kb_id = str(body.get("kb_id", "")).strip()
    if not kb_id:
        raise HTTPException(400, "kb_id is required")
    contextualize = bool(body.get("contextualize", True))
    blob = store.blob_path(file_id)
    if blob is None:
        raise HTTPException(404, f"file '{file_id}' not found")
    ext = Path(row["filename"]).suffix or ".bin"
    fd, tmp_path = tempfile.mkstemp(suffix=ext, prefix="fusion_rag_file_")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(blob.read_bytes())
        result = await _index_document(kb_id, tmp_path, contextualize=contextualize)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            logger.warning("failed to clean up file-ingest temp path: %s", tmp_path)
    if "error" in result:
        raise HTTPException(result.get("status_code", 400), result["error"])
    result["file_id"] = file_id
    result["filename"] = row["filename"]
    logger.info(
        "promoted file %s (%s) into kb %s: doc=%s chunks=%s",
        file_id, row["filename"], kb_id, result.get("doc_id"), result.get("chunks"),
    )
    return result
