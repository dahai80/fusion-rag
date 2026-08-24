from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from .._validators import validate_identifier
from .app_state import get_kb_manager
from .auth import verify_api_key

logger = logging.getLogger(__name__)

router = APIRouter(tags=["admin"])

# 硬伤1: read kb_manager from app.state via contextvar, not a module global.
_get_kb_manager = get_kb_manager


def _get_base(kb_id: str):
    # F12: confine kb_id before path construction.
    try:
        validate_identifier(kb_id, field="kb_id")
    except ValueError:
        raise HTTPException(400, f"Invalid kb_id: {kb_id}")
    try:
        return _get_kb_manager().get(kb_id)
    except KeyError:
        raise HTTPException(404, f"Knowledge base '{kb_id}' not found")


def _get_kb_storage_path(kb_id: str) -> str:
    kb = _get_base(kb_id)
    return kb.vector_path.rsplit("/vectors", 1)[0] if "/vectors" in kb.vector_path else kb.vector_path


# ── Version Snapshots ──


@router.post("/bases/{kb_id}/versions", dependencies=[Depends(verify_api_key)])
async def create_snapshot(kb_id: str, data: dict[str, Any]) -> dict[str, Any]:
    _get_base(kb_id)
    from ..engine.version_manager import VersionManager

    vm = VersionManager(f"{_get_kb_storage_path(kb_id)}/versions.db")
    description = data.get("description", "")
    return vm.create_snapshot(kb_id, _get_kb_storage_path(kb_id), description)


@router.get("/bases/{kb_id}/versions", dependencies=[Depends(verify_api_key)])
async def list_snapshots(kb_id: str) -> list[dict[str, Any]]:
    _get_base(kb_id)
    from ..engine.version_manager import VersionManager

    vm = VersionManager(f"{_get_kb_storage_path(kb_id)}/versions.db")
    return vm.list_snapshots(kb_id)


@router.get("/bases/{kb_id}/versions/{version_id}", dependencies=[Depends(verify_api_key)])
async def get_snapshot(kb_id: str, version_id: str) -> dict[str, Any]:
    _get_base(kb_id)
    from ..engine.version_manager import VersionManager

    vm = VersionManager(f"{_get_kb_storage_path(kb_id)}/versions.db")
    snapshot = vm.get_snapshot(kb_id, version_id)
    if not snapshot:
        raise HTTPException(404, f"Snapshot '{version_id}' not found")
    return snapshot


@router.post("/bases/{kb_id}/versions/{version_id}/rollback", dependencies=[Depends(verify_api_key)])
async def rollback_snapshot(kb_id: str, version_id: str) -> dict[str, Any]:
    _get_base(kb_id)
    from ..engine.version_manager import VersionManager

    vm = VersionManager(f"{_get_kb_storage_path(kb_id)}/versions.db")
    result = vm.rollback(kb_id, _get_kb_storage_path(kb_id), version_id)
    if not result.get("success", False):
        raise HTTPException(400, result.get("error", "Rollback failed"))
    return result


@router.delete("/bases/{kb_id}/versions/{version_id}", dependencies=[Depends(verify_api_key)])
async def delete_snapshot(kb_id: str, version_id: str) -> dict[str, Any]:
    _get_base(kb_id)
    from ..engine.version_manager import VersionManager

    vm = VersionManager(f"{_get_kb_storage_path(kb_id)}/versions.db")
    if not vm.delete_snapshot(kb_id, version_id):
        raise HTTPException(404, f"Snapshot '{version_id}' not found")
    return {"version_id": version_id, "status": "deleted"}


# ── Search Templates ──


@router.get("/bases/{kb_id}/templates", dependencies=[Depends(verify_api_key)])
async def list_templates(kb_id: str) -> list[dict[str, Any]]:
    _get_base(kb_id)
    from ..engine.search_template import SearchTemplateManager

    mgr = SearchTemplateManager(f"{_get_kb_storage_path(kb_id)}/templates.db")
    return mgr.list_templates(kb_id)


@router.post("/bases/{kb_id}/templates", dependencies=[Depends(verify_api_key)])
async def create_template(kb_id: str, data: dict[str, Any]) -> dict[str, Any]:
    _get_base(kb_id)
    from ..engine.search_template import SearchTemplate, SearchTemplateManager

    mgr = SearchTemplateManager(f"{_get_kb_storage_path(kb_id)}/templates.db")
    tpl = SearchTemplate(
        name=data.get("name", ""),
        description=data.get("description", ""),
        alpha=data.get("alpha", 0.7),
        rerank=data.get("rerank", False),
        top_k=data.get("top_k", 10),
        threshold=data.get("threshold", 0.5),
        rewrite_mode=data.get("rewrite_mode", ""),
        doc_type_filter=data.get("doc_type_filter", []),
    )
    return mgr.create_template(kb_id, tpl)


@router.delete("/bases/{kb_id}/templates/{name}", dependencies=[Depends(verify_api_key)])
async def delete_template(kb_id: str, name: str) -> dict[str, Any]:
    _get_base(kb_id)
    from ..engine.search_template import SearchTemplateManager

    mgr = SearchTemplateManager(f"{_get_kb_storage_path(kb_id)}/templates.db")
    if not mgr.delete_template(kb_id, name):
        raise HTTPException(404, f"Template '{name}' not found or is builtin")
    return {"name": name, "status": "deleted"}


# ── Permissions ──


@router.get("/bases/{kb_id}/permissions", dependencies=[Depends(verify_api_key)])
async def list_permissions(kb_id: str) -> list[dict[str, Any]]:
    _get_base(kb_id)
    from ..permissions import PermissionManager

    pm = PermissionManager(f"{_get_kb_storage_path(kb_id)}/permissions.db")
    return pm.list_rules(kb_id)


@router.post("/bases/{kb_id}/permissions", dependencies=[Depends(verify_api_key)])
async def add_permission(kb_id: str, data: dict[str, Any]) -> dict[str, Any]:
    _get_base(kb_id)
    from ..permissions import PermissionManager, PermissionRule

    pm = PermissionManager(f"{_get_kb_storage_path(kb_id)}/permissions.db")
    rule = PermissionRule(
        id="",
        kb_id=kb_id,
        subject=data.get("subject", ""),
        resource_type=data.get("resource_type", "kb"),
        resource_path=data.get("resource_path", "/"),
        actions=data.get("actions", ["read"]),
    )
    return pm.add_rule(rule)


@router.delete("/bases/{kb_id}/permissions/{rule_id}", dependencies=[Depends(verify_api_key)])
async def delete_permission(kb_id: str, rule_id: str) -> dict[str, Any]:
    _get_base(kb_id)
    from ..permissions import PermissionManager

    pm = PermissionManager(f"{_get_kb_storage_path(kb_id)}/permissions.db")
    if not pm.delete_rule(rule_id):
        raise HTTPException(404, f"Permission rule '{rule_id}' not found")
    return {"rule_id": rule_id, "status": "deleted"}


@router.post("/bases/{kb_id}/permissions/check")
async def check_permission(kb_id: str, data: dict[str, Any]) -> dict[str, Any]:
    _get_base(kb_id)
    from ..permissions import PermissionManager

    pm = PermissionManager(f"{_get_kb_storage_path(kb_id)}/permissions.db")
    allowed = pm.check_permission(
        kb_id,
        data.get("subject", ""),
        data.get("action", "read"),
        data.get("resource_path", "/"),
    )
    return {"allowed": allowed}


# ── Audit Logs ──


@router.get("/bases/{kb_id}/audit", dependencies=[Depends(verify_api_key)])
async def list_audit_logs(
    kb_id: str, limit: int = 50, offset: int = 0, caller: str | None = None
) -> list[dict[str, Any]]:
    _get_base(kb_id)
    from ..engine.audit_logger import AuditLogger

    al = AuditLogger(f"{_get_kb_storage_path(kb_id)}/audit.db")
    return al.query_logs(kb_id, limit=limit, offset=offset, caller=caller)


@router.get("/bases/{kb_id}/audit/{log_id}", dependencies=[Depends(verify_api_key)])
async def get_audit_log(kb_id: str, log_id: int) -> dict[str, Any]:
    _get_base(kb_id)
    from ..engine.audit_logger import AuditLogger

    al = AuditLogger(f"{_get_kb_storage_path(kb_id)}/audit.db")
    log = al.get_log(log_id)
    if not log or log.get("kb_id") != kb_id:
        raise HTTPException(404, f"Audit log '{log_id}' not found")
    return log


@router.get("/bases/{kb_id}/audit/export", dependencies=[Depends(verify_api_key)])
async def export_audit_logs(kb_id: str, format: str = "json") -> str:
    _get_base(kb_id)
    from ..engine.audit_logger import AuditLogger

    al = AuditLogger(f"{_get_kb_storage_path(kb_id)}/audit.db")
    return al.export_logs(kb_id, format=format)


# ── Incremental Sync ──


@router.post("/bases/{kb_id}/sync", dependencies=[Depends(verify_api_key)])
async def incremental_sync(kb_id: str, data: dict[str, Any]) -> dict[str, Any]:
    _get_base(kb_id)
    directory = data.get("directory", "")
    if not directory:
        raise HTTPException(400, "directory is required")

    from ..engine.incremental_sync import IncrementalSyncEngine
    from .routes import _get_meta_store

    sync = IncrementalSyncEngine()
    meta_store = _get_meta_store(kb_id)
    docs = meta_store.list_documents()
    for d in docs:
        d["file_path"] = d.get("file_path", "")
        d["file_hash"] = d.get("metadata", {}).get("file_hash", "") if isinstance(d.get("metadata"), dict) else ""
        d["doc_id"] = d.get("doc_id", d.get("id", ""))

    patterns = data.get("patterns")
    return sync.sync_directory(directory, docs, patterns)


# ── Bench ──


@router.post("/bases/{kb_id}/bench", dependencies=[Depends(verify_api_key)])
async def run_bench(kb_id: str, data: dict[str, Any]) -> dict[str, Any]:
    _get_base(kb_id)
    queries = data.get("queries", [])
    if not queries:
        raise HTTPException(400, "queries list is required")

    from ..engine.bench import BenchRunner
    from .routes import _get_embed_client, _get_vector_store

    bench = BenchRunner(f"{_get_kb_storage_path(kb_id)}/bench.db")
    vec_store = _get_vector_store(kb_id)
    embed = _get_embed_client()
    return await bench.run_search_bench(kb_id, vec_store, embed, queries)


@router.get("/bases/{kb_id}/bench/results", dependencies=[Depends(verify_api_key)])
async def list_bench_results(kb_id: str, test_name: str | None = None) -> list[dict[str, Any]]:
    _get_base(kb_id)
    from ..engine.bench import BenchRunner

    bench = BenchRunner(f"{_get_kb_storage_path(kb_id)}/bench.db")
    return bench.list_results(kb_id, test_name)
