from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from .._validators import validate_identifier
from .access import require_kb_action
from .app_state import (
    get_audit_logger,
    get_bench_runner,
    get_kb_manager,
    get_permission_manager,
    get_template_manager,
    get_version_manager,
)

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


@router.post("/bases/{kb_id}/versions", dependencies=[Depends(require_kb_action("write"))])
async def create_snapshot(kb_id: str, data: dict[str, Any]) -> dict[str, Any]:
    _get_base(kb_id)
    vm = get_version_manager(_get_kb_storage_path(kb_id))
    description = data.get("description", "")
    return vm.create_snapshot(kb_id, _get_kb_storage_path(kb_id), description)


@router.get("/bases/{kb_id}/versions", dependencies=[Depends(require_kb_action("read"))])
async def list_snapshots(kb_id: str) -> list[dict[str, Any]]:
    _get_base(kb_id)
    vm = get_version_manager(_get_kb_storage_path(kb_id))
    return vm.list_snapshots(kb_id)


@router.get("/bases/{kb_id}/versions/{version_id}", dependencies=[Depends(require_kb_action("read"))])
async def get_snapshot(kb_id: str, version_id: str) -> dict[str, Any]:
    _get_base(kb_id)
    vm = get_version_manager(_get_kb_storage_path(kb_id))
    snapshot = vm.get_snapshot(kb_id, version_id)
    if not snapshot:
        raise HTTPException(404, f"Snapshot '{version_id}' not found")
    return snapshot


@router.post("/bases/{kb_id}/versions/{version_id}/rollback", dependencies=[Depends(require_kb_action("write"))])
async def rollback_snapshot(kb_id: str, version_id: str) -> dict[str, Any]:
    _get_base(kb_id)
    vm = get_version_manager(_get_kb_storage_path(kb_id))
    result = vm.rollback(kb_id, _get_kb_storage_path(kb_id), version_id)
    if not result.get("success", False):
        raise HTTPException(400, result.get("error", "Rollback failed"))
    return result


@router.delete("/bases/{kb_id}/versions/{version_id}", dependencies=[Depends(require_kb_action("delete"))])
async def delete_snapshot(kb_id: str, version_id: str) -> dict[str, Any]:
    _get_base(kb_id)
    vm = get_version_manager(_get_kb_storage_path(kb_id))
    if not vm.delete_snapshot(kb_id, version_id):
        raise HTTPException(404, f"Snapshot '{version_id}' not found")
    return {"version_id": version_id, "status": "deleted"}


# ── Search Templates ──


@router.get("/bases/{kb_id}/templates", dependencies=[Depends(require_kb_action("read"))])
async def list_templates(kb_id: str) -> list[dict[str, Any]]:
    _get_base(kb_id)
    mgr = get_template_manager(_get_kb_storage_path(kb_id))
    return mgr.list_templates(kb_id)


@router.post("/bases/{kb_id}/templates", dependencies=[Depends(require_kb_action("write"))])
async def create_template(kb_id: str, data: dict[str, Any]) -> dict[str, Any]:
    _get_base(kb_id)
    from ..engine.search_template import SearchTemplate

    mgr = get_template_manager(_get_kb_storage_path(kb_id))
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


@router.delete("/bases/{kb_id}/templates/{name}", dependencies=[Depends(require_kb_action("delete"))])
async def delete_template(kb_id: str, name: str) -> dict[str, Any]:
    _get_base(kb_id)
    mgr = get_template_manager(_get_kb_storage_path(kb_id))
    if not mgr.delete_template(kb_id, name):
        raise HTTPException(404, f"Template '{name}' not found or is builtin")
    return {"name": name, "status": "deleted"}


# ── Permissions ──


@router.get("/bases/{kb_id}/permissions", dependencies=[Depends(require_kb_action("read"))])
async def list_permissions(kb_id: str) -> list[dict[str, Any]]:
    _get_base(kb_id)
    pm = get_permission_manager(_get_kb_storage_path(kb_id))
    return pm.list_rules(kb_id)


@router.post("/bases/{kb_id}/permissions", dependencies=[Depends(require_kb_action("write"))])
async def add_permission(kb_id: str, data: dict[str, Any]) -> dict[str, Any]:
    _get_base(kb_id)
    from ..permissions import PermissionRule

    pm = get_permission_manager(_get_kb_storage_path(kb_id))
    rule = PermissionRule(
        id="",
        kb_id=kb_id,
        subject=data.get("subject", ""),
        resource_type=data.get("resource_type", "kb"),
        resource_path=data.get("resource_path", "/"),
        actions=data.get("actions", ["read"]),
    )
    return pm.add_rule(rule)


@router.delete("/bases/{kb_id}/permissions/{rule_id}", dependencies=[Depends(require_kb_action("delete"))])
async def delete_permission(kb_id: str, rule_id: str) -> dict[str, Any]:
    _get_base(kb_id)
    pm = get_permission_manager(_get_kb_storage_path(kb_id))
    if not pm.delete_rule(rule_id):
        raise HTTPException(404, f"Permission rule '{rule_id}' not found")
    return {"rule_id": rule_id, "status": "deleted"}


@router.post("/bases/{kb_id}/permissions/check", dependencies=[Depends(require_kb_action("read"))])
async def check_permission(kb_id: str, data: dict[str, Any]) -> dict[str, Any]:
    _get_base(kb_id)
    pm = get_permission_manager(_get_kb_storage_path(kb_id))
    allowed = pm.check_permission(
        kb_id,
        data.get("subject", ""),
        data.get("action", "read"),
        data.get("resource_path", "/"),
    )
    return {"allowed": allowed}


# ── Audit Logs ──


@router.get("/bases/{kb_id}/audit", dependencies=[Depends(require_kb_action("read"))])
async def list_audit_logs(
    kb_id: str, limit: int = 50, offset: int = 0, caller: str | None = None
) -> list[dict[str, Any]]:
    _get_base(kb_id)
    al = get_audit_logger(_get_kb_storage_path(kb_id))
    return al.query_logs(kb_id, limit=limit, offset=offset, caller=caller)


@router.get("/bases/{kb_id}/audit/{log_id}", dependencies=[Depends(require_kb_action("read"))])
async def get_audit_log(kb_id: str, log_id: int) -> dict[str, Any]:
    _get_base(kb_id)
    al = get_audit_logger(_get_kb_storage_path(kb_id))
    log = al.get_log(log_id)
    if not log or log.get("kb_id") != kb_id:
        raise HTTPException(404, f"Audit log '{log_id}' not found")
    return log


@router.get("/bases/{kb_id}/audit/export", dependencies=[Depends(require_kb_action("read"))], response_model=None)
async def export_audit_logs(kb_id: str, output_format: str = Query("json", alias="format")) -> Response | str:
    # M7: param was named `format`, shadowing the builtin. Renamed to
    # output_format; Query(alias="format") keeps the public ?format= query
    # param name unchanged.
    _get_base(kb_id)
    al = get_audit_logger(_get_kb_storage_path(kb_id))
    payload = al.export_logs(kb_id, fmt=output_format)
    if output_format == "csv":
        # P0-12: force download (not inline render). csv.writer already prefixes
        # a single quote on cells starting with =/+/-/@/tab/CR to neutralize
        # spreadsheet formula execution on open. Content-Disposition: attachment
        # stops browsers from auto-opening the CSV in Excel/Numbers where the
        # neutralization is still load-bearing.
        return Response(
            content=payload,
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="audit.csv"'},
        )
    return payload


# ── Incremental Sync ──


@router.post("/bases/{kb_id}/sync", dependencies=[Depends(require_kb_action("write"))])
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


@router.post("/bases/{kb_id}/bench", dependencies=[Depends(require_kb_action("write"))])
async def run_bench(kb_id: str, data: dict[str, Any]) -> dict[str, Any]:
    _get_base(kb_id)
    queries = data.get("queries", [])
    if not queries:
        raise HTTPException(400, "queries list is required")

    from .routes import _get_embed_client, _get_vector_store

    bench = get_bench_runner(_get_kb_storage_path(kb_id))
    vec_store = _get_vector_store(kb_id)
    embed = _get_embed_client()
    return await bench.run_search_bench(kb_id, vec_store, embed, queries)


@router.get("/bases/{kb_id}/bench/results", dependencies=[Depends(require_kb_action("read"))])
async def list_bench_results(kb_id: str, test_name: str | None = None) -> list[dict[str, Any]]:
    _get_base(kb_id)
    bench = get_bench_runner(_get_kb_storage_path(kb_id))
    return bench.list_results(kb_id, test_name)
