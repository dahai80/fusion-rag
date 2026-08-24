from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from .._validators import validate_identifier
from .app_state import get_kb_manager, get_project_kb_map
from .auth import verify_api_key

logger = logging.getLogger(__name__)

router = APIRouter(tags=["project"])

# 硬伤1: read kb_manager + project_kb_map from app.state via contextvar.
_get_kb_manager = get_kb_manager


def _get_base(kb_id: str):
    # F12: confine kb_id before path construction (path-traversal guard).
    try:
        validate_identifier(kb_id, field="kb_id")
    except ValueError:
        raise HTTPException(400, f"Invalid kb_id: {kb_id}")
    try:
        return _get_kb_manager().get(kb_id)
    except KeyError:
        raise HTTPException(404, f"Knowledge base '{kb_id}' not found")


@router.post("/projects/{project_id}/kb", dependencies=[Depends(verify_api_key)])
async def map_project_kb(project_id: str, data: dict[str, Any]) -> dict[str, Any]:
    # F12: project_id is used as a map key but still validate the charset so a
    # malicious id cannot carry separators that could collide or confuse.
    try:
        validate_identifier(project_id, field="project_id")
    except ValueError:
        raise HTTPException(400, f"Invalid project_id: {project_id}")
    kb_id = data.get("kb_id", "")
    if not kb_id:
        name = data.get("name", f"project-{project_id}")
        description = data.get("description", f"Auto-created for project {project_id}")
        kb = _get_kb_manager().create(name=name, description=description)
        kb_id = kb.id
        logger.info("created kb %s for project %s", kb_id, project_id)
    else:
        _get_base(kb_id)

    get_project_kb_map()[project_id] = kb_id
    return {"project_id": project_id, "kb_id": kb_id}


@router.get("/projects/{project_id}/kb")
async def get_project_kb(project_id: str) -> dict[str, Any]:
    kb_id = get_project_kb_map().get(project_id)
    if not kb_id:
        raise HTTPException(404, f"No KB mapped for project '{project_id}'")
    return {"project_id": project_id, "kb_id": kb_id}


@router.delete("/projects/{project_id}/kb", dependencies=[Depends(verify_api_key)])
async def unmap_project_kb(project_id: str) -> dict[str, Any]:
    project_kb_map = get_project_kb_map()
    if project_id not in project_kb_map:
        raise HTTPException(404, f"No KB mapped for project '{project_id}'")
    kb_id = project_kb_map.pop(project_id)
    return {"project_id": project_id, "kb_id": kb_id, "unmapped": True}
