from __future__ import annotations

from fastapi import APIRouter, Body, Depends, Query
from sqlalchemy.orm import Session

from app.api.deps import TenantIdDep, get_db, require_permission
from app.models.core import User
from app.services.runtime.multi_agent_service import MultiAgentService

router = APIRouter(tags=["multi-agent"])


@router.post("/agent-teams")
def create_team(
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    user: User = Depends(require_permission("agents:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return MultiAgentService(db).create_team(tenant_id, user.id, dict(payload))


@router.get("/agent-teams")
def list_teams(
    tenant_id: TenantIdDep,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    name: str = "",
    user: User = Depends(require_permission("agents:read")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    items, total = MultiAgentService(db).list_teams(tenant_id, page, page_size, user.id, name)
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@router.get("/agent-teams/{team_id}")
def get_team(
    team_id: str,
    tenant_id: TenantIdDep,
    user: User = Depends(require_permission("agents:read")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return MultiAgentService(db).get_team(tenant_id, team_id, user.id)


@router.put("/agent-teams/{team_id}")
def update_team(
    team_id: str,
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    user: User = Depends(require_permission("agents:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return MultiAgentService(db).update_team(tenant_id, user.id, team_id, dict(payload))


@router.delete("/agent-teams/{team_id}")
def delete_team(
    team_id: str,
    tenant_id: TenantIdDep,
    user: User = Depends(require_permission("agents:write")),
    db: Session = Depends(get_db),
) -> dict[str, str]:
    return MultiAgentService(db).delete_team(tenant_id, user.id, team_id)


@router.post("/agent-teams/{team_id}/members")
def add_member(
    team_id: str,
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    user: User = Depends(require_permission("agents:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return MultiAgentService(db).add_member(tenant_id, user.id, team_id, dict(payload))


@router.put("/agent-teams/{team_id}/members/{member_id}")
def update_member(
    team_id: str,
    member_id: str,
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    user: User = Depends(require_permission("agents:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    service = MultiAgentService(db)
    service.require_team(tenant_id, user.id, team_id)
    member = service.resources.get("agent_team_members", tenant_id, member_id)
    if member.parent_id != team_id:
        from app.core.constants import ErrorCode
        from app.core.errors import AppError
        raise AppError(ErrorCode.NOT_FOUND, "team member not found", 404)
    return service.update_member(tenant_id, user.id, member_id, dict(payload))


@router.delete("/agent-teams/{team_id}/members/{member_id}")
def remove_member(
    team_id: str,
    member_id: str,
    tenant_id: TenantIdDep,
    user: User = Depends(require_permission("agents:write")),
    db: Session = Depends(get_db),
) -> dict[str, str]:
    return MultiAgentService(db).remove_member(tenant_id, user.id, team_id, member_id)


@router.post("/agent-teams/{team_id}/run")
async def run_team(
    team_id: str,
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    user: User = Depends(require_permission("agents:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return await MultiAgentService(db).run_team(tenant_id, user.id, team_id, dict(payload))


@router.get("/agent-teams/{team_id}/runs")
def list_team_runs(
    team_id: str,
    tenant_id: TenantIdDep,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    user: User = Depends(require_permission("agents:read")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    items, total = MultiAgentService(db).list_team_runs(tenant_id, team_id, page, page_size, user.id)
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@router.get("/agent-team-runs/{run_id}")
def get_team_run(
    run_id: str,
    tenant_id: TenantIdDep,
    user: User = Depends(require_permission("agents:read")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return MultiAgentService(db).get_team_run(tenant_id, run_id, user.id)
