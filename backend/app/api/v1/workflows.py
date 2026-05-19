from __future__ import annotations

from fastapi import APIRouter, Body, Depends, Query, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session

from app.api.deps import TenantIdDep, get_db, require_permission
from app.api.v1.domain_router import add_crud_routes, add_list_route
from app.core.response import ListResponse
from app.models.core import User
from app.schemas.common import ResourceRead
from app.services.resource_service import ResourceService
from app.services.workflow_service import WorkflowService
from app.websocket.manager import websocket_manager

router = APIRouter(tags=["workflows"])

add_crud_routes(router, table="workflow_definitions", prefix="/workflows", permission="workflows")
add_crud_routes(router, table="workflow_templates", prefix="/workflow-templates", permission="workflows")

@router.post("/workflows/{workflow_id}/versions")
def create_workflow_version(
    workflow_id: str,
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    user: User = Depends(require_permission("workflows:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return WorkflowService(db).create_version(tenant_id, user.id, workflow_id, dict(payload))


@router.get("/workflows/{workflow_id}/versions", response_model=ListResponse[ResourceRead])
def workflow_versions(
    workflow_id: str,
    tenant_id: TenantIdDep,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    _: User = Depends(require_permission("workflows:read")),
    db: Session = Depends(get_db),
) -> ListResponse[ResourceRead]:
    rows, total = ResourceService(db).list("workflow_versions", tenant_id, page, page_size, {"parent_id": workflow_id})
    return ListResponse(items=[ResourceRead.model_validate(ResourceService.to_dict(row)) for row in rows], total=total, page=page, page_size=page_size)


@router.post("/workflows/{workflow_id}/validate")
def validate_workflow(
    workflow_id: str,
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    user: User = Depends(require_permission("workflows:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return WorkflowService(db).validate(tenant_id, user.id, workflow_id, dict(payload))


@router.post("/workflows/{workflow_id}/publish")
def publish_workflow(
    workflow_id: str,
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    user: User = Depends(require_permission("workflows:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return WorkflowService(db).publish(tenant_id, user.id, workflow_id, dict(payload))


@router.post("/workflows/{workflow_id}/run")
async def run_workflow(
    workflow_id: str,
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    user: User = Depends(require_permission("workflows:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return await WorkflowService(db).run(tenant_id, user.id, workflow_id, dict(payload))


@router.get("/workflows/{workflow_id}/runs", response_model=ListResponse[ResourceRead])
def workflow_runs(
    workflow_id: str,
    tenant_id: TenantIdDep,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    _: User = Depends(require_permission("workflows:read")),
    db: Session = Depends(get_db),
) -> ListResponse[ResourceRead]:
    rows, total = ResourceService(db).list("workflow_runs", tenant_id, page, page_size, {"workflow_id": workflow_id})
    return ListResponse(items=[ResourceRead.model_validate(ResourceService.to_dict(row)) for row in rows], total=total, page=page, page_size=page_size)


@router.get("/workflow-runs/{run_id}", response_model=ResourceRead)
def workflow_run(run_id: str, tenant_id: TenantIdDep, _: User = Depends(require_permission("workflows:read")), db: Session = Depends(get_db)) -> ResourceRead:
    return ResourceRead.model_validate(ResourceService.to_dict(ResourceService(db).get("workflow_runs", tenant_id, run_id)))


@router.get("/workflow-runs/{run_id}/steps", response_model=ListResponse[ResourceRead])
def workflow_run_steps(
    run_id: str,
    tenant_id: TenantIdDep,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    _: User = Depends(require_permission("workflows:read")),
    db: Session = Depends(get_db),
) -> ListResponse[ResourceRead]:
    rows, total = ResourceService(db).list("workflow_run_steps", tenant_id, page, page_size, {"run_id": run_id})
    return ListResponse(items=[ResourceRead.model_validate(ResourceService.to_dict(row)) for row in rows], total=total, page=page, page_size=page_size)


@router.get("/workflow-runs/{run_id}/logs", response_model=ListResponse[ResourceRead])
def workflow_run_logs(
    run_id: str,
    tenant_id: TenantIdDep,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    _: User = Depends(require_permission("workflows:read")),
    db: Session = Depends(get_db),
) -> ListResponse[ResourceRead]:
    rows, total = ResourceService(db).list("workflow_run_logs", tenant_id, page, page_size, {"run_id": run_id})
    return ListResponse(items=[ResourceRead.model_validate(ResourceService.to_dict(row)) for row in rows], total=total, page=page, page_size=page_size)


@router.post("/workflow-runs/{run_id}/retry")
async def retry_workflow_run(
    run_id: str,
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    user: User = Depends(require_permission("workflows:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return await WorkflowService(db).retry(tenant_id, user.id, run_id, dict(payload))


@router.post("/workflow-runs/{run_id}/cancel")
def cancel_workflow_run(
    run_id: str,
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    user: User = Depends(require_permission("workflows:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return WorkflowService(db).cancel(tenant_id, user.id, run_id, dict(payload))


@router.post("/workflow-runs/{run_id}/replay")
def replay_workflow_run(run_id: str, tenant_id: TenantIdDep, user: User = Depends(require_permission("workflows:write")), db: Session = Depends(get_db)) -> dict[str, object]:
    return WorkflowService(db).replay(tenant_id, user.id, run_id)


@router.post("/workflow-runs/{run_id}/resume")
async def resume_workflow_run(
    run_id: str,
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    user: User = Depends(require_permission("workflows:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return await WorkflowService(db).resume_from_checkpoint(tenant_id, user.id, run_id, dict(payload))


@router.get("/workflow-runs/{run_id}/checkpoints")
async def workflow_run_checkpoints(
    run_id: str,
    tenant_id: TenantIdDep,
    _: User = Depends(require_permission("workflows:read")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    checkpoints = await WorkflowService(db).list_checkpoints(tenant_id, run_id)
    return {"run_id": run_id, "checkpoints": checkpoints}


@router.post("/workflow-events/{event_name}/trigger")
async def trigger_workflow_event(
    event_name: str,
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    user: User = Depends(require_permission("workflows:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return await WorkflowService(db).trigger_event(tenant_id, user.id, event_name, dict(payload))


add_list_route(router, method="get", path="/workflow-approvals", table="workflow_approvals", permission="workflows")


@router.post("/workflow-approvals/{approval_id}/approve")
def approve_workflow(
    approval_id: str,
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    user: User = Depends(require_permission("workflows:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return WorkflowService(db).decide_approval(tenant_id, user.id, approval_id, True, dict(payload))


@router.post("/workflow-approvals/{approval_id}/reject")
def reject_workflow(
    approval_id: str,
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    user: User = Depends(require_permission("workflows:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return WorkflowService(db).decide_approval(tenant_id, user.id, approval_id, False, dict(payload))


@router.websocket("/ws/workflow-runs/{run_id}")
async def workflow_ws(websocket: WebSocket, run_id: str) -> None:
    await websocket_manager.connect(run_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        websocket_manager.disconnect(run_id, websocket)
