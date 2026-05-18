from __future__ import annotations

from fastapi import APIRouter, Body, Depends, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session

from app.api.deps import TenantIdDep, get_db, require_permission
from app.models.core import User
from app.runtime.service import RuntimeControlService
from app.websocket.manager import websocket_manager

router = APIRouter(tags=["runtime"])


@router.post("/runtime/agents/{agent_id}/instances")
async def create_instance(agent_id: str, tenant_id: TenantIdDep, payload: dict[str, object] = Body(default_factory=dict), user: User = Depends(require_permission("runtime:write")), db: Session = Depends(get_db)) -> dict[str, object]:
    return await RuntimeControlService(db).create_instance(tenant_id, user.id, agent_id, dict(payload))


@router.post("/runtime/instances/{instance_id}/start")
async def start_instance(instance_id: str, tenant_id: TenantIdDep, user: User = Depends(require_permission("runtime:write")), db: Session = Depends(get_db)) -> dict[str, object]:
    return await RuntimeControlService(db).start(tenant_id, user.id, instance_id)


@router.post("/runtime/instances/{instance_id}/stop")
async def stop_instance(instance_id: str, tenant_id: TenantIdDep, user: User = Depends(require_permission("runtime:write")), db: Session = Depends(get_db)) -> dict[str, object]:
    return await RuntimeControlService(db).stop(tenant_id, user.id, instance_id)


@router.post("/runtime/instances/{instance_id}/restart")
async def restart_instance(instance_id: str, tenant_id: TenantIdDep, user: User = Depends(require_permission("runtime:write")), db: Session = Depends(get_db)) -> dict[str, object]:
    return await RuntimeControlService(db).restart(tenant_id, user.id, instance_id)


@router.delete("/runtime/instances/{instance_id}")
async def destroy_instance(instance_id: str, tenant_id: TenantIdDep, user: User = Depends(require_permission("runtime:write")), db: Session = Depends(get_db)) -> dict[str, object]:
    return await RuntimeControlService(db).destroy(tenant_id, user.id, instance_id)


@router.get("/runtime/instances/{instance_id}/health")
def instance_health(instance_id: str, db: Session = Depends(get_db), _: User = Depends(require_permission("runtime:read"))) -> dict[str, object]:
    return RuntimeControlService(db).adapter.health_check(instance_id)


@router.get("/runtime/instances")
def list_instances(tenant_id: TenantIdDep, db: Session = Depends(get_db), _: User = Depends(require_permission("runtime:read"))) -> list[dict[str, object]]:
    return RuntimeControlService(db).list_instances(tenant_id)


@router.post("/runtime/agents/{agent_id}/debug-run")
async def debug_run(agent_id: str, tenant_id: TenantIdDep, payload: dict[str, object] = Body(default_factory=dict), user: User = Depends(require_permission("runtime:write")), db: Session = Depends(get_db)) -> dict[str, object]:
    return await RuntimeControlService(db).debug_run(tenant_id, user.id, agent_id, dict(payload))


@router.get("/runtime/runs/{run_id}")
def get_run(run_id: str, tenant_id: TenantIdDep, db: Session = Depends(get_db), _: User = Depends(require_permission("runtime:read"))) -> dict[str, object]:
    row = RuntimeControlService(db).resources.get("agent_run_records", tenant_id, run_id)
    return RuntimeControlService(db).resources.to_dict(row)


@router.websocket("/ws/runtime/runs/{run_id}")
async def runtime_ws(websocket: WebSocket, run_id: str) -> None:
    await websocket_manager.connect(run_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        websocket_manager.disconnect(run_id, websocket)

