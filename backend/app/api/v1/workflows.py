from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.api.v1.domain_router import add_action_route, add_crud_routes, add_list_route
from app.websocket.manager import websocket_manager

router = APIRouter(tags=["workflows"])

add_crud_routes(router, table="workflow_definitions", prefix="/workflows", permission="workflows")
add_crud_routes(router, table="workflow_templates", prefix="/workflow-templates", permission="workflows")

for method, path, table, action, output in [
    ("post", "/workflows/{workflow_id}/versions", "workflow_definitions", "create_version", "workflow_versions"),
    ("get", "/workflows/{workflow_id}/versions", "workflow_versions", "versions", None),
    ("post", "/workflows/{workflow_id}/validate", "workflow_definitions", "validate", "workflow_run_events"),
    ("post", "/workflows/{workflow_id}/publish", "workflow_definitions", "publish", "workflow_versions"),
    ("post", "/workflows/{workflow_id}/run", "workflow_definitions", "run", "workflow_runs"),
    ("get", "/workflows/{workflow_id}/runs", "workflow_runs", "runs", None),
    ("get", "/workflow-runs/{run_id}", "workflow_runs", "run", None),
    ("get", "/workflow-runs/{run_id}/steps", "workflow_run_steps", "steps", None),
    ("get", "/workflow-runs/{run_id}/logs", "workflow_run_logs", "logs", None),
    ("post", "/workflow-runs/{run_id}/retry", "workflow_runs", "retry", "workflow_run_events"),
    ("post", "/workflow-runs/{run_id}/cancel", "workflow_runs", "cancel", "workflow_run_events"),
    ("post", "/workflow-runs/{run_id}/replay", "workflow_runs", "replay", "workflow_run_events"),
    ("get", "/workflow-approvals", "workflow_approvals", "approvals", None),
    ("post", "/workflow-approvals/{approval_id}/approve", "workflow_approvals", "approve", "workflow_run_events"),
    ("post", "/workflow-approvals/{approval_id}/reject", "workflow_approvals", "reject", "workflow_run_events"),
]:
    if method == "get":
        add_list_route(router, method=method, path=path, table=table, permission="workflows")
    else:
        add_action_route(router, method=method, path=path, table=table, permission="workflows", action=action, output_table=output)


@router.websocket("/ws/workflow-runs/{run_id}")
async def workflow_ws(websocket: WebSocket, run_id: str) -> None:
    await websocket_manager.connect(run_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        websocket_manager.disconnect(run_id, websocket)

