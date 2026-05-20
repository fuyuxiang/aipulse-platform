from __future__ import annotations

from fastapi import APIRouter, Body, Depends
from sqlalchemy.orm import Session

from app.api.deps import TenantIdDep, get_db, require_permission
from app.api.v1.domain_router import add_action_route, add_crud_routes, add_list_route
from app.models.core import User
from app.services.build.tool_service import ToolService

router = APIRouter(tags=["tools-center"])

for table, prefix in [
    ("tools", "/tools"),
    ("tool_approval_policies", "/tool-approval-policies"),
    ("tool_rate_limits", "/tool-rate-limits"),
    ("mcp_servers", "/mcp-servers"),
]:
    add_crud_routes(router, table=table, prefix=prefix, permission="tools")

for method, path, table, action, output in [
    ("post", "/tools/{tool_id}/versions", "tools", "create_version", "tool_versions"),
    ("get", "/tools/{tool_id}/versions", "tool_versions", "versions", None),
    ("post", "/tools/{tool_id}/permissions", "tools", "permissions", "tool_permissions"),
    ("get", "/tools/{tool_id}/permissions", "tool_permissions", "permissions", None),
    ("get", "/tool-call-logs", "tool_call_logs", "logs", None),
    ("get", "/tool-approval-tasks", "tool_approval_tasks", "approvals", None),
    ("get", "/mcp-servers/{server_id}/tools", "mcp_tools", "mcp_tools", None),
]:
    if method == "get":
        add_list_route(router, method=method, path=path, table=table, permission="tools")
    else:
        add_action_route(router, method=method, path=path, table=table, permission="tools", action=action, output_table=output)


@router.post("/tools/{tool_id}/invoke")
async def invoke_tool(
    tool_id: str,
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    user: User = Depends(require_permission("tools:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return await ToolService(db).invoke(tenant_id, user.id, tool_id, dict(payload))


@router.post("/tool-approval-tasks/{approval_id}/approve")
def approve_tool(
    approval_id: str,
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    user: User = Depends(require_permission("tools:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return ToolService(db).approve_task(tenant_id, user.id, approval_id, True, dict(payload))


@router.post("/tool-approval-tasks/{approval_id}/reject")
def reject_tool(
    approval_id: str,
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    user: User = Depends(require_permission("tools:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return ToolService(db).approve_task(tenant_id, user.id, approval_id, False, dict(payload))


@router.post("/mcp-servers/{server_id}/sync-tools")
def sync_mcp_tools(
    server_id: str,
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    user: User = Depends(require_permission("tools:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return ToolService(db).sync_mcp_tools(tenant_id, user.id, server_id, dict(payload))
