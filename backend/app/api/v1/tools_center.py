from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.domain_router import add_action_route, add_crud_routes, add_list_route

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
    ("post", "/tools/{tool_id}/invoke", "tools", "invoke", "tool_call_logs"),
    ("get", "/tool-call-logs", "tool_call_logs", "logs", None),
    ("get", "/tool-approval-tasks", "tool_approval_tasks", "approvals", None),
    ("post", "/tool-approval-tasks/{approval_id}/approve", "tool_approval_tasks", "approve", "tool_call_logs"),
    ("post", "/tool-approval-tasks/{approval_id}/reject", "tool_approval_tasks", "reject", "tool_call_logs"),
    ("post", "/mcp-servers/{server_id}/sync-tools", "mcp_servers", "sync_tools", "mcp_tools"),
    ("get", "/mcp-servers/{server_id}/tools", "mcp_tools", "mcp_tools", None),
]:
    if method == "get":
        add_list_route(router, method=method, path=path, table=table, permission="tools")
    else:
        add_action_route(router, method=method, path=path, table=table, permission="tools", action=action, output_table=output)

