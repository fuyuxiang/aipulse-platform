from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.domain_router import add_action_route, add_crud_routes, add_list_route

router = APIRouter(tags=["agents"])

add_crud_routes(router, table="agents", prefix="/agents", permission="agents")
add_crud_routes(router, table="agent_templates", prefix="/agent-templates", permission="agents")

for method, path, action, output in [
    ("post", "/agents/{agent_id}/clone", "clone", "agents"),
    ("post", "/agents/{agent_id}/versions", "create_version", "agent_versions"),
    ("post", "/agents/{agent_id}/release", "release", "agent_releases"),
    ("post", "/agents/{agent_id}/gray-release", "gray_release", "agent_release_strategies"),
    ("post", "/agents/{agent_id}/rollback", "rollback", "agent_releases"),
    ("post", "/agents/{agent_id}/disable", "disable", None),
    ("post", "/agents/{agent_id}/enable", "enable", None),
    ("post", "/agents/{agent_id}/debug-run", "debug_run", "agent_debug_sessions"),
    ("post", "/agents/import", "import", "agent_import_exports"),
    ("post", "/agent-templates", "create_template", "agent_templates"),
]:
    add_action_route(router, method=method, path=path, table="agents", permission="agents", action=action, output_table=output)

for path, table in [
    ("/agents/{agent_id}/versions", "agent_versions"),
    ("/agents/{agent_id}/runs", "agent_run_records"),
    ("/agent-templates", "agent_templates"),
]:
    add_list_route(router, method="get", path=path, table=table, permission="agents")

add_list_route(router, method="get", path="/agents/{agent_id}/status", table="agent_runtime_instances", permission="agents")
add_action_route(router, method="get", path="/agents/{agent_id}/export", table="agents", permission="agents", action="export", output_table="agent_import_exports")
add_list_route(router, method="get", path="/agents/{agent_id}/versions/{version_id}", table="agent_versions", permission="agents")

