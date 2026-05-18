from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.domain_router import add_action_route, add_crud_routes, add_list_route

router = APIRouter(tags=["memory"])

add_crud_routes(router, table="memory_items", prefix="/memories", permission="memory")
add_crud_routes(router, table="memory_access_policies", prefix="/memory-access-policies", permission="memory")
add_crud_routes(router, table="memory_lifecycle_policies", prefix="/memory-lifecycle-policies", permission="memory")

for method, path, table, action, output in [
    ("post", "/memories/search", "memory_items", "search", "memory_operations"),
    ("post", "/memories/extract", "memory_items", "extract", "memory_extraction_jobs"),
    ("post", "/memories/merge", "memory_items", "merge", "memory_merge_jobs"),
    ("post", "/memories/{memory_id}/archive", "memory_items", "archive", "memory_operations"),
    ("post", "/memories/{memory_id}/desensitize", "memory_items", "desensitize", "memory_operations"),
    ("get", "/memory-audit-logs", "memory_audit_logs", "audit_logs", None),
    ("get", "/memory-conflicts", "memory_conflicts", "conflicts", None),
    ("post", "/memory-conflicts/{conflict_id}/resolve", "memory_conflicts", "resolve", "memory_operations"),
    ("post", "/memories/cleanup-expired", "memory_items", "cleanup_expired", "memory_cleanup_jobs"),
]:
    if method == "get":
        add_list_route(router, method=method, path=path, table=table, permission="memory")
    else:
        add_action_route(router, method=method, path=path, table=table, permission="memory", action=action, output_table=output)

