from __future__ import annotations

from fastapi import APIRouter, Body, Depends
from sqlalchemy.orm import Session

from app.api.deps import TenantIdDep, get_db, require_permission
from app.api.v1.domain_router import add_action_route, add_crud_routes, add_list_route
from app.models.core import User
from app.services.memory_service import MemoryService

router = APIRouter(tags=["memory"])

add_crud_routes(router, table="memory_items", prefix="/memories", permission="memory")
add_crud_routes(router, table="memory_access_policies", prefix="/memory-access-policies", permission="memory")
add_crud_routes(router, table="memory_lifecycle_policies", prefix="/memory-lifecycle-policies", permission="memory")

for method, path, table, action, output in [
    ("get", "/memory-audit-logs", "memory_audit_logs", "audit_logs", None),
    ("get", "/memory-conflicts", "memory_conflicts", "conflicts", None),
]:
    if method == "get":
        add_list_route(router, method=method, path=path, table=table, permission="memory")
    else:
        add_action_route(router, method=method, path=path, table=table, permission="memory", action=action, output_table=output)


@router.post("/memories/search")
def search_memories(
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    user: User = Depends(require_permission("memory:read")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return MemoryService(db).search(tenant_id, user.id, dict(payload))


@router.post("/memories/extract")
def extract_memories(
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    user: User = Depends(require_permission("memory:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return MemoryService(db).extract(tenant_id, user.id, dict(payload))


@router.post("/memories/merge")
def merge_memories(
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    user: User = Depends(require_permission("memory:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return MemoryService(db).merge(tenant_id, user.id, dict(payload))


@router.post("/memories/{memory_id}/archive")
def archive_memory(memory_id: str, tenant_id: TenantIdDep, user: User = Depends(require_permission("memory:write")), db: Session = Depends(get_db)) -> dict[str, object]:
    return MemoryService(db).archive(tenant_id, user.id, memory_id)


@router.post("/memories/{memory_id}/desensitize")
def desensitize_memory(memory_id: str, tenant_id: TenantIdDep, user: User = Depends(require_permission("memory:write")), db: Session = Depends(get_db)) -> dict[str, object]:
    return MemoryService(db).desensitize(tenant_id, user.id, memory_id)


@router.post("/memory-conflicts/{conflict_id}/resolve")
def resolve_memory_conflict(
    conflict_id: str,
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    user: User = Depends(require_permission("memory:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return MemoryService(db).resolve_conflict(tenant_id, user.id, conflict_id, dict(payload))


@router.post("/memories/cleanup-expired")
def cleanup_expired_memories(tenant_id: TenantIdDep, user: User = Depends(require_permission("memory:write")), db: Session = Depends(get_db)) -> dict[str, object]:
    return MemoryService(db).cleanup_expired(tenant_id, user.id)
