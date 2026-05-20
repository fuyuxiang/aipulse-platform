from __future__ import annotations

from fastapi import APIRouter, Body, Depends
from sqlalchemy.orm import Session

from app.api.deps import TenantIdDep, get_db, require_permission
from app.api.v1.domain_router import add_list_route
from app.models.core import User
from app.schemas.common import ResourceRead
from app.services.observe.audit_service import AuditService
from app.services._shared.resource_service import ResourceService

router = APIRouter(tags=["audit"])

add_list_route(router, method="get", path="/audit-logs", table="audit_logs", permission="audit")
add_list_route(router, method="get", path="/audit-exports", table="audit_exports", permission="audit")


@router.get("/audit-logs/{audit_id}", response_model=ResourceRead)
def audit_detail(audit_id: str, tenant_id: TenantIdDep, _: User = Depends(require_permission("audit:read")), db: Session = Depends(get_db)) -> ResourceRead:
    return ResourceRead.model_validate(ResourceService.to_dict(ResourceService(db).get("audit_logs", tenant_id, audit_id)))


@router.post("/audit-logs/export")
def export_audit(
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    user: User = Depends(require_permission("audit:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return AuditService(db).export(tenant_id, user.id, dict(payload))


@router.get("/audit-integrity/verify")
def verify_integrity(tenant_id: TenantIdDep, db: Session = Depends(get_db), _: User = Depends(require_permission("audit:read"))) -> dict[str, object]:
    return AuditService(db).verify_chain(tenant_id)
