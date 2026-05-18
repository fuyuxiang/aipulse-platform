from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import TenantIdDep, get_db, require_permission
from app.api.v1.domain_router import add_action_route, add_list_route
from app.models.core import User
from app.services.audit_service import AuditService

router = APIRouter(tags=["audit"])

add_list_route(router, method="get", path="/audit-logs", table="audit_logs", permission="audit")
add_list_route(router, method="get", path="/audit-logs/{audit_id}", table="audit_logs", permission="audit")
add_action_route(router, method="post", path="/audit-logs/export", table="audit_logs", permission="audit", action="export", output_table="audit_exports")
add_list_route(router, method="get", path="/audit-exports", table="audit_exports", permission="audit")


@router.get("/audit-integrity/verify")
def verify_integrity(tenant_id: TenantIdDep, db: Session = Depends(get_db), _: User = Depends(require_permission("audit:read"))) -> dict[str, object]:
    return AuditService(db).verify_chain(tenant_id)

