from __future__ import annotations

from fastapi import APIRouter, Body, Depends
from sqlalchemy.orm import Session

from app.api.deps import TenantIdDep, get_db, require_permission
from app.api.v1.domain_router import add_crud_routes, add_list_route
from app.models.core import User
from app.services.settings.security_service import SecurityService

router = APIRouter(tags=["security"])

for table, prefix in [
    ("sensitive_rules", "/security/sensitive-rules"),
    ("content_safety_policies", "/security/content-policies"),
    ("prompt_injection_rules", "/security/prompt-injection-rules"),
    ("ip_allowlists", "/security/ip-allowlists"),
    ("api_rate_limit_rules", "/security/rate-limit-rules"),
    ("secret_refs", "/security/secrets"),
    ("risk_approval_policies", "/security/risk-approval-policies"),
]:
    add_crud_routes(router, table=table, prefix=prefix, permission="security")


@router.post("/security/check")
def check(tenant_id: TenantIdDep, payload: dict[str, object] = Body(default_factory=dict), user: User = Depends(require_permission("security:write")), db: Session = Depends(get_db)) -> dict[str, object]:
    return SecurityService(db).check(tenant_id, user.id, dict(payload))


@router.post("/security/desensitize")
def desensitize(payload: dict[str, object] = Body(default_factory=dict), _: User = Depends(require_permission("security:write"))) -> dict[str, object]:
    return SecurityService.desensitize(dict(payload))


add_list_route(router, method="get", path="/security/events", table="security_events", permission="security")
