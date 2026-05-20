from __future__ import annotations

from fastapi import APIRouter, Body, Depends, Query
from sqlalchemy.orm import Session

from app.api.deps import TenantIdDep, get_db, require_permission
from app.models.core import User
from app.services.settings.guardrail_service import GuardrailService

router = APIRouter(tags=["guardrails"])


@router.post("/guardrails/policies")
def create_policy(
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    user: User = Depends(require_permission("security:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return GuardrailService(db).create_policy(tenant_id, user.id, dict(payload))


@router.get("/guardrails/policies")
def list_policies(
    tenant_id: TenantIdDep,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    _: User = Depends(require_permission("security:read")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    items, total = GuardrailService(db).list_policies(tenant_id, page, page_size)
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@router.get("/guardrails/policies/{policy_id}")
def get_policy(
    policy_id: str,
    tenant_id: TenantIdDep,
    _: User = Depends(require_permission("security:read")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return GuardrailService(db).get_policy(tenant_id, policy_id)


@router.put("/guardrails/policies/{policy_id}")
def update_policy(
    policy_id: str,
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    user: User = Depends(require_permission("security:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return GuardrailService(db).update_policy(tenant_id, user.id, policy_id, dict(payload))


@router.delete("/guardrails/policies/{policy_id}")
def delete_policy(
    policy_id: str,
    tenant_id: TenantIdDep,
    user: User = Depends(require_permission("security:write")),
    db: Session = Depends(get_db),
) -> dict[str, str]:
    return GuardrailService(db).delete_policy(tenant_id, user.id, policy_id)


@router.post("/guardrails/policies/{policy_id}/rules")
def create_rule(
    policy_id: str,
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    user: User = Depends(require_permission("security:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return GuardrailService(db).create_rule(tenant_id, user.id, policy_id, dict(payload))


@router.get("/guardrails/policies/{policy_id}/rules")
def list_rules(
    policy_id: str,
    tenant_id: TenantIdDep,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    _: User = Depends(require_permission("security:read")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    items, total = GuardrailService(db).list_rules(tenant_id, policy_id, page, page_size)
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@router.post("/guardrails/check-input")
def check_input(
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    user: User = Depends(require_permission("security:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return GuardrailService(db).check_input(tenant_id, user.id, dict(payload))


@router.post("/guardrails/check-output")
def check_output(
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    user: User = Depends(require_permission("security:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return GuardrailService(db).check_output(tenant_id, user.id, dict(payload))


@router.get("/guardrails/executions")
def list_executions(
    tenant_id: TenantIdDep,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    agent_id: str = "",
    _: User = Depends(require_permission("security:read")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    filters = {"agent_id": agent_id} if agent_id else None
    items, total = GuardrailService(db).list_executions(tenant_id, page, page_size, filters)
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@router.get("/guardrails/violations")
def list_violations(
    tenant_id: TenantIdDep,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    agent_id: str = "",
    _: User = Depends(require_permission("security:read")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    filters = {"agent_id": agent_id} if agent_id else None
    items, total = GuardrailService(db).list_violations(tenant_id, page, page_size, filters)
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@router.get("/guardrails/stats")
def guardrail_stats(
    tenant_id: TenantIdDep,
    _: User = Depends(require_permission("security:read")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return GuardrailService(db).get_stats(tenant_id)
