from __future__ import annotations

from fastapi import APIRouter, Body, Depends, Query
from sqlalchemy.orm import Session

from app.api.deps import TenantIdDep, get_db, require_permission
from app.models.core import User
from app.services.prompt_studio_service import PromptStudioService

router = APIRouter(tags=["prompt-studio"])


@router.post("/prompt-templates")
def create_template(
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    user: User = Depends(require_permission("agents:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return PromptStudioService(db).create_template(tenant_id, user.id, dict(payload))


@router.get("/prompt-templates")
def list_templates(
    tenant_id: TenantIdDep,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    category: str = "",
    _: User = Depends(require_permission("agents:read")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    filters = {"category": category} if category else None
    items, total = PromptStudioService(db).list_templates(tenant_id, page, page_size, filters)
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@router.get("/prompt-templates/{template_id}")
def get_template(
    template_id: str,
    tenant_id: TenantIdDep,
    _: User = Depends(require_permission("agents:read")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return PromptStudioService(db).get_template(tenant_id, template_id)


@router.put("/prompt-templates/{template_id}")
def update_template(
    template_id: str,
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    user: User = Depends(require_permission("agents:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return PromptStudioService(db).update_template(tenant_id, user.id, template_id, dict(payload))


@router.delete("/prompt-templates/{template_id}")
def delete_template(
    template_id: str,
    tenant_id: TenantIdDep,
    user: User = Depends(require_permission("agents:write")),
    db: Session = Depends(get_db),
) -> dict[str, str]:
    return PromptStudioService(db).delete_template(tenant_id, user.id, template_id)


@router.get("/prompt-templates/{template_id}/versions")
def list_versions(
    template_id: str,
    tenant_id: TenantIdDep,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    _: User = Depends(require_permission("agents:read")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    items, total = PromptStudioService(db).list_versions(tenant_id, template_id, page, page_size)
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@router.post("/prompt-templates/{template_id}/render")
def render_template(
    template_id: str,
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    _: User = Depends(require_permission("agents:read")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    variables = {k: str(v) for k, v in payload.get("variables", {}).items()}
    return PromptStudioService(db).render_template(tenant_id, template_id, variables)


@router.post("/prompt-playground/run")
async def playground_run(
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    user: User = Depends(require_permission("agents:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return await PromptStudioService(db).playground_run(tenant_id, user.id, dict(payload))


@router.post("/prompt-ab-tests")
def create_ab_test(
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    user: User = Depends(require_permission("evaluation:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return PromptStudioService(db).create_ab_test(tenant_id, user.id, dict(payload))


@router.post("/prompt-ab-tests/{test_id}/run")
async def run_ab_test(
    test_id: str,
    tenant_id: TenantIdDep,
    user: User = Depends(require_permission("evaluation:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return await PromptStudioService(db).run_ab_test(tenant_id, user.id, test_id)


@router.get("/prompt-ab-tests")
def list_ab_tests(
    tenant_id: TenantIdDep,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    _: User = Depends(require_permission("evaluation:read")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    items, total = PromptStudioService(db).list_ab_tests(tenant_id, page, page_size)
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@router.get("/prompt-ab-tests/{test_id}")
def get_ab_test(
    test_id: str,
    tenant_id: TenantIdDep,
    _: User = Depends(require_permission("evaluation:read")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return PromptStudioService(db).get_ab_test(tenant_id, test_id)
