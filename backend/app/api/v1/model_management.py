from __future__ import annotations

from fastapi import APIRouter, Body, Depends
from sqlalchemy.orm import Session

from app.api.deps import TenantIdDep, get_db, require_permission
from app.api.v1.domain_router import add_action_route, add_crud_routes, add_list_route
from app.models.core import User
from app.schemas.common import ResourceRead
from app.services.model_services import ModelInvocationService, ModelManagementService
from app.services.resource_service import ResourceService

router = APIRouter(tags=["model-management"])

for table, prefix in [
    ("model_providers", "/model-providers"),
    ("model_credentials", "/model-credentials"),
    ("model_endpoints", "/model-endpoints"),
    ("models", "/models"),
    ("model_access_policies", "/model-access-policies"),
    ("model_quota", "/model-quotas"),
    ("model_rate_limits", "/model-rate-limits"),
]:
    add_crud_routes(router, table=table, prefix=prefix, permission="models")

for method, path, table, action, output in [
    ("get", "/model-providers/{provider_id}/credentials", "model_credentials", "list_credentials", None),
    ("get", "/models/{model_id}/versions", "model_versions", "list_versions", None),
]:
    if method == "get":
        add_list_route(router, method=method, path=path, table=table, permission="models")
    else:
        add_action_route(router, method=method, path=path, table=table, permission="models", action=action, output_table=output)


@router.get("/model-providers/{provider_id}/capabilities")
def provider_capabilities(provider_id: str, tenant_id: TenantIdDep, user: User = Depends(require_permission("models:read")), db: Session = Depends(get_db)) -> dict[str, object]:
    return ModelManagementService(db).provider_capabilities(tenant_id, user.id, provider_id)


@router.post("/model-providers/{provider_id}/credentials")
def create_provider_credential(
    provider_id: str,
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    user: User = Depends(require_permission("models:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return ModelManagementService(db).create_credential(tenant_id, user.id, provider_id, dict(payload))


@router.post("/model-credentials/{credential_id}/test")
def test_credential(credential_id: str, tenant_id: TenantIdDep, user: User = Depends(require_permission("models:write")), db: Session = Depends(get_db)) -> dict[str, object]:
    return ModelManagementService(db).test_credential(tenant_id, user.id, credential_id)


@router.post("/models/{model_id}/versions")
def create_model_version(
    model_id: str,
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    user: User = Depends(require_permission("models:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return ModelManagementService(db).create_model_version(tenant_id, user.id, model_id, dict(payload))


@router.post("/models/{model_id}/enable", response_model=ResourceRead)
def enable_model(model_id: str, tenant_id: TenantIdDep, user: User = Depends(require_permission("models:write")), db: Session = Depends(get_db)) -> ResourceRead:
    row = ResourceService(db).update("models", tenant_id, user.id, model_id, {"enabled": True, "status": "active"})
    return ResourceRead.model_validate(ResourceService.to_dict(row))


@router.post("/models/{model_id}/disable", response_model=ResourceRead)
def disable_model(model_id: str, tenant_id: TenantIdDep, user: User = Depends(require_permission("models:write")), db: Session = Depends(get_db)) -> ResourceRead:
    row = ResourceService(db).update("models", tenant_id, user.id, model_id, {"enabled": False, "status": "disabled"})
    return ResourceRead.model_validate(ResourceService.to_dict(row))


@router.post("/models/{model_id}/health-check")
def health_check_model(model_id: str, tenant_id: TenantIdDep, user: User = Depends(require_permission("models:write")), db: Session = Depends(get_db)) -> dict[str, object]:
    return ModelManagementService(db).health_check(tenant_id, user.id, model_id)


@router.get("/models/{model_id}/health")
def model_health(model_id: str, tenant_id: TenantIdDep, _: User = Depends(require_permission("models:read")), db: Session = Depends(get_db)) -> dict[str, object]:
    return ModelManagementService(db).latest_health(tenant_id, model_id)


@router.post("/models/{model_id}/test-chat")
async def test_chat(model_id: str, tenant_id: TenantIdDep, payload: dict[str, object] = Body(default_factory=dict), user: User = Depends(require_permission("models:write")), db: Session = Depends(get_db)) -> dict[str, object]:
    return await ModelInvocationService(db).invoke(tenant_id, user.id, model_id, "chat_llm", dict(payload))


@router.post("/models/{model_id}/test-vision")
async def test_vision(model_id: str, tenant_id: TenantIdDep, payload: dict[str, object] = Body(default_factory=dict), user: User = Depends(require_permission("models:write")), db: Session = Depends(get_db)) -> dict[str, object]:
    return await ModelInvocationService(db).invoke(tenant_id, user.id, model_id, "vision_language", dict(payload))


@router.post("/models/{model_id}/test-embedding")
async def test_embedding(model_id: str, tenant_id: TenantIdDep, payload: dict[str, object] = Body(default_factory=dict), user: User = Depends(require_permission("models:write")), db: Session = Depends(get_db)) -> dict[str, object]:
    return await ModelInvocationService(db).invoke(tenant_id, user.id, model_id, "embedding", dict(payload))


@router.post("/models/{model_id}/test-rerank")
async def test_rerank(model_id: str, tenant_id: TenantIdDep, payload: dict[str, object] = Body(default_factory=dict), user: User = Depends(require_permission("models:write")), db: Session = Depends(get_db)) -> dict[str, object]:
    return await ModelInvocationService(db).invoke(tenant_id, user.id, model_id, "rerank", dict(payload))


@router.post("/models/{model_id}/test-moderation")
async def test_moderation(model_id: str, tenant_id: TenantIdDep, payload: dict[str, object] = Body(default_factory=dict), user: User = Depends(require_permission("models:write")), db: Session = Depends(get_db)) -> dict[str, object]:
    return await ModelInvocationService(db).invoke(tenant_id, user.id, model_id, "moderation", dict(payload))


for path, table in [
    ("/model-call-logs", "model_call_logs"),
    ("/model-cost-stats", "model_cost_stats"),
    ("/model-test-records", "model_test_records"),
]:
    add_list_route(router, method="get", path=path, table=table, permission="models")
