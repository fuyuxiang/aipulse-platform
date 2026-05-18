from __future__ import annotations

from fastapi import APIRouter, Body, Depends
from sqlalchemy.orm import Session

from app.api.deps import TenantIdDep, get_db, require_permission
from app.api.v1.domain_router import add_action_route, add_crud_routes, add_list_route
from app.models.core import User
from app.services.model_services import ModelInvocationService

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
    ("get", "/model-providers/{provider_id}/capabilities", "model_provider_capabilities", "list", None),
    ("post", "/model-providers/{provider_id}/credentials", "model_providers", "create_credential", "model_credentials"),
    ("get", "/model-providers/{provider_id}/credentials", "model_credentials", "list_credentials", None),
    ("post", "/model-credentials/{credential_id}/test", "model_credentials", "test", "model_test_records"),
    ("post", "/models/{model_id}/versions", "models", "create_version", "model_versions"),
    ("get", "/models/{model_id}/versions", "model_versions", "list_versions", None),
    ("post", "/models/{model_id}/enable", "models", "enable", None),
    ("post", "/models/{model_id}/disable", "models", "disable", None),
    ("post", "/models/{model_id}/health-check", "models", "health_check", "model_health_checks"),
    ("get", "/models/{model_id}/health", "model_health_checks", "health", None),
]:
    if method == "get":
        add_list_route(router, method=method, path=path, table=table, permission="models")
    else:
        add_action_route(router, method=method, path=path, table=table, permission="models", action=action, output_table=output)


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
