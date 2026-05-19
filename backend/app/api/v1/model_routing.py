from __future__ import annotations

from fastapi import APIRouter, Body, Depends
from sqlalchemy.orm import Session

from app.api.deps import TenantIdDep, get_db, require_permission
from app.api.v1.domain_router import add_action_route, add_crud_routes, add_list_route
from app.models.core import User
from app.services.model_services import ModelInvocationService, ModelRoutingService

router = APIRouter(tags=["model-routing"])

add_crud_routes(router, table="model_routing_policies", prefix="/model-routing-policies", permission="model-routing")

for method, path, table, action, output in [
    ("post", "/model-routing-policies/{policy_id}/rules", "model_routing_policies", "create_rule", "model_route_rules"),
    ("get", "/model-routing-policies/{policy_id}/rules", "model_route_rules", "rules", None),
    ("put", "/model-route-rules/{rule_id}", "model_route_rules", "update_rule", None),
    ("delete", "/model-route-rules/{rule_id}", "model_route_rules", "delete_rule", None),
    ("post", "/model-routing-policies/{policy_id}/fallback-chain", "model_routing_policies", "fallback_chain", "model_fallback_chains"),
    ("get", "/model-routing-policies/{policy_id}/fallback-chain", "model_fallback_chains", "fallback_chain", None),
]:
    if method == "get":
        add_list_route(router, method=method, path=path, table=table, permission="model-routing")
    else:
        add_action_route(router, method=method, path=path, table=table, permission="model-routing", action=action, output_table=output)


@router.post("/models/route")
def route_model(tenant_id: TenantIdDep, payload: dict[str, object] = Body(default_factory=dict), user: User = Depends(require_permission("model-routing:write")), db: Session = Depends(get_db)) -> dict[str, object]:
    return ModelRoutingService(db).route(tenant_id, user.id, dict(payload))


@router.post("/models/invoke")
async def invoke_model(tenant_id: TenantIdDep, payload: dict[str, object] = Body(default_factory=dict), user: User = Depends(require_permission("model-routing:write")), db: Session = Depends(get_db)) -> dict[str, object]:
    route = ModelRoutingService(db).route(tenant_id, user.id, dict(payload))
    return await ModelInvocationService(db).invoke(tenant_id, user.id, str(route["model_id"]), str(route["model_type"]), dict(payload))


@router.post("/model-circuit-breakers/{model_id}/reset")
def reset_model_circuit_breaker(model_id: str, tenant_id: TenantIdDep, user: User = Depends(require_permission("model-routing:write")), db: Session = Depends(get_db)) -> dict[str, object]:
    return ModelRoutingService(db).reset_circuit_breaker(tenant_id, user.id, model_id)


for path, table in [
    ("/model-selection-logs", "model_selection_logs"),
    ("/model-quota-usage", "model_quota_usage"),
    ("/model-latency-stats", "model_latency_stats"),
]:
    add_list_route(router, method="get", path=path, table=table, permission="model-routing")
