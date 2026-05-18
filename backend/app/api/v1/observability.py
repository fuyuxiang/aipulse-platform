from __future__ import annotations

from fastapi import APIRouter, Body, Depends
from sqlalchemy.orm import Session

from app.api.deps import TenantIdDep, get_db, require_permission
from app.api.v1.domain_router import add_crud_routes, add_list_route
from app.models.core import User
from app.services.resource_service import ResourceService

router = APIRouter(tags=["observability"])

for table, prefix in [
    ("runtime_metrics", "/observability/metrics"),
    ("agent_run_logs", "/observability/logs"),
    ("trace_spans", "/observability/traces"),
    ("alert_rules", "/alert-rules"),
]:
    add_crud_routes(router, table=table, prefix=prefix, permission="observability")


@router.get("/observability/dashboard")
def dashboard(tenant_id: TenantIdDep, db: Session = Depends(get_db), current_user: User = Depends(require_permission("observability:read"))) -> dict[str, object]:
    service = ResourceService(db)
    summary = {}
    for key, table in {
        "agents": "agents",
        "agent_runs": "agent_run_records",
        "workflow_runs": "workflow_runs",
        "model_calls": "model_call_logs",
        "tool_calls": "tool_call_logs",
        "rag_retrievals": "knowledge_retrieval_logs",
        "alerts": "alert_events",
        "bad_cases": "bad_cases",
    }.items():
        rows, total = service.list(table, tenant_id, 1, 1)
        assert current_user.id or rows == []
        summary[key] = total
    return {"summary": summary}


@router.get("/observability/traces/{trace_id}")
def trace_detail(trace_id: str, tenant_id: TenantIdDep, db: Session = Depends(get_db), _: User = Depends(require_permission("observability:read"))) -> dict[str, object]:
    rows, total = ResourceService(db).list("trace_spans", tenant_id, 1, 200, {"trace_id": trace_id})
    return {"trace_id": trace_id, "total": total, "spans": [ResourceService.to_dict(row) for row in rows]}


@router.post("/observability/traces")
def write_trace(tenant_id: TenantIdDep, payload: dict[str, object] = Body(default_factory=dict), db: Session = Depends(get_db), user: User = Depends(require_permission("observability:write"))) -> dict[str, object]:
    row = ResourceService(db).create("trace_spans", tenant_id, user.id, {"name": str(payload.get("name", "span")), "trace_id": str(payload.get("trace_id", "")), "spec": payload})
    return ResourceService.to_dict(row)


@router.get("/observability/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@router.get("/observability/runtime-status")
def runtime_status(tenant_id: TenantIdDep, db: Session = Depends(get_db), _: User = Depends(require_permission("observability:read"))) -> dict[str, object]:
    rows, total = ResourceService(db).list("agent_runtime_instances", tenant_id, 1, 200)
    return {"total": total, "instances": [ResourceService.to_dict(row) for row in rows]}


for path, table in [
    ("/alert-events", "alert_events"),
]:
    add_list_route(router, method="get", path=path, table=table, permission="observability")
