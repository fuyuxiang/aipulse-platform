from __future__ import annotations

from fastapi import APIRouter, Body, Depends
from sqlalchemy.orm import Session

from app.api.deps import TenantIdDep, get_db, require_permission
from app.api.v1.domain_router import add_action_route, add_crud_routes, add_list_route
from app.models.core import User
from app.services.evaluation_service import EvaluationService

router = APIRouter(tags=["evaluation"])

add_crud_routes(router, table="evaluation_datasets", prefix="/evaluation/datasets", permission="evaluation")
add_crud_routes(router, table="bad_cases", prefix="/bad-cases", permission="evaluation")

for method, path, table, action, output in [
    ("post", "/evaluation/datasets/{dataset_id}/cases", "evaluation_datasets", "create_case", "evaluation_cases"),
    ("get", "/evaluation/datasets/{dataset_id}/cases", "evaluation_cases", "cases", None),
    ("get", "/evaluation/runs", "evaluation_runs", "runs", None),
    ("get", "/evaluation/runs/{run_id}", "evaluation_runs", "run", None),
    ("get", "/evaluation/runs/{run_id}/results", "evaluation_results", "results", None),
    ("post", "/evaluation/prompt-compare", "prompt_comparison_runs", "prompt_compare", "prompt_comparison_runs"),
    ("post", "/evaluation/regression", "regression_runs", "regression", "regression_runs"),
    ("put", "/bad-cases/{bad_case_id}/label", "bad_cases", "label", None),
]:
    if method == "get":
        add_list_route(router, method=method, path=path, table=table, permission="evaluation")
    else:
        add_action_route(router, method=method, path=path, table=table, permission="evaluation", action=action, output_table=output)


@router.post("/evaluation/runs")
def run_evaluation(tenant_id: TenantIdDep, payload: dict[str, object] = Body(default_factory=dict), user: User = Depends(require_permission("evaluation:write")), db: Session = Depends(get_db)) -> dict[str, object]:
    return EvaluationService(db).run(tenant_id, user.id, dict(payload))
