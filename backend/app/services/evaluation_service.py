from __future__ import annotations

from statistics import mean
from typing import Any

from sqlalchemy.orm import Session

from app.services.resource_service import ResourceService


class EvaluationService:
    def __init__(self, db: Session):
        self.resources = ResourceService(db)

    def run(self, tenant_id: str, user_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        cases = payload.get("cases", [])
        results = []
        for index, case in enumerate(cases):
            expected = str(case.get("expected", "")).strip().lower()
            actual = str(case.get("actual", case.get("input", ""))).strip().lower()
            score = 1.0 if expected and expected in actual else float(case.get("score", 0.0))
            results.append({"case_index": index, "score": score, "latency_ms": int(case.get("latency_ms", 0)), "cost": float(case.get("cost", 0))})
        metrics = {
            "accuracy": mean([item["score"] for item in results]) if results else 0.0,
            "latency_ms": mean([item["latency_ms"] for item in results]) if results else 0.0,
            "cost": sum(item["cost"] for item in results),
            "failure_rate": 1.0 - (mean([item["score"] for item in results]) if results else 0.0),
        }
        run = self.resources.create(
            "evaluation_runs",
            tenant_id,
            user_id,
            {"name": payload.get("name", "evaluation run"), "status": "completed", "input_payload": payload, "output_payload": {"metrics": metrics, "results": results}},
        )
        self.resources.create("evaluation_metrics", tenant_id, user_id, {"name": "evaluation metrics", "parent_id": run.id, "spec": metrics})
        return {"run_id": run.id, "metrics": metrics, "results": results}

