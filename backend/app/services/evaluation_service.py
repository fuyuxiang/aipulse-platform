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
        for result in results:
            self.resources.create(
                "evaluation_results",
                tenant_id,
                user_id,
                {
                    "name": f"case {result['case_index']}",
                    "status": "passed" if float(result["score"]) >= float(payload.get("pass_score", 1.0)) else "failed",
                    "parent_id": run.id,
                    "model_id": str(payload.get("model_id") or ""),
                    "model_type": str(payload.get("model_type") or ""),
                    "latency_ms": int(result["latency_ms"]),
                    "cost": float(result["cost"]),
                    "spec": result,
                },
            )
        self.resources.create("evaluation_metrics", tenant_id, user_id, {"name": "evaluation metrics", "parent_id": run.id, "spec": metrics})
        return {"run_id": run.id, "metrics": metrics, "results": results}

    def prompt_compare(self, tenant_id: str, user_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        prompts = [str(item) for item in payload.get("prompts", [])]
        cases = list(payload.get("cases", []))
        scores: list[dict[str, Any]] = []
        for prompt in prompts:
            score = sum(1 for case in cases if str(case.get("expected", "")).lower() in f"{prompt} {case.get('actual', case.get('input', ''))}".lower())
            scores.append({"prompt": prompt, "score": score / max(1, len(cases))})
        winner = max(scores, key=self._score_value) if scores else {"prompt": "", "score": 0.0}
        row = self.resources.create(
            "prompt_comparison_runs",
            tenant_id,
            user_id,
            {"name": payload.get("name", "prompt comparison"), "status": "completed", "input_payload": payload, "output_payload": {"scores": scores, "winner": winner}},
        )
        return {"run_id": row.id, "scores": scores, "winner": winner}

    def regression(self, tenant_id: str, user_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        baseline = {str(item.get("id") or index): item for index, item in enumerate(payload.get("baseline", []))}
        current = {str(item.get("id") or index): item for index, item in enumerate(payload.get("current", []))}
        regressions = []
        for case_id, base in baseline.items():
            current_case = current.get(case_id, {})
            base_score = float(base.get("score", 0))
            current_score = float(current_case.get("score", 0))
            if current_score < base_score:
                regressions.append({"case_id": case_id, "baseline": base_score, "current": current_score, "delta": current_score - base_score})
        row = self.resources.create(
            "regression_runs",
            tenant_id,
            user_id,
            {"name": payload.get("name", "regression run"), "status": "failed" if regressions else "passed", "input_payload": payload, "output_payload": {"regressions": regressions}},
        )
        return {"run_id": row.id, "status": row.status, "regressions": regressions}

    @staticmethod
    def _score_value(item: dict[str, Any]) -> float:
        return float(item.get("score", 0.0))
