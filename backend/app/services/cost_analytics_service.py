from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.services.resource_service import ResourceService


class CostAnalyticsService:
    def __init__(self, db: Session):
        self.db = db
        self.resources = ResourceService(db)

    def record_cost(self, tenant_id: str, user_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        agent_id = str(payload.get("agent_id", ""))
        model_id = str(payload.get("model_id", ""))
        token_usage = payload.get("token_usage", {})
        input_tokens = int(token_usage.get("input_tokens", 0))
        output_tokens = int(token_usage.get("output_tokens", 0))
        total_tokens = input_tokens + output_tokens

        input_price = float(payload.get("input_price_per_1k", 0.01))
        output_price = float(payload.get("output_price_per_1k", 0.03))
        cost = (input_tokens / 1000 * input_price) + (output_tokens / 1000 * output_price)

        record = self.resources.create("cost_records", tenant_id, user_id, {
            "name": f"cost-{uuid.uuid4().hex[:6]}",
            "code": f"cr-{uuid.uuid4().hex[:8]}",
            "status": "recorded",
            "agent_id": agent_id,
            "model_id": model_id,
            "user_id": user_id,
            "cost": cost,
            "token_usage": token_usage,
            "spec": {
                "agent_id": agent_id,
                "model_id": model_id,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "cost": cost,
                "input_price_per_1k": input_price,
                "output_price_per_1k": output_price,
                "session_id": str(payload.get("session_id", "")),
                "run_id": str(payload.get("run_id", "")),
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            },
        })
        self._check_budget(tenant_id, user_id, agent_id, cost)
        return ResourceService.to_dict(record)

    def get_summary(self, tenant_id: str, filters: dict[str, Any] | None = None) -> dict[str, Any]:
        rows, total = self.resources.list("cost_records", tenant_id, 1, 10000, filters)
        total_cost = 0.0
        total_input_tokens = 0
        total_output_tokens = 0
        by_agent: dict[str, float] = {}
        by_model: dict[str, float] = {}
        by_user: dict[str, float] = {}
        daily: dict[str, float] = {}

        for row in rows:
            spec = row.spec or {}
            cost = float(spec.get("cost", 0))
            total_cost += cost
            total_input_tokens += int(spec.get("input_tokens", 0))
            total_output_tokens += int(spec.get("output_tokens", 0))

            agent_id = spec.get("agent_id", "unknown")
            by_agent[agent_id] = by_agent.get(agent_id, 0) + cost

            model_id = spec.get("model_id", "unknown")
            by_model[model_id] = by_model.get(model_id, 0) + cost

            uid = row.user_id or "unknown"
            by_user[uid] = by_user.get(uid, 0) + cost

            recorded_at = str(spec.get("recorded_at", ""))[:10]
            if recorded_at:
                daily[recorded_at] = daily.get(recorded_at, 0) + cost

        return {
            "total_cost": round(total_cost, 4),
            "total_records": total,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
            "by_agent": [{"agent_id": k, "cost": round(v, 4)} for k, v in sorted(by_agent.items(), key=lambda x: -x[1])[:20]],
            "by_model": [{"model_id": k, "cost": round(v, 4)} for k, v in sorted(by_model.items(), key=lambda x: -x[1])[:20]],
            "by_user": [{"user_id": k, "cost": round(v, 4)} for k, v in sorted(by_user.items(), key=lambda x: -x[1])[:20]],
            "daily_trend": [{"date": k, "cost": round(v, 4)} for k, v in sorted(daily.items())[-30:]],
        }

    def get_agent_cost(self, tenant_id: str, agent_id: str) -> dict[str, Any]:
        rows, total = self.resources.list("cost_records", tenant_id, 1, 10000, {"agent_id": agent_id})
        total_cost = sum(float((row.spec or {}).get("cost", 0)) for row in rows)
        total_tokens = sum(int((row.spec or {}).get("total_tokens", 0)) for row in rows)
        return {
            "agent_id": agent_id,
            "total_cost": round(total_cost, 4),
            "total_records": total,
            "total_tokens": total_tokens,
        }

    def create_budget(self, tenant_id: str, user_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        budget = self.resources.create("cost_budgets", tenant_id, user_id, {
            "name": str(payload.get("name", "")),
            "code": f"cb-{uuid.uuid4().hex[:8]}",
            "status": "active",
            "agent_id": str(payload.get("agent_id", "")),
            "spec": {
                "scope": str(payload.get("scope", "tenant")),
                "agent_id": str(payload.get("agent_id", "")),
                "user_id": str(payload.get("user_id", "")),
                "period": str(payload.get("period", "monthly")),
                "limit_amount": float(payload.get("limit_amount", 100.0)),
                "warning_threshold": float(payload.get("warning_threshold", 0.8)),
                "action_on_exceed": str(payload.get("action_on_exceed", "alert")),
                "current_usage": 0.0,
                "period_start": datetime.now(timezone.utc).isoformat(),
            },
        })
        return ResourceService.to_dict(budget)

    def update_budget(self, tenant_id: str, user_id: str, budget_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        row = self.resources.update("cost_budgets", tenant_id, user_id, budget_id, payload)
        return ResourceService.to_dict(row)

    def delete_budget(self, tenant_id: str, user_id: str, budget_id: str) -> dict[str, str]:
        return self.resources.delete("cost_budgets", tenant_id, user_id, budget_id)

    def list_budgets(self, tenant_id: str, page: int, page_size: int) -> tuple[list[dict[str, Any]], int]:
        rows, total = self.resources.list("cost_budgets", tenant_id, page, page_size)
        return [ResourceService.to_dict(row) for row in rows], total

    def list_cost_alerts(self, tenant_id: str, page: int, page_size: int) -> tuple[list[dict[str, Any]], int]:
        rows, total = self.resources.list("cost_alerts", tenant_id, page, page_size)
        return [ResourceService.to_dict(row) for row in rows], total

    def _check_budget(self, tenant_id: str, user_id: str, agent_id: str, cost: float) -> None:
        budgets, _ = self.resources.list("cost_budgets", tenant_id, 1, 100)
        for budget in budgets:
            spec = dict(budget.spec or {})
            scope = spec.get("scope", "tenant")
            if scope == "agent" and spec.get("agent_id") != agent_id:
                continue
            if scope == "user" and spec.get("user_id") != user_id:
                continue

            current = spec.get("current_usage", 0.0) + cost
            spec["current_usage"] = current
            self.resources.update("cost_budgets", tenant_id, user_id, budget.id, {"spec": spec})

            limit_amount = spec.get("limit_amount", 100.0)
            warning_threshold = spec.get("warning_threshold", 0.8)

            if current >= limit_amount:
                self.resources.create("cost_alerts", tenant_id, user_id, {
                    "name": f"budget-exceeded-{budget.name}",
                    "code": f"ca-{uuid.uuid4().hex[:8]}",
                    "status": "triggered",
                    "parent_id": budget.id,
                    "agent_id": agent_id,
                    "spec": {
                        "alert_type": "budget_exceeded",
                        "budget_id": budget.id,
                        "limit": limit_amount,
                        "current": current,
                        "triggered_at": datetime.now(timezone.utc).isoformat(),
                    },
                })
            elif current >= limit_amount * warning_threshold:
                self.resources.create("cost_alerts", tenant_id, user_id, {
                    "name": f"budget-warning-{budget.name}",
                    "code": f"ca-{uuid.uuid4().hex[:8]}",
                    "status": "warning",
                    "parent_id": budget.id,
                    "agent_id": agent_id,
                    "spec": {
                        "alert_type": "budget_warning",
                        "budget_id": budget.id,
                        "limit": limit_amount,
                        "current": current,
                        "threshold": warning_threshold,
                        "triggered_at": datetime.now(timezone.utc).isoformat(),
                    },
                })
