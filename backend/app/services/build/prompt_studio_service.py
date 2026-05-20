from __future__ import annotations

import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.services._shared.resource_service import ResourceService


class PromptStudioService:
    def __init__(self, db: Session):
        self.db = db
        self.resources = ResourceService(db)

    def create_template(self, tenant_id: str, user_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        name = str(payload.get("name", ""))
        content = str(payload.get("content", ""))
        variables = self._extract_variables(content)
        template = self.resources.create("prompt_templates", tenant_id, user_id, {
            "name": name,
            "code": f"pt-{uuid.uuid4().hex[:8]}",
            "status": "active",
            "spec": {
                "content": content,
                "description": str(payload.get("description", "")),
                "category": str(payload.get("category", "general")),
                "tags": payload.get("tags", []),
                "variables": variables,
                "default_model_id": str(payload.get("default_model_id", "")),
                "default_parameters": payload.get("default_parameters", {
                    "temperature": 0.7,
                    "max_tokens": 4096,
                    "top_p": 1.0,
                }),
                "version": 1,
                "usage_count": 0,
                "last_used_at": None,
            },
        })
        for var in variables:
            self.resources.create("prompt_variables", tenant_id, user_id, {
                "name": var["name"],
                "code": f"pv-{uuid.uuid4().hex[:8]}",
                "status": "active",
                "parent_id": template.id,
                "spec": var,
            })
        return ResourceService.to_dict(template)

    def update_template(self, tenant_id: str, user_id: str, template_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        existing = self.resources.get("prompt_templates", tenant_id, template_id)
        old_spec = dict(existing.spec or {})

        if "content" in payload:
            new_content = str(payload["content"])
            variables = self._extract_variables(new_content)
            old_spec["content"] = new_content
            old_spec["variables"] = variables
            old_spec["version"] = old_spec.get("version", 1) + 1

            self.resources.create("prompt_template_versions", tenant_id, user_id, {
                "name": f"v{old_spec['version']}",
                "code": f"ptv-{uuid.uuid4().hex[:8]}",
                "status": "active",
                "parent_id": template_id,
                "version": str(old_spec["version"]),
                "spec": {
                    "content": new_content,
                    "variables": variables,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "change_note": str(payload.get("change_note", "")),
                },
            })

        for key in ("name", "description", "category", "tags", "default_model_id", "default_parameters"):
            if key in payload:
                if key == "name":
                    pass
                else:
                    old_spec[key] = payload[key]

        update_data: dict[str, Any] = {"spec": old_spec}
        if "name" in payload:
            update_data["name"] = payload["name"]
        row = self.resources.update("prompt_templates", tenant_id, user_id, template_id, update_data)
        return ResourceService.to_dict(row)

    def delete_template(self, tenant_id: str, user_id: str, template_id: str) -> dict[str, str]:
        return self.resources.delete("prompt_templates", tenant_id, user_id, template_id)

    def get_template(self, tenant_id: str, template_id: str) -> dict[str, Any]:
        return ResourceService.to_dict(self.resources.get("prompt_templates", tenant_id, template_id))

    def list_templates(self, tenant_id: str, page: int, page_size: int, filters: dict[str, Any] | None = None) -> tuple[list[dict[str, Any]], int]:
        rows, total = self.resources.list("prompt_templates", tenant_id, page, page_size, filters)
        return [ResourceService.to_dict(row) for row in rows], total

    def list_versions(self, tenant_id: str, template_id: str, page: int, page_size: int) -> tuple[list[dict[str, Any]], int]:
        rows, total = self.resources.list("prompt_template_versions", tenant_id, page, page_size, {"parent_id": template_id})
        return [ResourceService.to_dict(row) for row in rows], total

    def render_template(self, tenant_id: str, template_id: str, variables: dict[str, str]) -> dict[str, Any]:
        template = self.resources.get("prompt_templates", tenant_id, template_id)
        spec = dict(template.spec or {})
        content = spec.get("content", "")
        rendered = content
        for key, value in variables.items():
            rendered = rendered.replace(f"{{{{{key}}}}}", str(value))
        missing = re.findall(r"\{\{(\w+)\}\}", rendered)
        return {"rendered": rendered, "missing_variables": missing, "original": content}

    async def playground_run(self, tenant_id: str, user_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        template_id = str(payload.get("template_id", ""))
        variables = payload.get("variables", {})
        model_id = str(payload.get("model_id", ""))
        parameters = payload.get("parameters", {})
        raw_prompt = str(payload.get("raw_prompt", ""))

        if template_id:
            render_result = self.render_template(tenant_id, template_id, variables)
            prompt = render_result["rendered"]
            if not model_id:
                template = self.resources.get("prompt_templates", tenant_id, template_id)
                model_id = str((template.spec or {}).get("default_model_id") or "")
        else:
            prompt = raw_prompt

        start_time = time.time()
        response = await self._execute_prompt(tenant_id, user_id, prompt, model_id, parameters)
        latency_ms = int((time.time() - start_time) * 1000)
        token_usage = self._estimate_tokens(prompt, response)

        run = self.resources.create("prompt_playground_runs", tenant_id, user_id, {
            "name": f"playground-{uuid.uuid4().hex[:6]}",
            "code": f"ppr-{uuid.uuid4().hex[:8]}",
            "status": "completed",
            "parent_id": template_id,
            "model_id": model_id,
            "spec": {
                "prompt": prompt,
                "response": response,
                "variables": variables,
                "parameters": parameters,
                "model_id": model_id,
                "template_id": template_id,
                "latency_ms": latency_ms,
                "token_usage": token_usage,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        })

        if template_id:
            tmpl = self.resources.get("prompt_templates", tenant_id, template_id)
            spec = dict(tmpl.spec or {})
            spec["usage_count"] = spec.get("usage_count", 0) + 1
            spec["last_used_at"] = datetime.now(timezone.utc).isoformat()
            self.resources.update("prompt_templates", tenant_id, user_id, template_id, {"spec": spec})

        return {
            "run_id": run.id,
            "prompt": prompt,
            "response": response,
            "latency_ms": latency_ms,
            "token_usage": token_usage,
            "model_id": model_id,
        }

    def create_ab_test(self, tenant_id: str, user_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        name = str(payload.get("name", ""))
        variant_a = payload.get("variant_a", {})
        variant_b = payload.get("variant_b", {})
        test_cases = payload.get("test_cases", [])
        model_id = str(payload.get("model_id", ""))

        ab_test = self.resources.create("prompt_ab_tests", tenant_id, user_id, {
            "name": name,
            "code": f"pab-{uuid.uuid4().hex[:8]}",
            "status": "running",
            "model_id": model_id,
            "spec": {
                "variant_a": variant_a,
                "variant_b": variant_b,
                "test_cases": test_cases,
                "model_id": model_id,
                "total_cases": len(test_cases),
                "completed_cases": 0,
                "started_at": datetime.now(timezone.utc).isoformat(),
            },
        })
        return ResourceService.to_dict(ab_test)

    async def run_ab_test(self, tenant_id: str, user_id: str, test_id: str) -> dict[str, Any]:
        test = self.resources.get("prompt_ab_tests", tenant_id, test_id)
        spec = dict(test.spec or {})
        variant_a = spec.get("variant_a", {})
        variant_b = spec.get("variant_b", {})
        test_cases = spec.get("test_cases", [])
        model_id = spec.get("model_id", "")

        results = []
        for i, case in enumerate(test_cases):
            variables = case.get("variables", {})
            expected = case.get("expected", "")

            prompt_a = self._render_inline(variant_a.get("content", ""), variables)
            prompt_b = self._render_inline(variant_b.get("content", ""), variables)

            start_a = time.time()
            response_a = await self._execute_prompt(tenant_id, user_id, prompt_a, model_id, {})
            latency_a = int((time.time() - start_a) * 1000)

            start_b = time.time()
            response_b = await self._execute_prompt(tenant_id, user_id, prompt_b, model_id, {})
            latency_b = int((time.time() - start_b) * 1000)

            score_a = self._score_response(response_a, expected)
            score_b = self._score_response(response_b, expected)

            result = {
                "case_index": i,
                "variables": variables,
                "expected": expected,
                "variant_a": {"prompt": prompt_a, "response": response_a, "latency_ms": latency_a, "score": score_a},
                "variant_b": {"prompt": prompt_b, "response": response_b, "latency_ms": latency_b, "score": score_b},
                "winner": "a" if score_a > score_b else ("b" if score_b > score_a else "tie"),
            }
            results.append(result)

            self.resources.create("prompt_ab_results", tenant_id, user_id, {
                "name": f"result-{i}",
                "code": f"par-{uuid.uuid4().hex[:8]}",
                "status": "completed",
                "parent_id": test_id,
                "spec": result,
            })

        a_wins = sum(1 for r in results if r["winner"] == "a")
        b_wins = sum(1 for r in results if r["winner"] == "b")
        ties = sum(1 for r in results if r["winner"] == "tie")
        avg_score_a = sum(r["variant_a"]["score"] for r in results) / len(results) if results else 0
        avg_score_b = sum(r["variant_b"]["score"] for r in results) / len(results) if results else 0

        spec["completed_cases"] = len(results)
        spec["finished_at"] = datetime.now(timezone.utc).isoformat()
        spec["summary"] = {
            "a_wins": a_wins, "b_wins": b_wins, "ties": ties,
            "avg_score_a": avg_score_a, "avg_score_b": avg_score_b,
            "winner": "a" if a_wins > b_wins else ("b" if b_wins > a_wins else "tie"),
        }
        self.resources.update("prompt_ab_tests", tenant_id, user_id, test_id, {"status": "completed", "spec": spec})

        return {"test_id": test_id, "results": results, "summary": spec["summary"]}

    def list_ab_tests(self, tenant_id: str, page: int, page_size: int) -> tuple[list[dict[str, Any]], int]:
        rows, total = self.resources.list("prompt_ab_tests", tenant_id, page, page_size)
        return [ResourceService.to_dict(row) for row in rows], total

    def get_ab_test(self, tenant_id: str, test_id: str) -> dict[str, Any]:
        test = ResourceService.to_dict(self.resources.get("prompt_ab_tests", tenant_id, test_id))
        results, _ = self.resources.list("prompt_ab_results", tenant_id, 1, 200, {"parent_id": test_id})
        test["results"] = [ResourceService.to_dict(r) for r in results]
        return test

    async def _execute_prompt(self, tenant_id: str, user_id: str, prompt: str, model_id: str, parameters: dict[str, Any]) -> str:
        if not model_id:
            from app.core.constants import ErrorCode
            from app.core.errors import AppError

            raise AppError(ErrorCode.VALIDATION_ERROR, "model_id is required for prompt playground execution", 422)
        from app.services.settings.model_services import ModelInvocationService

        result = await ModelInvocationService(self.db).invoke(
            tenant_id,
            user_id,
            model_id,
            "chat_llm",
            {"messages": [{"role": "user", "content": prompt}], **parameters},
        )
        return str((result.get("result") or {}).get("content") or "")

    def _render_inline(self, content: str, variables: dict[str, str]) -> str:
        rendered = content
        for key, value in variables.items():
            rendered = rendered.replace(f"{{{{{key}}}}}", str(value))
        return rendered

    @staticmethod
    def _extract_variables(content: str) -> list[dict[str, Any]]:
        matches = re.findall(r"\{\{(\w+)\}\}", content)
        seen = set()
        variables = []
        for name in matches:
            if name not in seen:
                seen.add(name)
                variables.append({
                    "name": name,
                    "type": "string",
                    "required": True,
                    "default": "",
                    "description": "",
                })
        return variables

    @staticmethod
    def _estimate_tokens(prompt: str, response: str) -> dict[str, int]:
        input_tokens = max(1, len(prompt) // 4)
        output_tokens = max(1, len(response) // 4)
        return {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": input_tokens + output_tokens}

    @staticmethod
    def _score_response(response: str, expected: str) -> float:
        if not expected:
            return 0.5
        response_lower = response.lower()
        expected_lower = expected.lower()
        if expected_lower in response_lower:
            return 1.0
        expected_words = set(expected_lower.split())
        response_words = set(response_lower.split())
        if not expected_words:
            return 0.5
        overlap = len(expected_words & response_words) / len(expected_words)
        return round(overlap, 2)
