from __future__ import annotations

import asyncio
import hashlib
import math
import time
from datetime import datetime, timezone
from typing import Any, cast

from sqlalchemy.orm import Session

from app.core.constants import MODEL_TYPES, ErrorCode
from app.core.errors import AppError
from app.core.tracing import current_trace_id
from app.services.resource_service import ResourceService


def deterministic_embedding(text: str, dimensions: int = 128) -> list[float]:
    values: list[float] = []
    seed = hashlib.sha256(text.encode("utf-8")).digest()
    counter = 0
    while len(values) < dimensions:
        block = hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
        values.extend(((byte / 127.5) - 1.0) for byte in block)
        counter += 1
    vector = values[:dimensions]
    norm = math.sqrt(sum(item * item for item in vector)) or 1.0
    return [round(item / norm, 8) for item in vector]


class ModelInvocationService:
    def __init__(self, db: Session):
        self.db = db
        self.resources = ResourceService(db)

    async def invoke(self, tenant_id: str, user_id: str, model_id: str, model_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        if model_type not in MODEL_TYPES:
            raise AppError(ErrorCode.VALIDATION_ERROR, f"unsupported model type: {model_type}", 422)
        model = self.resources.get("models", tenant_id, model_id)
        if not model.enabled:
            raise AppError(ErrorCode.BUSINESS_ERROR, "model disabled", 409)
        started = time.perf_counter()
        result = await self._local_invoke(model_type, payload, model.config or {})
        latency_ms = int((time.perf_counter() - started) * 1000)
        usage = result.get("usage", {})
        cost = self._estimate_cost(model.config or {}, usage)
        self.resources.create(
            "model_call_logs",
            tenant_id,
            user_id,
            {
                "name": f"{model_type} call",
                "status": "success",
                "model_id": model_id,
                "model_type": model_type,
                "provider_id": model.provider_id,
                "trace_id": current_trace_id(),
                "latency_ms": latency_ms,
                "cost": cost,
                "token_usage": usage,
                "input_payload": self._summarize(payload),
                "output_payload": self._summarize(result),
            },
        )
        return {"model_id": model_id, "model_type": model_type, "latency_ms": latency_ms, "cost": cost, "result": result}

    async def _local_invoke(self, model_type: str, payload: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        await asyncio.sleep(0)
        if model_type == "embedding":
            texts = payload.get("texts") or [payload.get("text", "")]
            dimensions = int(config.get("embedding_dimensions") or payload.get("dimensions") or 128)
            return {"embeddings": [deterministic_embedding(str(text), dimensions) for text in texts], "usage": {"input_tokens": sum(len(str(t).split()) for t in texts)}}
        if model_type == "rerank":
            query = str(payload.get("query", ""))
            documents = [str(item) for item in payload.get("documents", [])]
            scores = [{"index": idx, "score": self._overlap_score(query, doc), "document": doc} for idx, doc in enumerate(documents)]
            scores.sort(key=lambda item: float(cast(float, item["score"])), reverse=True)
            return {"rankings": scores[: int(payload.get("top_n") or len(scores) or 1)], "usage": {"input_tokens": len(query.split()) + sum(len(doc.split()) for doc in documents)}}
        if model_type == "moderation":
            text = str(payload.get("text", ""))
            risky_terms = set(config.get("risky_terms") or ["secret", "password", "token", "ignore previous"])
            hits = [term for term in risky_terms if term.lower() in text.lower()]
            return {"risk_labels": hits, "risk_score": min(1.0, len(hits) * 0.35), "allowed": not hits, "usage": {"input_tokens": len(text.split())}}
        if model_type == "vision_language":
            return {"content": "vision input accepted", "objects": payload.get("images", []), "usage": {"input_tokens": len(str(payload).split()), "output_tokens": 4}}
        if model_type == "speech_to_text":
            return {"text": payload.get("transcript_hint", ""), "usage": {"input_tokens": 0, "output_tokens": len(str(payload.get("transcript_hint", "")).split())}}
        if model_type == "text_to_speech":
            text = str(payload.get("text", ""))
            return {"audio_ref": f"local://tts/{hashlib.sha1(text.encode()).hexdigest()}.wav", "usage": {"input_tokens": len(text.split())}}
        if model_type == "image_generation":
            prompt = str(payload.get("prompt", ""))
            return {"image_ref": f"local://image/{hashlib.sha1(prompt.encode()).hexdigest()}.png", "usage": {"input_tokens": len(prompt.split())}}
        messages = payload.get("messages") or [{"role": "user", "content": payload.get("prompt", "")}]
        text = "\n".join(str(item.get("content", "")) for item in messages)
        return {"content": f"local {model_type} response: {text[:800]}", "usage": {"input_tokens": len(text.split()), "output_tokens": min(64, max(1, len(text.split()) // 2))}}

    @staticmethod
    def _overlap_score(query: str, document: str) -> float:
        q = {part.lower() for part in query.split() if part}
        d = {part.lower() for part in document.split() if part}
        return round(len(q & d) / max(1, len(q)), 6)

    @staticmethod
    def _estimate_cost(config: dict[str, Any], usage: dict[str, Any]) -> float:
        input_cost = float(config.get("pricing_input_per_1k") or 0) * float(usage.get("input_tokens", 0)) / 1000
        output_cost = float(config.get("pricing_output_per_1k") or 0) * float(usage.get("output_tokens", 0)) / 1000
        return round(input_cost + output_cost, 8)

    @staticmethod
    def _summarize(payload: dict[str, Any]) -> dict[str, Any]:
        text = str(payload)
        return {"summary": text[:500], "size": len(text)}


class ModelManagementService:
    def __init__(self, db: Session):
        self.db = db
        self.resources = ResourceService(db)

    def provider_capabilities(self, tenant_id: str, user_id: str, provider_id: str) -> dict[str, Any]:
        provider = self.resources.get("model_providers", tenant_id, provider_id)
        rows, total = self.resources.list("model_provider_capabilities", tenant_id, 1, 200, {"provider_id": provider.id})
        if total == 0:
            capabilities = sorted(MODEL_TYPES)
            for capability in capabilities:
                self.resources.create(
                    "model_provider_capabilities",
                    tenant_id,
                    user_id,
                    {"name": capability, "status": "active", "provider_id": provider.id, "model_type": capability, "spec": {"provider_type": provider.provider_type}},
                )
            rows, total = self.resources.list("model_provider_capabilities", tenant_id, 1, 200, {"provider_id": provider.id})
        return {"provider_id": provider.id, "total": total, "items": [ResourceService.to_dict(row) for row in rows]}

    def create_credential(self, tenant_id: str, user_id: str, provider_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        provider = self.resources.get("model_providers", tenant_id, provider_id)
        credential = self.resources.create(
            "model_credentials",
            tenant_id,
            user_id,
            {
                "name": str(payload.get("name") or f"{provider.name} credential"),
                "status": "active",
                "provider_id": provider.id,
                "provider_type": provider.provider_type,
                "config": dict(payload.get("config") or {}),
                "spec": dict(payload.get("spec") or payload),
            },
        )
        return ResourceService.to_dict(credential)

    def test_credential(self, tenant_id: str, user_id: str, credential_id: str) -> dict[str, Any]:
        credential = self.resources.get("model_credentials", tenant_id, credential_id)
        spec = credential.spec or {}
        ok = bool(spec.get("secret_ref") or spec.get("secret_sha256") or spec.get("has_secret") or credential.provider_type == "echo_agent_native")
        record = self.resources.create(
            "model_test_records",
            tenant_id,
            user_id,
            {
                "name": "credential test",
                "status": "success" if ok else "failed",
                "provider_id": credential.provider_id,
                "parent_id": credential.id,
                "output_payload": {"credential_id": credential.id, "available": ok, "secret_ref": spec.get("secret_ref", "")},
            },
        )
        return {"credential_id": credential.id, "test_record_id": record.id, "available": ok, "status": record.status}

    def create_model_version(self, tenant_id: str, user_id: str, model_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        model = self.resources.get("models", tenant_id, model_id)
        version = str(payload.get("version") or model.version or datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"))
        row = self.resources.create(
            "model_versions",
            tenant_id,
            user_id,
            {
                "name": f"{model.name} {version}",
                "status": "active",
                "parent_id": model.id,
                "model_id": model.id,
                "model_type": model.model_type,
                "provider_id": model.provider_id,
                "version": version,
                "config": payload.get("config") or model.config,
                "spec": {"model": ResourceService.to_dict(model), "change_summary": payload.get("change_summary", "")},
            },
        )
        return ResourceService.to_dict(row)

    def health_check(self, tenant_id: str, user_id: str, model_id: str) -> dict[str, Any]:
        model = self.resources.get("models", tenant_id, model_id)
        checks = {
            "enabled": bool(model.enabled),
            "provider_id": bool(model.provider_id),
            "model_type": model.model_type in MODEL_TYPES,
            "not_circuit_open": not ModelRoutingService(self.db)._circuit_open(tenant_id, model.id),
        }
        healthy = all(checks.values())
        record = self.resources.create(
            "model_health_checks",
            tenant_id,
            user_id,
            {
                "name": f"health {model.name}",
                "status": "healthy" if healthy else "unhealthy",
                "parent_id": model.id,
                "model_id": model.id,
                "model_type": model.model_type,
                "provider_id": model.provider_id,
                "output_payload": {"healthy": healthy, "checks": checks},
                "finished_at": datetime.now(timezone.utc),
            },
        )
        self.resources.update("models", tenant_id, user_id, model.id, {"status": "active" if healthy else "degraded", "config": {**(model.config or {}), "health_status": record.status}})
        return {"model_id": model.id, "health_check_id": record.id, "healthy": healthy, "checks": checks, "status": record.status}

    def latest_health(self, tenant_id: str, model_id: str) -> dict[str, Any]:
        model = self.resources.get("models", tenant_id, model_id)
        rows, total = self.resources.list("model_health_checks", tenant_id, 1, 1, {"model_id": model.id})
        return {"model_id": model.id, "total": total, "latest": ResourceService.to_dict(rows[0]) if rows else None}


class ModelRoutingService:
    def __init__(self, db: Session):
        self.db = db
        self.resources = ResourceService(db)

    def route(self, tenant_id: str, user_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        model_type = payload.get("model_type", "chat_llm")
        agent_id = payload.get("agent_id", "")
        policy = self._policy(tenant_id, payload)
        strategy = str(payload.get("strategy") or (policy.config or {}).get("strategy") if policy else payload.get("strategy") or "capability_match")
        rows, _ = self.resources.list("models", tenant_id, 1, 500, {"status": "active"})
        candidates = self._filter_candidates(tenant_id, [row for row in rows if row.enabled and (not model_type or row.model_type == model_type)], payload)
        if not candidates:
            raise AppError(ErrorCode.NOT_FOUND, f"no enabled model for type {model_type}", 404)
        selected = self._select(candidates, strategy, payload)
        credential_id = self._rotate_child("model_credentials", tenant_id, selected.provider_id, "credential_id", payload)
        endpoint_id = self._rotate_child("model_endpoints", tenant_id, selected.provider_id, "endpoint_id", payload)
        self.resources.create(
            "model_selection_logs",
            tenant_id,
            user_id,
            {
                "name": "model route",
                "status": "success",
                "model_id": selected.id,
                "model_type": selected.model_type,
                "provider_id": selected.provider_id,
                "agent_id": agent_id,
                "input_payload": payload,
                "output_payload": {"selected_model_id": selected.id, "strategy": strategy, "credential_id": credential_id, "endpoint_id": endpoint_id},
            },
        )
        self.resources.create(
            "model_quota_usage",
            tenant_id,
            user_id,
            {"name": "model route quota usage", "status": "reserved", "model_id": selected.id, "model_type": selected.model_type, "provider_id": selected.provider_id},
        )
        return {
            "model_id": selected.id,
            "model_type": selected.model_type,
            "provider_id": selected.provider_id,
            "credential_id": credential_id,
            "endpoint_id": endpoint_id,
            "strategy": strategy,
            "reason": "policy and capability match",
        }

    def _policy(self, tenant_id: str, payload: dict[str, Any]) -> Any | None:
        policy_id = str(payload.get("policy_id") or "")
        if policy_id:
            return self.resources.get("model_routing_policies", tenant_id, policy_id)
        rows, _ = self.resources.list("model_routing_policies", tenant_id, 1, 200, {"status": "active"})
        for row in rows:
            spec = row.spec or {}
            if spec.get("agent_id") and spec.get("agent_id") != payload.get("agent_id"):
                continue
            if spec.get("workflow_id") and spec.get("workflow_id") != payload.get("workflow_id"):
                continue
            if spec.get("model_type") and spec.get("model_type") != payload.get("model_type"):
                continue
            return row
        return None

    def _filter_candidates(self, tenant_id: str, candidates: list[Any], payload: dict[str, Any]) -> list[Any]:
        filtered = []
        required_context = int(payload.get("min_context_window") or 0)
        input_modality = str(payload.get("input_modality") or "")
        output_modality = str(payload.get("output_modality") or "")
        for row in candidates:
            config = row.config or {}
            if required_context and int(config.get("context_window") or 0) < required_context:
                continue
            if input_modality and input_modality not in set(config.get("modality_input") or [input_modality]):
                continue
            if output_modality and output_modality not in set(config.get("modality_output") or [output_modality]):
                continue
            if payload.get("requires_vision") and not config.get("supports_vision"):
                continue
            if self._circuit_open(tenant_id, row.id) or self._quota_exceeded(tenant_id, row.id):
                continue
            filtered.append(row)
        return filtered

    def _select(self, candidates: list[Any], strategy: str, payload: dict[str, Any]) -> Any:
        fixed_model_id = str(payload.get("model_id") or payload.get("fixed_model_id") or "")
        if strategy == "fixed" and fixed_model_id:
            for row in candidates:
                if row.id == fixed_model_id or row.model_id == fixed_model_id:
                    return row
            raise AppError(ErrorCode.NOT_FOUND, f"fixed model not available: {fixed_model_id}", 404)
        if strategy == "weighted":
            return self._weighted(candidates, payload)
        if strategy == "priority":
            return sorted(candidates, key=lambda row: int((row.config or {}).get("priority") or 1000))[0]
        if strategy == "cost_first":
            return min(candidates, key=lambda row: float((row.config or {}).get("pricing_input_per_1k") or 0))
        if strategy == "latency_first":
            return min(candidates, key=lambda row: int((row.config or {}).get("latency_ms") or 0))
        if strategy == "quality_first":
            return max(candidates, key=lambda row: float((row.config or {}).get("quality_score") or 0))
        return candidates[0]

    def _circuit_open(self, tenant_id: str, model_id: str) -> bool:
        rows, _ = self.resources.list("model_circuit_breaker_states", tenant_id, 1, 20, {"model_id": model_id})
        return any(row.status == "open" for row in rows)

    def _quota_exceeded(self, tenant_id: str, model_id: str) -> bool:
        quotas, _ = self.resources.list("model_quota", tenant_id, 1, 20, {"model_id": model_id})
        if not quotas:
            return False
        usage_rows, _ = self.resources.list("model_quota_usage", tenant_id, 1, 1000, {"model_id": model_id})
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        for quota in quotas:
            limit = int((quota.config or {}).get("calls_per_day") or 0)
            if limit <= 0:
                continue
            used = sum(1 for row in usage_rows if (now - row.created_at).total_seconds() < 86400)
            if used >= limit:
                return True
        return False

    @staticmethod
    def _weighted(candidates: list[Any], payload: dict[str, Any]) -> Any:
        total = sum(max(1, int((row.config or {}).get("weight") or 1)) for row in candidates)
        seed = hashlib.sha256(str(payload).encode("utf-8")).hexdigest()
        pick = int(seed[:8], 16) % total
        cursor = 0
        for row in candidates:
            cursor += max(1, int((row.config or {}).get("weight") or 1))
            if pick < cursor:
                return row
        return candidates[0]

    def _rotate_child(self, table: str, tenant_id: str, provider_id: str, key: str, payload: dict[str, Any]) -> str:
        explicit = str(payload.get(key) or "")
        if explicit:
            return explicit
        rows, _ = self.resources.list(table, tenant_id, 1, 100, {"provider_id": provider_id})
        enabled = [row for row in rows if row.enabled and row.status in {"active", "healthy", "success"}]
        if not enabled:
            return ""
        seed = int(hashlib.sha256(f"{provider_id}:{payload}".encode("utf-8")).hexdigest()[:8], 16)
        return str(enabled[seed % len(enabled)].id)

    def reset_circuit_breaker(self, tenant_id: str, user_id: str, model_id: str) -> dict[str, Any]:
        rows, _ = self.resources.list("model_circuit_breaker_states", tenant_id, 1, 100, {"model_id": model_id})
        changed = []
        for row in rows:
            self.resources.update("model_circuit_breaker_states", tenant_id, user_id, row.id, {"status": "closed", "error_message": ""})
            changed.append(row.id)
        if not changed:
            state = self.resources.create(
                "model_circuit_breaker_states",
                tenant_id,
                user_id,
                {"name": "circuit breaker", "status": "closed", "model_id": model_id, "spec": {"reset": True}},
            )
            changed.append(state.id)
        return {"model_id": model_id, "status": "closed", "state_ids": changed}
