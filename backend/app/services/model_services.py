from __future__ import annotations

import asyncio
import hashlib
import math
import time
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


class ModelRoutingService:
    def __init__(self, db: Session):
        self.db = db
        self.resources = ResourceService(db)

    def route(self, tenant_id: str, user_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        model_type = payload.get("model_type", "chat_llm")
        agent_id = payload.get("agent_id", "")
        rows, _ = self.resources.list("models", tenant_id, 1, 200, {"status": "active"})
        candidates = [row for row in rows if row.enabled and (not model_type or row.model_type == model_type)]
        if not candidates:
            raise AppError(ErrorCode.NOT_FOUND, f"no enabled model for type {model_type}", 404)
        selected = self._select(candidates, payload.get("strategy", "capability_match"))
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
                "output_payload": {"selected_model_id": selected.id},
            },
        )
        return {"model_id": selected.id, "model_type": selected.model_type, "provider_id": selected.provider_id, "reason": "local policy match"}

    @staticmethod
    def _select(candidates: list[Any], strategy: str) -> Any:
        if strategy == "cost_first":
            return min(candidates, key=lambda row: float((row.config or {}).get("pricing_input_per_1k") or 0))
        if strategy == "latency_first":
            return min(candidates, key=lambda row: int((row.config or {}).get("latency_ms") or 0))
        if strategy == "quality_first":
            return max(candidates, key=lambda row: float((row.config or {}).get("quality_score") or 0))
        return candidates[0]
