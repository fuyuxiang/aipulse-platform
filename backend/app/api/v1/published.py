from __future__ import annotations

import hashlib
import hmac
from datetime import datetime, timezone
from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, Header
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.core.constants import ErrorCode
from app.core.errors import AppError
from app.models.resources import RESOURCE_MODELS
from app.services.runtime.agent_runner_service import AgentRunnerService
from app.services.settings.guardrail_service import GuardrailService
from app.services._shared.resource_service import ResourceService

router = APIRouter(tags=["published-agents"])


@router.post("/published/{agent_id}/chat")
async def published_agent_chat(
    agent_id: str,
    payload: dict[str, Any] = Body(default_factory=dict),
    api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    publication, key = _authenticate_publication(db, agent_id, api_key or "", "chat")
    tenant_id = publication.tenant_id
    spec = dict(publication.spec or {})
    content = str(payload.get("content") or payload.get("message") or payload.get("prompt") or "")
    if not content.strip():
        raise AppError(ErrorCode.VALIDATION_ERROR, "content/message/prompt is required", 422)

    guardrail_policy_ids = list(spec.get("guardrail_policy_ids") or [])
    guardrails = GuardrailService(db)
    input_check = guardrails.check_input(tenant_id, "published-api", {"content": content, "policy_ids": guardrail_policy_ids, "agent_id": agent_id})
    if not input_check.get("passed", True):
        raise AppError(ErrorCode.FORBIDDEN, "input blocked by guardrails", 403)

    _enforce_rate_limit(ResourceService(db), publication, "published-api")
    result = await AgentRunnerService(db).run(
        tenant_id,
        "published-api",
        agent_id,
        {
            "prompt": input_check.get("masked_content") or content,
            "session_id": str(payload.get("session_id") or "published"),
            "knowledge_base_ids": spec.get("knowledge_base_ids", []),
            "tool_ids": spec.get("tool_ids", []),
            "guardrail_policy_ids": guardrail_policy_ids,
            "memory_policy": {"enabled": True, "write_scope": "tenant", "shared": True, "include_shared": True},
            "resource_limits": {"timeout_seconds": int(spec.get("timeout_seconds") or 120)},
        },
    )
    response = str(result.get("response") or "")
    output_check = guardrails.check_output(
        tenant_id,
        "published-api",
        {"content": response, "policy_ids": guardrail_policy_ids, "agent_id": agent_id, "knowledge_context": payload.get("knowledge_context", [])},
    )
    if not output_check.get("passed", True):
        raise AppError(ErrorCode.FORBIDDEN, "output blocked by guardrails", 403)
    _record_key_use(ResourceService(db), key, "published-api")
    return {
        "agent_id": agent_id,
        "publication_id": publication.id,
        "run_id": result.get("run_id"),
        "session_id": result.get("session_id"),
        "response": output_check.get("masked_content") or response,
        "guardrails": {"input": input_check, "output": output_check},
    }


def _authenticate_publication(db: Session, agent_id: str, api_key: str, permission: str) -> tuple[Any, Any]:
    if not api_key:
        raise AppError(ErrorCode.UNAUTHORIZED, "missing X-API-Key", 401)
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    key_model = RESOURCE_MODELS["agent_api_keys"]
    rows = db.scalars(
        select(key_model).where(
            key_model.agent_id == agent_id,
            key_model.status == "active",
            key_model.enabled.is_(True),
            key_model.deleted_at.is_(None),
        )
    ).all()
    for row in rows:
        spec = dict(row.spec or {})
        if not hmac.compare_digest(str(spec.get("key_hash") or ""), key_hash):
            continue
        permissions = set(spec.get("permissions") or [])
        if permission not in permissions and "*" not in permissions:
            raise AppError(ErrorCode.FORBIDDEN, "api key permission denied", 403)
        publication = ResourceService(db).get("agent_publications", row.tenant_id, row.parent_id)
        pub_spec = dict(publication.spec or {})
        if publication.status != "active" or pub_spec.get("type") != "api":
            raise AppError(ErrorCode.FORBIDDEN, "publication is not active", 403)
        return publication, row
    raise AppError(ErrorCode.UNAUTHORIZED, "invalid X-API-Key", 401)


def _enforce_rate_limit(resources: ResourceService, publication: Any, user_id: str) -> None:
    spec = dict(publication.spec or {})
    rate_limit = dict(spec.get("rate_limit") or {})
    state = dict(spec.get("rate_state") or {})
    now = datetime.now(timezone.utc)
    minute_key = now.strftime("%Y%m%d%H%M")
    day_key = now.strftime("%Y%m%d")
    rpm = int(rate_limit.get("requests_per_minute") or 0)
    rpd = int(rate_limit.get("requests_per_day") or 0)
    if state.get("minute_key") != minute_key:
        state["minute_key"] = minute_key
        state["minute_count"] = 0
    if state.get("day_key") != day_key:
        state["day_key"] = day_key
        state["day_count"] = 0
    if rpm > 0 and int(state.get("minute_count") or 0) >= rpm:
        raise AppError(ErrorCode.RATE_LIMITED, "publication minute rate limit exceeded", 429)
    if rpd > 0 and int(state.get("day_count") or 0) >= rpd:
        raise AppError(ErrorCode.RATE_LIMITED, "publication daily rate limit exceeded", 429)
    state["minute_count"] = int(state.get("minute_count") or 0) + 1
    state["day_count"] = int(state.get("day_count") or 0) + 1
    spec["rate_state"] = state
    spec["total_requests"] = int(spec.get("total_requests") or 0) + 1
    resources.update("agent_publications", publication.tenant_id, user_id, publication.id, {"spec": spec})


def _record_key_use(resources: ResourceService, key: Any, user_id: str) -> None:
    spec = dict(key.spec or {})
    spec["last_used_at"] = datetime.now(timezone.utc).isoformat()
    spec["total_uses"] = int(spec.get("total_uses") or 0) + 1
    resources.update("agent_api_keys", key.tenant_id, user_id, key.id, {"spec": spec})
