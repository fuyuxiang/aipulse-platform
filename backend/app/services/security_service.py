from __future__ import annotations

import re
from typing import Any

from sqlalchemy.orm import Session

from app.services.resource_service import ResourceService


class SecurityService:
    def __init__(self, db: Session):
        self.resources = ResourceService(db)

    def check(self, tenant_id: str, user_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        text = str(payload.get("text") or payload.get("prompt") or payload)
        rules, _ = self.resources.list("sensitive_rules", tenant_id, 1, 200)
        hits: list[dict[str, str]] = []
        for rule in rules:
            pattern = (rule.spec or {}).get("pattern") or rule.code or rule.name
            if pattern and re.search(pattern, text, re.IGNORECASE):
                hits.append({"rule_id": rule.id, "name": rule.name, "pattern": pattern})
        injection = any(term in text.lower() for term in ("ignore previous", "system prompt", "developer message", "bypass"))
        status = "blocked" if hits or injection else "allowed"
        self.resources.create(
            "security_events",
            tenant_id,
            user_id,
            {"name": "security check", "status": status, "input_payload": payload, "output_payload": {"hits": hits, "prompt_injection": injection}},
        )
        return {"status": status, "hits": hits, "prompt_injection": injection}

    @staticmethod
    def desensitize(payload: dict[str, Any]) -> dict[str, Any]:
        text = str(payload.get("text", ""))
        text = re.sub(r"([A-Za-z0-9._%+-]+)@([A-Za-z0-9.-]+)", "***@\\2", text)
        text = re.sub(r"(password|token|secret|api[_-]?key)\\s*[:=]\\s*\\S+", "\\1=***", text, flags=re.IGNORECASE)
        return {"text": text}

