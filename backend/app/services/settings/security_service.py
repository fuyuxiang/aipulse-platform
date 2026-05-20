from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.services._shared.resource_service import ResourceService


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
        injection_hits = self._prompt_injection_hits(tenant_id, text)
        content_hits = self._content_policy_hits(tenant_id, text)
        ip_allowed = self._ip_allowed(tenant_id, str(payload.get("ip_address") or ""))
        rate_limited = self._rate_limited(tenant_id, user_id)
        status = "blocked" if hits or injection_hits or content_hits or not ip_allowed or rate_limited else "allowed"
        self.resources.create(
            "security_events",
            tenant_id,
            user_id,
            {
                "name": "security check",
                "status": status,
                "user_id": user_id,
                "input_payload": payload,
                "output_payload": {
                    "hits": hits,
                    "prompt_injection": bool(injection_hits),
                    "prompt_injection_hits": injection_hits,
                    "content_hits": content_hits,
                    "ip_allowed": ip_allowed,
                    "rate_limited": rate_limited,
                },
            },
        )
        return {"status": status, "hits": hits, "prompt_injection": bool(injection_hits), "content_hits": content_hits, "ip_allowed": ip_allowed, "rate_limited": rate_limited}

    @staticmethod
    def desensitize(payload: dict[str, Any]) -> dict[str, Any]:
        text = str(payload.get("text", ""))
        text = re.sub(r"([A-Za-z0-9._%+-]+)@([A-Za-z0-9.-]+)", "***@\\2", text)
        text = re.sub(r"(password|token|secret|api[_-]?key)\\s*[:=]\\s*\\S+", "\\1=***", text, flags=re.IGNORECASE)
        return {"text": text}

    def _prompt_injection_hits(self, tenant_id: str, text: str) -> list[dict[str, str]]:
        defaults = ["ignore previous", "system prompt", "developer message", "bypass"]
        rules, total = self.resources.list("prompt_injection_rules", tenant_id, 1, 200)
        patterns = [(rule.id, rule.name, str((rule.spec or {}).get("pattern") or rule.code or rule.name)) for rule in rules]
        if total == 0:
            patterns = [("builtin", term, re.escape(term)) for term in defaults]
        return [{"rule_id": rule_id, "name": name, "pattern": pattern} for rule_id, name, pattern in patterns if pattern and re.search(pattern, text, re.IGNORECASE)]

    def _content_policy_hits(self, tenant_id: str, text: str) -> list[dict[str, str]]:
        rows, _ = self.resources.list("content_safety_policies", tenant_id, 1, 200, {"status": "active"})
        hits: list[dict[str, str]] = []
        for row in rows:
            terms = list((row.spec or {}).get("blocked_terms") or (row.config or {}).get("blocked_terms") or [])
            for term in terms:
                if str(term).lower() in text.lower():
                    hits.append({"policy_id": row.id, "name": row.name, "term": str(term)})
        return hits

    def _ip_allowed(self, tenant_id: str, ip_address: str) -> bool:
        if not ip_address:
            return True
        rows, total = self.resources.list("ip_allowlists", tenant_id, 1, 200, {"status": "active"})
        if total == 0:
            return True
        allowed = set()
        for row in rows:
            allowed.add(row.code)
            allowed.update(str(item) for item in (row.spec or {}).get("ips", []))
        return ip_address in allowed

    def _rate_limited(self, tenant_id: str, user_id: str) -> bool:
        rules, total = self.resources.list("api_rate_limit_rules", tenant_id, 1, 20, {"status": "active"})
        if total == 0:
            return False
        events, _ = self.resources.list("security_events", tenant_id, 1, 1000, {"user_id": user_id})
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        for rule in rules:
            limit = int((rule.config or {}).get("limit") or (rule.spec or {}).get("limit") or 0)
            window = int((rule.config or {}).get("window_seconds") or (rule.spec or {}).get("window_seconds") or 60)
            if limit > 0 and sum(1 for event in events if (now - event.created_at).total_seconds() <= window) >= limit:
                return True
        return False
