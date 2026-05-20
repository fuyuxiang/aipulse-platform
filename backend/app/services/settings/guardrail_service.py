from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.services._shared.resource_service import ResourceService


class GuardrailService:
    PII_PATTERNS = {
        "phone": r"(?<!\d)1[3-9]\d{9}(?!\d)",
        "id_card": r"\b\d{17}[\dXx]\b",
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "bank_card": r"\b\d{16,19}\b",
        "ip_address": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        "credit_card": r"\b(?:4\d{12}(?:\d{3})?|5[1-5]\d{14}|3[47]\d{13})\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "passport": r"\b[A-Z]\d{8}\b",
    }

    INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"ignore\s+(all\s+)?above\s+instructions",
        r"disregard\s+(all\s+)?previous",
        r"forget\s+(all\s+)?previous",
        r"you\s+are\s+now\s+(?:a|an)\s+",
        r"new\s+instructions?\s*:",
        r"system\s*:\s*you\s+are",
        r"<\s*system\s*>",
        r"\[\s*INST\s*\]",
        r"jailbreak",
        r"DAN\s+mode",
        r"developer\s+mode\s+enabled",
        r"pretend\s+you\s+(?:are|have)\s+no\s+(?:rules|restrictions|limitations)",
        r"act\s+as\s+(?:if\s+)?you\s+(?:have|had)\s+no\s+(?:rules|restrictions)",
    ]

    UNSAFE_CONTENT_KEYWORDS = {
        "violence": ["杀人", "暴力", "伤害", "自杀", "murder", "kill", "violence", "harm"],
        "illegal": ["毒品", "贩毒", "走私", "洗钱", "drug", "smuggle", "launder"],
        "hate_speech": ["歧视", "种族", "仇恨", "hate speech", "racist", "discrimination"],
        "sexual": ["色情", "淫秽", "pornography", "explicit sexual"],
        "self_harm": ["自残", "自杀方法", "suicide method", "self-harm instructions"],
    }

    def __init__(self, db: Session):
        self.db = db
        self.resources = ResourceService(db)

    def create_policy(self, tenant_id: str, user_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        policy = self.resources.create("guardrail_policies", tenant_id, user_id, {
            "name": str(payload.get("name", "")),
            "code": f"gp-{uuid.uuid4().hex[:8]}",
            "status": "active",
            "spec": {
                "description": str(payload.get("description", "")),
                "scope": str(payload.get("scope", "both")),
                "action": str(payload.get("action", "block")),
                "enabled_checks": payload.get("enabled_checks", [
                    "pii_detection", "prompt_injection", "content_safety",
                    "output_format", "hallucination_detection", "topic_restriction",
                ]),
                "pii_config": payload.get("pii_config", {
                    "enabled": True,
                    "action": "mask",
                    "patterns": list(self.PII_PATTERNS.keys()),
                }),
                "injection_config": payload.get("injection_config", {
                    "enabled": True,
                    "action": "block",
                    "sensitivity": "medium",
                }),
                "content_safety_config": payload.get("content_safety_config", {
                    "enabled": True,
                    "action": "block",
                    "categories": list(self.UNSAFE_CONTENT_KEYWORDS.keys()),
                }),
                "output_format_config": payload.get("output_format_config", {
                    "enabled": False,
                    "format": "text",
                    "max_length": 10000,
                    "required_fields": [],
                }),
                "hallucination_config": payload.get("hallucination_config", {
                    "enabled": True,
                    "action": "warn",
                    "knowledge_base_ids": [],
                    "threshold": 0.7,
                }),
                "topic_restriction_config": payload.get("topic_restriction_config", {
                    "enabled": False,
                    "allowed_topics": [],
                    "blocked_topics": [],
                }),
                "custom_rules": payload.get("custom_rules", []),
            },
        })
        return ResourceService.to_dict(policy)

    def update_policy(self, tenant_id: str, user_id: str, policy_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        row = self.resources.update("guardrail_policies", tenant_id, user_id, policy_id, payload)
        return ResourceService.to_dict(row)

    def delete_policy(self, tenant_id: str, user_id: str, policy_id: str) -> dict[str, str]:
        return self.resources.delete("guardrail_policies", tenant_id, user_id, policy_id)

    def get_policy(self, tenant_id: str, policy_id: str) -> dict[str, Any]:
        return ResourceService.to_dict(self.resources.get("guardrail_policies", tenant_id, policy_id))

    def list_policies(self, tenant_id: str, page: int, page_size: int) -> tuple[list[dict[str, Any]], int]:
        rows, total = self.resources.list("guardrail_policies", tenant_id, page, page_size)
        return [ResourceService.to_dict(row) for row in rows], total

    def create_rule(self, tenant_id: str, user_id: str, policy_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        rule = self.resources.create("guardrail_rules", tenant_id, user_id, {
            "name": str(payload.get("name", "")),
            "code": f"gr-{uuid.uuid4().hex[:8]}",
            "status": "active",
            "parent_id": policy_id,
            "spec": {
                "rule_type": str(payload.get("rule_type", "regex")),
                "pattern": str(payload.get("pattern", "")),
                "action": str(payload.get("action", "block")),
                "message": str(payload.get("message", "")),
                "severity": str(payload.get("severity", "medium")),
                "scope": str(payload.get("scope", "both")),
                "conditions": payload.get("conditions", {}),
            },
        })
        return ResourceService.to_dict(rule)

    def list_rules(self, tenant_id: str, policy_id: str, page: int, page_size: int) -> tuple[list[dict[str, Any]], int]:
        rows, total = self.resources.list("guardrail_rules", tenant_id, page, page_size, {"parent_id": policy_id})
        return [ResourceService.to_dict(row) for row in rows], total

    def check_input(self, tenant_id: str, user_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        content = str(payload.get("content", ""))
        policy_ids = payload.get("policy_ids", [])
        agent_id = str(payload.get("agent_id", ""))
        return self._execute_check(tenant_id, user_id, content, policy_ids, agent_id, "input")

    def check_output(self, tenant_id: str, user_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        content = str(payload.get("content", ""))
        policy_ids = payload.get("policy_ids", [])
        agent_id = str(payload.get("agent_id", ""))
        knowledge_context = payload.get("knowledge_context", [])
        return self._execute_check(tenant_id, user_id, content, policy_ids, agent_id, "output", knowledge_context)

    def _execute_check(
        self, tenant_id: str, user_id: str, content: str,
        policy_ids: list[str], agent_id: str, scope: str,
        knowledge_context: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        violations: list[dict[str, Any]] = []
        passed = True
        action_taken = "pass"
        masked_content = content

        policies_to_check = []
        if policy_ids:
            for pid in policy_ids:
                try:
                    policies_to_check.append(self.resources.get("guardrail_policies", tenant_id, pid))
                except Exception:
                    pass
        else:
            rows, _ = self.resources.list("guardrail_policies", tenant_id, 1, 100)
            policies_to_check = rows

        for policy in policies_to_check:
            spec = dict(policy.spec or {})
            policy_scope = spec.get("scope", "both")
            if policy_scope != "both" and policy_scope != scope:
                continue

            enabled_checks = spec.get("enabled_checks", [])

            if "pii_detection" in enabled_checks:
                pii_config = spec.get("pii_config", {})
                if pii_config.get("enabled", True):
                    pii_violations, masked_content = self._check_pii(content, pii_config)
                    violations.extend(pii_violations)

            if "prompt_injection" in enabled_checks and scope == "input":
                injection_config = spec.get("injection_config", {})
                if injection_config.get("enabled", True):
                    injection_violations = self._check_injection(content, injection_config)
                    violations.extend(injection_violations)

            if "content_safety" in enabled_checks:
                safety_config = spec.get("content_safety_config", {})
                if safety_config.get("enabled", True):
                    safety_violations = self._check_content_safety(content, safety_config)
                    violations.extend(safety_violations)

            if "output_format" in enabled_checks and scope == "output":
                format_config = spec.get("output_format_config", {})
                if format_config.get("enabled", False):
                    format_violations = self._check_output_format(content, format_config)
                    violations.extend(format_violations)

            if "hallucination_detection" in enabled_checks and scope == "output":
                hallucination_config = spec.get("hallucination_config", {})
                if hallucination_config.get("enabled", True) and knowledge_context:
                    hallucination_violations = self._check_hallucination(content, knowledge_context, hallucination_config)
                    violations.extend(hallucination_violations)

            if "topic_restriction" in enabled_checks:
                topic_config = spec.get("topic_restriction_config", {})
                if topic_config.get("enabled", False):
                    topic_violations = self._check_topic(content, topic_config)
                    violations.extend(topic_violations)

            custom_rules = spec.get("custom_rules", [])
            for rule in custom_rules:
                rule_violations = self._check_custom_rule(content, rule)
                violations.extend(rule_violations)

        for v in violations:
            if v.get("action") == "block":
                passed = False
                action_taken = "block"
                break
            elif v.get("action") == "warn" and action_taken != "block":
                action_taken = "warn"

        execution = self.resources.create("guardrail_executions", tenant_id, user_id, {
            "name": f"check-{scope}-{uuid.uuid4().hex[:6]}",
            "code": f"gex-{uuid.uuid4().hex[:8]}",
            "status": "completed",
            "agent_id": agent_id,
            "spec": {
                "scope": scope,
                "content_length": len(content),
                "policy_count": len(policies_to_check),
                "violation_count": len(violations),
                "passed": passed,
                "action_taken": action_taken,
                "checked_at": datetime.now(timezone.utc).isoformat(),
            },
        })

        for v in violations:
            self.resources.create("guardrail_violations", tenant_id, user_id, {
                "name": v.get("type", "violation"),
                "code": f"gv-{uuid.uuid4().hex[:8]}",
                "status": action_taken,
                "parent_id": execution.id,
                "agent_id": agent_id,
                "spec": v,
            })

        return {
            "passed": passed,
            "action": action_taken,
            "violations": violations,
            "masked_content": masked_content if masked_content != content else None,
            "execution_id": execution.id,
        }

    def _check_pii(self, content: str, config: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
        violations = []
        masked = content
        patterns_to_check = config.get("patterns", list(self.PII_PATTERNS.keys()))
        action = config.get("action", "mask")

        for pii_type in patterns_to_check:
            pattern = self.PII_PATTERNS.get(pii_type)
            if not pattern:
                continue
            matches = re.finditer(pattern, content)
            for match in matches:
                violations.append({
                    "type": "pii_detection",
                    "subtype": pii_type,
                    "action": action,
                    "severity": "high",
                    "position": {"start": match.start(), "end": match.end()},
                    "matched_text": match.group()[:4] + "***",
                    "message": f"检测到 {pii_type} 类型的个人信息",
                })
                if action == "mask":
                    masked = masked[:match.start()] + "***" + masked[match.end():]

        return violations, masked

    def _check_injection(self, content: str, config: dict[str, Any]) -> list[dict[str, Any]]:
        violations = []
        sensitivity = config.get("sensitivity", "medium")
        action = config.get("action", "block")

        patterns = self.INJECTION_PATTERNS
        if sensitivity == "low":
            patterns = patterns[:5]
        elif sensitivity == "high":
            patterns = patterns

        for pattern in patterns:
            if re.search(pattern, content, re.IGNORECASE):
                violations.append({
                    "type": "prompt_injection",
                    "subtype": "pattern_match",
                    "action": action,
                    "severity": "critical",
                    "pattern": pattern,
                    "message": "检测到潜在的 Prompt 注入攻击",
                })
                break

        injection_indicators = 0
        if len(content) > 2000:
            injection_indicators += 1
        if content.count("\n") > 20:
            injection_indicators += 1
        if re.search(r"```|<script|<iframe", content, re.IGNORECASE):
            injection_indicators += 1
        if re.search(r"(role|system|assistant)\s*:", content, re.IGNORECASE):
            injection_indicators += 1

        threshold = {"low": 4, "medium": 3, "high": 2}.get(sensitivity, 3)
        if injection_indicators >= threshold and not violations:
            violations.append({
                "type": "prompt_injection",
                "subtype": "heuristic",
                "action": "warn",
                "severity": "medium",
                "indicators": injection_indicators,
                "message": "内容具有 Prompt 注入的特征",
            })

        return violations

    def _check_content_safety(self, content: str, config: dict[str, Any]) -> list[dict[str, Any]]:
        violations = []
        categories = config.get("categories", list(self.UNSAFE_CONTENT_KEYWORDS.keys()))
        action = config.get("action", "block")

        for category in categories:
            keywords = self.UNSAFE_CONTENT_KEYWORDS.get(category, [])
            for keyword in keywords:
                if keyword.lower() in content.lower():
                    violations.append({
                        "type": "content_safety",
                        "subtype": category,
                        "action": action,
                        "severity": "high",
                        "keyword": keyword,
                        "message": f"检测到不安全内容类别: {category}",
                    })
                    break

        return violations

    def _check_output_format(self, content: str, config: dict[str, Any]) -> list[dict[str, Any]]:
        violations = []
        max_length = config.get("max_length", 10000)
        required_format = config.get("format", "text")

        if len(content) > max_length:
            violations.append({
                "type": "output_format",
                "subtype": "length_exceeded",
                "action": "warn",
                "severity": "low",
                "max_length": max_length,
                "actual_length": len(content),
                "message": f"输出长度 {len(content)} 超过限制 {max_length}",
            })

        if required_format == "json":
            import json as json_mod
            try:
                json_mod.loads(content)
            except (json_mod.JSONDecodeError, ValueError):
                violations.append({
                    "type": "output_format",
                    "subtype": "invalid_json",
                    "action": "warn",
                    "severity": "medium",
                    "message": "输出不是有效的 JSON 格式",
                })

        required_fields = config.get("required_fields", [])
        for field in required_fields:
            if field not in content:
                violations.append({
                    "type": "output_format",
                    "subtype": "missing_field",
                    "action": "warn",
                    "severity": "low",
                    "field": field,
                    "message": f"输出缺少必需字段: {field}",
                })

        return violations

    def _check_hallucination(self, content: str, knowledge_context: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
        violations = []
        threshold = config.get("threshold", 0.7)
        action = config.get("action", "warn")

        if not knowledge_context:
            return violations

        knowledge_text = " ".join(k.get("content", "") for k in knowledge_context).lower()
        sentences = [s.strip() for s in re.split(r'[.。!！?？\n]', content) if len(s.strip()) > 20]

        unsupported_count = 0
        for sentence in sentences:
            words = set(re.findall(r'\w+', sentence.lower()))
            if not words:
                continue
            knowledge_words = set(re.findall(r'\w+', knowledge_text))
            overlap = len(words & knowledge_words) / len(words) if words else 0
            if overlap < threshold:
                unsupported_count += 1

        if sentences and unsupported_count / len(sentences) > 0.5:
            violations.append({
                "type": "hallucination_detection",
                "subtype": "low_grounding",
                "action": action,
                "severity": "medium",
                "unsupported_ratio": unsupported_count / len(sentences),
                "total_sentences": len(sentences),
                "unsupported_sentences": unsupported_count,
                "message": f"输出中 {unsupported_count}/{len(sentences)} 句话缺乏知识库支撑",
            })

        return violations

    def _check_topic(self, content: str, config: dict[str, Any]) -> list[dict[str, Any]]:
        violations = []
        allowed_topics = config.get("allowed_topics", [])
        blocked_topics = config.get("blocked_topics", [])

        content_lower = content.lower()
        for topic in blocked_topics:
            if topic.lower() in content_lower:
                violations.append({
                    "type": "topic_restriction",
                    "subtype": "blocked_topic",
                    "action": "block",
                    "severity": "medium",
                    "topic": topic,
                    "message": f"内容涉及被禁止的话题: {topic}",
                })

        return violations

    def _check_custom_rule(self, content: str, rule: dict[str, Any]) -> list[dict[str, Any]]:
        violations = []
        rule_type = rule.get("type", "regex")
        pattern = rule.get("pattern", "")
        action = rule.get("action", "warn")

        if rule_type == "regex" and pattern:
            if re.search(pattern, content, re.IGNORECASE):
                violations.append({
                    "type": "custom_rule",
                    "subtype": rule.get("name", "custom"),
                    "action": action,
                    "severity": rule.get("severity", "medium"),
                    "pattern": pattern,
                    "message": rule.get("message", "触发自定义规则"),
                })
        elif rule_type == "keyword" and pattern:
            if pattern.lower() in content.lower():
                violations.append({
                    "type": "custom_rule",
                    "subtype": rule.get("name", "custom"),
                    "action": action,
                    "severity": rule.get("severity", "medium"),
                    "keyword": pattern,
                    "message": rule.get("message", "触发自定义关键词规则"),
                })

        return violations

    def list_executions(self, tenant_id: str, page: int, page_size: int, filters: dict[str, Any] | None = None) -> tuple[list[dict[str, Any]], int]:
        rows, total = self.resources.list("guardrail_executions", tenant_id, page, page_size, filters)
        return [ResourceService.to_dict(row) for row in rows], total

    def list_violations(self, tenant_id: str, page: int, page_size: int, filters: dict[str, Any] | None = None) -> tuple[list[dict[str, Any]], int]:
        rows, total = self.resources.list("guardrail_violations", tenant_id, page, page_size, filters)
        return [ResourceService.to_dict(row) for row in rows], total

    def get_stats(self, tenant_id: str) -> dict[str, Any]:
        _, total_executions = self.resources.list("guardrail_executions", tenant_id, 1, 1)
        _, total_violations = self.resources.list("guardrail_violations", tenant_id, 1, 1)
        _, total_policies = self.resources.list("guardrail_policies", tenant_id, 1, 1)
        return {
            "total_executions": total_executions,
            "total_violations": total_violations,
            "total_policies": total_policies,
        }
