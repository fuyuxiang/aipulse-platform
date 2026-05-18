from __future__ import annotations

from enum import StrEnum


class ErrorCode(StrEnum):
    OK = "OK"
    BAD_REQUEST = "BAD_REQUEST"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    NOT_FOUND = "NOT_FOUND"
    CONFLICT = "CONFLICT"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    RATE_LIMITED = "RATE_LIMITED"
    TENANT_REQUIRED = "TENANT_REQUIRED"
    TENANT_INACTIVE = "TENANT_INACTIVE"
    BUSINESS_ERROR = "BUSINESS_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"


DEFAULT_TENANT_CODE = "default"
DEFAULT_ADMIN_USERNAME = "admin"
DEFAULT_ADMIN_PASSWORD = "admin123456"

MODEL_TYPES = {
    "chat_llm",
    "completion_llm",
    "reasoning_llm",
    "vision_language",
    "embedding",
    "rerank",
    "moderation",
    "speech_to_text",
    "text_to_speech",
    "image_generation",
}

MODEL_PROVIDER_TYPES = {
    "openai_compatible",
    "azure_openai",
    "anthropic_compatible",
    "google_gemini_compatible",
    "ollama",
    "vllm_openai_compatible",
    "local_http",
    "custom_http",
    "echo_agent_native",
}

ROUTING_POLICY_TYPES = {
    "fixed",
    "weighted",
    "priority",
    "cost_first",
    "latency_first",
    "quality_first",
    "capability_match",
    "tenant_policy",
    "agent_policy",
    "workflow_policy",
}

