from __future__ import annotations

from app.main import app


def test_required_paths_are_registered() -> None:
    schema = app.openapi()
    paths = schema["paths"]
    required = [
        "/api/v1/agents",
        "/api/v1/runtime/agents/{agent_id}/instances",
        "/api/v1/model-providers",
        "/api/v1/models/{model_id}/test-embedding",
        "/api/v1/model-routing-policies",
        "/api/v1/workflows",
        "/api/v1/knowledge-bases",
        "/api/v1/tools",
        "/api/v1/memories",
        "/api/v1/observability/dashboard",
        "/api/v1/audit-integrity/verify",
        "/api/v1/security/check",
        "/api/v1/evaluation/runs",
    ]
    for path in required:
        assert path in paths

