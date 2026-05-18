from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def headers(client: TestClient) -> dict[str, str]:
    response = client.post("/api/v1/auth/login", json={"tenant": "default", "username": "admin", "password": "admin123456"})
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def test_agent_model_knowledge_memory_security_evaluation_flow() -> None:
    client = TestClient(app)
    auth = headers(client)
    agent = client.post("/api/v1/agents", headers=auth, json={"name": "Support Agent", "code": "support", "config": {"model_type": "chat_llm"}})
    assert agent.status_code == 200
    assert agent.json()["tenant_id"]
    models = client.get("/api/v1/models", headers=auth)
    assert models.status_code == 200
    model_id = models.json()["items"][0]["id"]
    embedding = client.post(f"/api/v1/models/{model_id}/test-embedding", headers=auth, json={"texts": ["enterprise agent platform"]})
    assert embedding.status_code == 200
    assert embedding.json()["result"]["embeddings"]
    kb = client.post("/api/v1/knowledge-bases", headers=auth, json={"name": "KB", "config": {"embedding_model_id": model_id, "embedding_dimensions": 128}})
    assert kb.status_code == 200
    memory = client.post("/api/v1/memories", headers=auth, json={"name": "preference", "spec": {"scope": "user", "form": "semantic"}})
    assert memory.status_code == 200
    security = client.post("/api/v1/security/check", headers=auth, json={"text": "normal request"})
    assert security.status_code == 200
    assert security.json()["status"] == "allowed"
    evaluation = client.post("/api/v1/evaluation/runs", headers=auth, json={"cases": [{"input": "a", "expected": "a", "actual": "a"}]})
    assert evaluation.status_code == 200
    assert evaluation.json()["metrics"]["accuracy"] == 1.0


def test_runtime_debug_run_uses_echo_agent_adapter() -> None:
    client = TestClient(app)
    auth = headers(client)
    created = client.post("/api/v1/runtime/agents/test-agent/instances", headers=auth, json={"session_id": "it", "resource_limits": {"timeout_seconds": 30}})
    assert created.status_code == 200
    instance_id = created.json()["id"]
    started = client.post(f"/api/v1/runtime/instances/{instance_id}/start", headers=auth)
    assert started.status_code == 200
    assert started.json()["status"] == "running"
    run = client.post("/api/v1/runtime/agents/test-agent/debug-run", headers=auth, json={"prompt": "hello", "session_id": "it"})
    assert run.status_code == 200
    assert "response" in run.json()
    stopped = client.post(f"/api/v1/runtime/instances/{instance_id}/stop", headers=auth)
    assert stopped.status_code == 200
    assert stopped.json()["status"] == "stopped"

