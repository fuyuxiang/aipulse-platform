from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def auth_headers(client: TestClient) -> dict[str, str]:
    response = client.post("/api/v1/auth/login", json={"tenant": "default", "username": "admin", "password": "admin123456"})
    assert response.status_code == 200
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_login_current_user_and_audit_chain() -> None:
    client = TestClient(app)
    headers = auth_headers(client)
    me = client.get("/api/v1/auth/me", headers=headers)
    assert me.status_code == 200
    assert me.json()["username"] == "admin"
    audit = client.get("/api/v1/audit-integrity/verify", headers=headers)
    assert audit.status_code == 200
    assert audit.json()["valid"] is True


def test_rbac_rejects_missing_token() -> None:
    client = TestClient(app)
    response = client.get("/api/v1/users")
    assert response.status_code == 401

