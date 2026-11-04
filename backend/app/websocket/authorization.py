from __future__ import annotations

from fastapi import WebSocket
from app.db.session import SessionLocal
from app.core.errors import AppError
from app.services._shared.auth_service import AuthService
from app.services.settings.rbac_service import RBACService
from app.services._shared.resource_service import ResourceService


async def authorize_run_socket(websocket: WebSocket, table: str, run_id: str, permission: str) -> str | None:
    token = websocket.headers.get("authorization", "").removeprefix("Bearer ").strip()
    try:
        with SessionLocal() as db:
            user = AuthService(db).authenticate_token(token)
            RBACService(AuthService(db)).require(user, permission)
            run = ResourceService(db).get(table, user.tenant_id, run_id)
            if not user.is_superuser and run.created_by != user.id:
                await websocket.close(code=4403)
                return None
            return f"{user.tenant_id}:{run_id}"
    except (AppError, ValueError):
        await websocket.close(code=4403)
        return None
