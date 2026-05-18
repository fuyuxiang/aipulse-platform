from __future__ import annotations

from app.core.constants import ErrorCode
from app.core.errors import AppError
from app.models.core import User
from app.services.auth_service import AuthService


class RBACService:
    def __init__(self, auth: AuthService):
        self.auth = auth

    def require(self, user: User, permission: str) -> None:
        permissions = self.auth.permissions_for(user)
        if "*" in permissions or permission in permissions:
            return
        domain = permission.split(":", 1)[0]
        if f"{domain}:*" in permissions:
            return
        raise AppError(ErrorCode.FORBIDDEN, f"permission denied: {permission}", 403)

