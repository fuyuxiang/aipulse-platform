from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from app.api.deps import CurrentUserDep, get_db
from app.core.response import ListResponse
from app.schemas.auth import CurrentUser, LoginRequest, RefreshRequest, RevokeTokenRequest, TokenPair
from app.services._shared.auth_service import AuthService

router = APIRouter(tags=["auth"])


@router.post("/auth/login", response_model=TokenPair)
def login(payload: LoginRequest, request: Request, db: Session = Depends(get_db)) -> TokenPair:
    user, tenant = AuthService(db).authenticate(
        payload.tenant,
        payload.username,
        payload.password,
        ip=request.client.host if request.client else "",
        user_agent=request.headers.get("User-Agent", ""),
    )
    return TokenPair(**AuthService(db).issue_tokens(user, tenant))


@router.post("/auth/refresh", response_model=TokenPair)
def refresh(payload: RefreshRequest, db: Session = Depends(get_db)) -> TokenPair:
    return TokenPair(**AuthService(db).refresh(payload.refresh_token))


@router.post("/auth/revoke")
def revoke(payload: RevokeTokenRequest, current_user: CurrentUserDep, db: Session = Depends(get_db)) -> dict[str, str]:
    return AuthService(db).revoke(payload.token, current_user)


@router.post("/auth/logout")
def logout(payload: RevokeTokenRequest, current_user: CurrentUserDep, db: Session = Depends(get_db)) -> dict[str, str]:
    return AuthService(db).revoke(payload.token, current_user)


@router.get("/auth/me", response_model=CurrentUser)
def me(current_user: CurrentUserDep, db: Session = Depends(get_db)) -> CurrentUser:
    permissions = AuthService(db).permissions_for(current_user)
    return CurrentUser(
        id=current_user.id,
        tenant_id=current_user.tenant_id,
        username=current_user.username,
        display_name=current_user.display_name,
        email=current_user.email,
        is_superuser=current_user.is_superuser,
        must_change_password=current_user.must_change_password,
        permissions=permissions,
        login_at=current_user.last_login_at,
    )


@router.get("/auth/menu-permissions", response_model=ListResponse[str])
def menu_permissions(current_user: CurrentUserDep, db: Session = Depends(get_db)) -> ListResponse[str]:
    permissions = AuthService(db).permissions_for(current_user)
    return ListResponse(items=permissions, total=len(permissions), page=1, page_size=len(permissions) or 1)
