from __future__ import annotations

from fastapi import APIRouter, Body, Depends, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.api.deps import TenantIdDep, get_db, require_permission
from app.core.response import ListResponse
from app.models.core import User
from app.schemas.common import ResourceRead
from app.services.chat_service import ChatService
from app.services.resource_service import ResourceService

router = APIRouter(tags=["chat"])


@router.post("/chat/sessions")
def create_session(
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    user: User = Depends(require_permission("chat:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return ChatService(db).create_session(tenant_id, user.id, dict(payload))


@router.get("/chat/sessions")
def list_sessions(
    tenant_id: TenantIdDep,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    user: User = Depends(require_permission("chat:read")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    items, total = ChatService(db).list_sessions(tenant_id, user.id, page, page_size)
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@router.get("/chat/sessions/{session_id}")
def get_session(
    session_id: str,
    tenant_id: TenantIdDep,
    _: User = Depends(require_permission("chat:read")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return ChatService(db).get_session(tenant_id, session_id)


@router.put("/chat/sessions/{session_id}")
def update_session(
    session_id: str,
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    user: User = Depends(require_permission("chat:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return ChatService(db).update_session(tenant_id, user.id, session_id, dict(payload))


@router.delete("/chat/sessions/{session_id}")
def delete_session(
    session_id: str,
    tenant_id: TenantIdDep,
    user: User = Depends(require_permission("chat:write")),
    db: Session = Depends(get_db),
) -> dict[str, str]:
    return ChatService(db).delete_session(tenant_id, user.id, session_id)


@router.post("/chat/sessions/{session_id}/archive")
def archive_session(
    session_id: str,
    tenant_id: TenantIdDep,
    user: User = Depends(require_permission("chat:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return ChatService(db).archive_session(tenant_id, user.id, session_id)


@router.post("/chat/sessions/{session_id}/pin")
def pin_session(
    session_id: str,
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    user: User = Depends(require_permission("chat:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return ChatService(db).pin_session(tenant_id, user.id, session_id, bool(payload.get("pinned", True)))


@router.get("/chat/sessions/{session_id}/messages")
def list_messages(
    session_id: str,
    tenant_id: TenantIdDep,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    _: User = Depends(require_permission("chat:read")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    items, total = ChatService(db).list_messages(tenant_id, session_id, page, page_size)
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@router.post("/chat/sessions/{session_id}/messages")
def send_message(
    session_id: str,
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    user: User = Depends(require_permission("chat:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return ChatService(db).send_message(tenant_id, user.id, session_id, dict(payload))


@router.post("/chat/sessions/{session_id}/stream")
async def stream_reply(
    session_id: str,
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    user: User = Depends(require_permission("chat:write")),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    generator = ChatService(db).stream_reply(tenant_id, user.id, session_id, dict(payload))
    return StreamingResponse(generator, media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    })


@router.post("/chat/messages/{message_id}/feedback")
def feedback_message(
    message_id: str,
    tenant_id: TenantIdDep,
    payload: dict[str, object] = Body(default_factory=dict),
    user: User = Depends(require_permission("chat:write")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return ChatService(db).feedback_message(tenant_id, user.id, message_id, dict(payload))


@router.post("/chat/sessions/{session_id}/messages/{message_id}/regenerate")
async def regenerate_message(
    session_id: str,
    message_id: str,
    tenant_id: TenantIdDep,
    user: User = Depends(require_permission("chat:write")),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    service = ChatService(db)
    service.regenerate_message(tenant_id, user.id, session_id, message_id)
    msg = service.get_message(tenant_id, message_id)
    parent_content = (msg.get("spec") or {}).get("content", "")
    msgs, _ = service.list_messages(tenant_id, session_id, 1, 50)
    user_msg_content = ""
    for m in reversed(msgs):
        if (m.get("spec") or {}).get("role") == "user":
            user_msg_content = (m.get("spec") or {}).get("content", "")
            break
    generator = service.stream_reply(tenant_id, user.id, session_id, {"content": user_msg_content, "_regenerate": True})
    return StreamingResponse(generator, media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    })
