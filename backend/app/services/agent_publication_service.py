from __future__ import annotations

import hashlib
import secrets
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.services.resource_service import ResourceService


class AgentPublicationService:
    def __init__(self, db: Session):
        self.db = db
        self.resources = ResourceService(db)

    def publish_as_api(self, tenant_id: str, user_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        agent_id = str(payload.get("agent_id", ""))
        name = str(payload.get("name", ""))
        api_key = f"ak-{secrets.token_hex(24)}"
        api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        publication = self.resources.create("agent_publications", tenant_id, user_id, {
            "name": name,
            "code": f"pub-{uuid.uuid4().hex[:8]}",
            "status": "active",
            "agent_id": agent_id,
            "spec": {
                "type": "api",
                "agent_id": agent_id,
                "endpoint": f"/api/v1/published/{agent_id}/chat",
                "api_key_hash": api_key_hash,
                "rate_limit": payload.get("rate_limit", {"requests_per_minute": 60, "requests_per_day": 10000}),
                "allowed_origins": payload.get("allowed_origins", ["*"]),
                "max_tokens": int(payload.get("max_tokens", 4096)),
                "temperature": float(payload.get("temperature", 0.7)),
                "system_prompt_override": str(payload.get("system_prompt_override", "")),
                "knowledge_base_ids": payload.get("knowledge_base_ids", []),
                "tool_ids": payload.get("tool_ids", []),
                "guardrail_policy_ids": payload.get("guardrail_policy_ids", []),
                "description": str(payload.get("description", "")),
                "version": str(payload.get("version", "1.0.0")),
                "total_requests": 0,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        })

        self.resources.create("agent_api_keys", tenant_id, user_id, {
            "name": f"key-{name}",
            "code": f"aak-{uuid.uuid4().hex[:8]}",
            "status": "active",
            "parent_id": publication.id,
            "agent_id": agent_id,
            "spec": {
                "key_hash": api_key_hash,
                "key_prefix": api_key[:8],
                "permissions": payload.get("permissions", ["chat", "stream"]),
                "expires_at": str(payload.get("expires_at", "")),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "last_used_at": None,
                "total_uses": 0,
            },
        })

        return {
            **ResourceService.to_dict(publication),
            "api_key": api_key,
            "endpoint": f"/api/v1/published/{agent_id}/chat",
        }

    def publish_as_widget(self, tenant_id: str, user_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        agent_id = str(payload.get("agent_id", ""))
        widget_id = uuid.uuid4().hex[:12]

        widget = self.resources.create("agent_widgets", tenant_id, user_id, {
            "name": str(payload.get("name", "")),
            "code": f"wgt-{uuid.uuid4().hex[:8]}",
            "status": "active",
            "agent_id": agent_id,
            "spec": {
                "widget_id": widget_id,
                "agent_id": agent_id,
                "theme": payload.get("theme", {
                    "primary_color": "#1890ff",
                    "border_radius": 8,
                    "position": "bottom-right",
                    "width": 400,
                    "height": 600,
                }),
                "branding": payload.get("branding", {
                    "title": str(payload.get("title", "AI 助手")),
                    "subtitle": str(payload.get("subtitle", "")),
                    "avatar": str(payload.get("avatar", "")),
                    "welcome_message": str(payload.get("welcome_message", "你好！有什么可以帮助你的？")),
                    "placeholder": str(payload.get("placeholder", "输入消息...")),
                }),
                "behavior": payload.get("behavior", {
                    "auto_open": False,
                    "auto_open_delay_ms": 3000,
                    "show_typing_indicator": True,
                    "enable_file_upload": False,
                    "enable_voice_input": False,
                    "enable_feedback": True,
                    "max_messages_per_session": 100,
                }),
                "allowed_domains": payload.get("allowed_domains", ["*"]),
                "embed_code": self._generate_embed_code(widget_id),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        })
        return ResourceService.to_dict(widget)

    def publish_as_channel(self, tenant_id: str, user_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        agent_id = str(payload.get("agent_id", ""))
        channel_type = str(payload.get("channel_type", "webhook"))

        channel = self.resources.create("agent_channels", tenant_id, user_id, {
            "name": str(payload.get("name", "")),
            "code": f"ch-{uuid.uuid4().hex[:8]}",
            "status": "active",
            "agent_id": agent_id,
            "spec": {
                "channel_type": channel_type,
                "agent_id": agent_id,
                "config": payload.get("config", {}),
                "webhook_url": str(payload.get("webhook_url", "")),
                "webhook_secret": secrets.token_hex(16),
                "channel_specific": self._get_channel_config(channel_type, payload),
                "message_transform": payload.get("message_transform", {}),
                "response_transform": payload.get("response_transform", {}),
                "enabled": True,
                "total_messages": 0,
                "last_message_at": None,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        })
        return ResourceService.to_dict(channel)

    def update_publication(self, tenant_id: str, user_id: str, pub_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        row = self.resources.update("agent_publications", tenant_id, user_id, pub_id, payload)
        return ResourceService.to_dict(row)

    def delete_publication(self, tenant_id: str, user_id: str, pub_id: str) -> dict[str, str]:
        return self.resources.delete("agent_publications", tenant_id, user_id, pub_id)

    def list_publications(self, tenant_id: str, page: int, page_size: int, filters: dict[str, Any] | None = None) -> tuple[list[dict[str, Any]], int]:
        rows, total = self.resources.list("agent_publications", tenant_id, page, page_size, filters)
        return [ResourceService.to_dict(row) for row in rows], total

    def get_publication(self, tenant_id: str, pub_id: str) -> dict[str, Any]:
        return ResourceService.to_dict(self.resources.get("agent_publications", tenant_id, pub_id))

    def list_widgets(self, tenant_id: str, page: int, page_size: int) -> tuple[list[dict[str, Any]], int]:
        rows, total = self.resources.list("agent_widgets", tenant_id, page, page_size)
        return [ResourceService.to_dict(row) for row in rows], total

    def get_widget(self, tenant_id: str, widget_id: str) -> dict[str, Any]:
        return ResourceService.to_dict(self.resources.get("agent_widgets", tenant_id, widget_id))

    def update_widget(self, tenant_id: str, user_id: str, widget_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        row = self.resources.update("agent_widgets", tenant_id, user_id, widget_id, payload)
        return ResourceService.to_dict(row)

    def delete_widget(self, tenant_id: str, user_id: str, widget_id: str) -> dict[str, str]:
        return self.resources.delete("agent_widgets", tenant_id, user_id, widget_id)

    def list_channels(self, tenant_id: str, page: int, page_size: int) -> tuple[list[dict[str, Any]], int]:
        rows, total = self.resources.list("agent_channels", tenant_id, page, page_size)
        return [ResourceService.to_dict(row) for row in rows], total

    def get_channel(self, tenant_id: str, channel_id: str) -> dict[str, Any]:
        return ResourceService.to_dict(self.resources.get("agent_channels", tenant_id, channel_id))

    def update_channel(self, tenant_id: str, user_id: str, channel_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        row = self.resources.update("agent_channels", tenant_id, user_id, channel_id, payload)
        return ResourceService.to_dict(row)

    def delete_channel(self, tenant_id: str, user_id: str, channel_id: str) -> dict[str, str]:
        return self.resources.delete("agent_channels", tenant_id, user_id, channel_id)

    def rotate_api_key(self, tenant_id: str, user_id: str, pub_id: str) -> dict[str, Any]:
        new_key = f"ak-{secrets.token_hex(24)}"
        new_hash = hashlib.sha256(new_key.encode()).hexdigest()

        pub = self.resources.get("agent_publications", tenant_id, pub_id)
        spec = dict(pub.spec or {})
        spec["api_key_hash"] = new_hash
        self.resources.update("agent_publications", tenant_id, user_id, pub_id, {"spec": spec})

        keys, _ = self.resources.list("agent_api_keys", tenant_id, 1, 100, {"parent_id": pub_id})
        for key in keys:
            self.resources.update("agent_api_keys", tenant_id, user_id, key.id, {"status": "revoked"})

        self.resources.create("agent_api_keys", tenant_id, user_id, {
            "name": f"key-rotated-{uuid.uuid4().hex[:6]}",
            "code": f"aak-{uuid.uuid4().hex[:8]}",
            "status": "active",
            "parent_id": pub_id,
            "agent_id": pub.agent_id,
            "spec": {
                "key_hash": new_hash,
                "key_prefix": new_key[:8],
                "permissions": ["chat", "stream"],
                "created_at": datetime.now(timezone.utc).isoformat(),
                "last_used_at": None,
                "total_uses": 0,
            },
        })

        return {"api_key": new_key, "publication_id": pub_id}

    def list_api_keys(self, tenant_id: str, pub_id: str) -> list[dict[str, Any]]:
        rows, _ = self.resources.list("agent_api_keys", tenant_id, 1, 100, {"parent_id": pub_id})
        result = []
        for row in rows:
            data = ResourceService.to_dict(row)
            if "spec" in data and isinstance(data["spec"], dict):
                data["spec"].pop("key_hash", None)
            result.append(data)
        return result

    @staticmethod
    def _generate_embed_code(widget_id: str) -> str:
        return (
            f'<script src="/widget/{widget_id}/loader.js" async></script>\n'
            f'<div id="aipulse-widget" data-widget-id="{widget_id}"></div>'
        )

    @staticmethod
    def _get_channel_config(channel_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        configs: dict[str, dict[str, Any]] = {
            "wechat_work": {
                "corp_id": str(payload.get("corp_id", "")),
                "agent_id": str(payload.get("wechat_agent_id", "")),
                "secret": str(payload.get("secret", "")),
                "token": str(payload.get("token", "")),
                "encoding_aes_key": str(payload.get("encoding_aes_key", "")),
            },
            "dingtalk": {
                "app_key": str(payload.get("app_key", "")),
                "app_secret": str(payload.get("app_secret", "")),
                "robot_code": str(payload.get("robot_code", "")),
            },
            "slack": {
                "bot_token": str(payload.get("bot_token", "")),
                "signing_secret": str(payload.get("signing_secret", "")),
                "app_id": str(payload.get("app_id", "")),
            },
            "telegram": {
                "bot_token": str(payload.get("bot_token", "")),
            },
            "webhook": {
                "url": str(payload.get("webhook_url", "")),
                "method": str(payload.get("method", "POST")),
                "headers": payload.get("headers", {}),
            },
        }
        return configs.get(channel_type, {})
