from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ResourceCreate(BaseModel):
    name: str = ""
    code: str = ""
    description: str = ""
    status: str = "active"
    enabled: bool = True
    resource_type: str = ""
    parent_id: str = ""
    owner_id: str = ""
    version: str = ""
    model_type: str = ""
    provider_type: str = ""
    provider_id: str = ""
    model_id: str = ""
    agent_id: str = ""
    workflow_id: str = ""
    session_id: str = ""
    user_id: str = ""
    knowledge_base_id: str = ""
    tool_name: str = ""
    config: dict[str, Any] = Field(default_factory=dict)
    spec: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    input_payload: dict[str, Any] = Field(default_factory=dict)


class ResourceUpdate(BaseModel):
    name: str | None = None
    code: str | None = None
    description: str | None = None
    status: str | None = None
    enabled: bool | None = None
    resource_type: str | None = None
    parent_id: str | None = None
    owner_id: str | None = None
    version: str | None = None
    model_type: str | None = None
    provider_type: str | None = None
    provider_id: str | None = None
    model_id: str | None = None
    agent_id: str | None = None
    workflow_id: str | None = None
    session_id: str | None = None
    user_id: str | None = None
    knowledge_base_id: str | None = None
    tool_name: str | None = None
    config: dict[str, Any] | None = None
    spec: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    input_payload: dict[str, Any] | None = None


class ResourceRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    tenant_id: str
    name: str = ""
    code: str = ""
    description: str = ""
    status: str = ""
    enabled: bool = True
    resource_type: str = ""
    parent_id: str = ""
    owner_id: str = ""
    version: str = ""
    model_type: str = ""
    provider_type: str = ""
    provider_id: str = ""
    model_id: str = ""
    agent_id: str = ""
    workflow_id: str = ""
    session_id: str = ""
    user_id: str = ""
    knowledge_base_id: str = ""
    tool_name: str = ""
    trace_id: str = ""
    latency_ms: int = 0
    cost: float = 0.0
    token_usage: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    spec: dict[str, Any] = Field(default_factory=dict)
    metadata_json: dict[str, Any] = Field(default_factory=dict)
    input_payload: dict[str, Any] = Field(default_factory=dict)
    output_payload: dict[str, Any] = Field(default_factory=dict)
    error_code: str = ""
    error_message: str = ""
    created_at: datetime
    updated_at: datetime
    created_by: str = ""
    updated_by: str = ""


class ActionRequest(BaseModel):
    action: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)


class ActionResponse(BaseModel):
    id: str
    action: str
    status: str
    resource_type: str
    resource_id: str = ""
    output: dict[str, Any] = Field(default_factory=dict)

