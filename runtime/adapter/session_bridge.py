from __future__ import annotations

from pathlib import Path


class SessionBridge:
    def __init__(self, workspace: Path):
        self.workspace = workspace

    def session_key(self, tenant_id: str, agent_id: str, version_id: str, session_id: str) -> str:
        return f"aipulse:{tenant_id}:{agent_id}:{version_id}:{session_id}"

