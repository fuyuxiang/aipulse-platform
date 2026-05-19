from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


class ConfigBridge:
    def __init__(self, echo_agent_path: Path):
        self.echo_agent_path = echo_agent_path.resolve()

    def ensure_importable(self) -> None:
        path = str(self.echo_agent_path)
        if path not in sys.path:
            sys.path.insert(0, path)

    def build_echo_config(self, workspace: Path, model_config: dict[str, Any] | None = None, tool_policy: dict[str, Any] | None = None) -> Any:
        self.ensure_importable()
        from echo_agent.config.schema import Config, ModelsConfig, ProviderConfig, ToolsConfig, ChannelsConfig, CLIChannelConfig, ObservabilityConfig

        model_config = model_config or {}
        provider = ProviderConfig(
            name=model_config.get("provider_name", "aipulse-local"),
            api_key=model_config.get("api_key", ""),
            api_base=model_config.get("api_base", ""),
            models=[model_config.get("model_name", "aipulse-local")],
            extra_headers=dict(model_config.get("extra_headers") or {}),
            timeout_seconds=int(model_config.get("timeout_seconds") or 120),
        )
        return Config(
            workspace=str(workspace),
            channels=ChannelsConfig(cli=CLIChannelConfig(enabled=False)),
            models=ModelsConfig(default_model=model_config.get("model_name", "aipulse-local"), providers=[provider]),
            tools=ToolsConfig(
                profile=tool_policy.get("profile", "coding") if tool_policy else "coding",
                allow=tool_policy.get("allow", []) if tool_policy else [],
                deny=tool_policy.get("deny", []) if tool_policy else [],
                restrict_to_workspace=True,
            ),
            observability=ObservabilityConfig(otel_enabled=False),
        )
