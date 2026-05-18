from __future__ import annotations

import asyncio
import hashlib
import math
from typing import Any

from runtime.adapter.config_bridge import ConfigBridge


class LocalEnterpriseProvider:
    def __init__(self, default_model: str = "aipulse-local"):
        from echo_agent.models.provider import LLMProvider

        class _Provider(LLMProvider):
            async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, model: str | None = None, tool_choice: str | dict | None = None, **kwargs: Any) -> Any:
                from echo_agent.models.provider import LLMResponse

                await asyncio.sleep(0)
                text = "\n".join(str(item.get("content", "")) for item in messages)
                return LLMResponse(content=f"[{model or default_model}] {text[:1200]}", model=model or default_model, usage={"input_tokens": len(text.split()), "output_tokens": min(128, max(1, len(text.split())))})

            def get_default_model(self) -> str:
                return default_model

            async def embed(self, text: str, model: str | None = None) -> list[float] | None:
                values: list[float] = []
                seed = hashlib.sha256(text.encode("utf-8")).digest()
                counter = 0
                while len(values) < 128:
                    values.extend(((byte / 127.5) - 1.0) for byte in hashlib.sha256(seed + counter.to_bytes(4, "big")).digest())
                    counter += 1
                vector = values[:128]
                norm = math.sqrt(sum(item * item for item in vector)) or 1.0
                return [item / norm for item in vector]

        self.provider = _Provider()


class ModelBridge:
    def __init__(self, config_bridge: ConfigBridge):
        self.config_bridge = config_bridge

    def build_provider(self, model_config: dict[str, Any] | None = None) -> Any:
        self.config_bridge.ensure_importable()
        return LocalEnterpriseProvider((model_config or {}).get("model_name", "aipulse-local")).provider
