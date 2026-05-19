from __future__ import annotations

import asyncio
import hashlib
import math
import os
from typing import Any

import httpx

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
        model_config = model_config or {}
        provider_type = str(model_config.get("provider_type") or "").lower()
        if provider_type in {"openai", "openai_compatible", "azure_openai"} or model_config.get("api_style") == "openai":
            return OpenAICompatibleProvider(model_config).provider
        return LocalEnterpriseProvider(model_config.get("model_name", "aipulse-local")).provider


class OpenAICompatibleProvider:
    def __init__(self, config: dict[str, Any]):
        from echo_agent.models.provider import LLMProvider

        api_key = _api_key(config)
        api_base = str(config.get("api_base") or config.get("base_url") or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        default_model = str(config.get("model_name") or config.get("model") or "")
        timeout = float(config.get("timeout_seconds") or 120)
        extra_headers = dict(config.get("extra_headers") or {})

        class _Provider(LLMProvider):
            async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, model: str | None = None, tool_choice: str | dict | None = None, **kwargs: Any) -> Any:
                from echo_agent.models.provider import LLMResponse, ToolCallRequest

                if not api_key:
                    raise RuntimeError("OpenAI-compatible provider requires api_key_env/api_key")
                model_name = model or default_model
                if not model_name:
                    raise RuntimeError("OpenAI-compatible provider requires model_name")
                body: dict[str, Any] = {
                    "model": model_name,
                    "messages": messages,
                    "temperature": kwargs.get("temperature", config.get("temperature", 0.7)),
                    "max_tokens": kwargs.get("max_tokens", config.get("max_tokens", 4096)),
                }
                if tools:
                    body["tools"] = tools
                if tool_choice:
                    body["tool_choice"] = tool_choice
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", **extra_headers}
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(f"{api_base}/chat/completions", headers=headers, json=body)
                response.raise_for_status()
                data = response.json()
                choice = (data.get("choices") or [{}])[0]
                message = dict(choice.get("message") or {})
                usage = dict(data.get("usage") or {})
                tool_calls = []
                for call in message.get("tool_calls") or []:
                    function = dict(call.get("function") or {})
                    arguments = function.get("arguments") or {}
                    if isinstance(arguments, str):
                        import json

                        arguments = json.loads(arguments or "{}")
                    tool_calls.append(ToolCallRequest(id=str(call.get("id") or ""), name=str(function.get("name") or ""), arguments=dict(arguments)))
                return LLMResponse(
                    content=message.get("content"),
                    tool_calls=tool_calls,
                    finish_reason=str(choice.get("finish_reason") or "stop"),
                    usage={
                        "input_tokens": int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
                        "output_tokens": int(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
                    },
                    model=str(data.get("model") or model_name),
                )

            def get_default_model(self) -> str:
                return default_model

            async def embed(self, text: str, model: str | None = None) -> list[float] | None:
                if not api_key:
                    raise RuntimeError("OpenAI-compatible provider requires api_key_env/api_key")
                model_name = model or str(config.get("embedding_model_name") or default_model)
                body = {"model": model_name, "input": text}
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", **extra_headers}
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(f"{api_base}/embeddings", headers=headers, json=body)
                response.raise_for_status()
                data = response.json()
                return list((data.get("data") or [{}])[0].get("embedding") or [])

        self.provider = _Provider(api_key=api_key, api_base=api_base)


def _api_key(config: dict[str, Any]) -> str:
    env_name = str(config.get("api_key_env") or config.get("secret_env") or "")
    if env_name:
        return os.getenv(env_name, "")
    return str(config.get("api_key") or config.get("secret") or os.getenv("OPENAI_API_KEY") or "")
