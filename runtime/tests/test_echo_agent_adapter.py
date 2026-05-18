from __future__ import annotations

import pytest

from runtime.adapter.echo_agent_adapter import EchoAgentRuntimeAdapter
from runtime.adapter.lifecycle import RuntimeContext


@pytest.mark.asyncio
async def test_adapter_lifecycle_and_debug_run(tmp_path) -> None:
    adapter = EchoAgentRuntimeAdapter(tmp_path / "missing-echo-agent", tmp_path / "data")
    assert adapter is not None
    real_adapter = EchoAgentRuntimeAdapter(__import__("pathlib").Path(__file__).resolve().parents[2] / "echo-agent", tmp_path / "data")
    context = RuntimeContext(tenant_id="t1", agent_id="a1", version_id="v1", session_id="s1", workspace=str(tmp_path / "ws"))
    instance = await real_adapter.create(context)
    assert instance.status == "created"
    started = await real_adapter.start(instance.id)
    assert started.status == "running"
    result = await real_adapter.debug_run(instance.id, "hello", "s1")
    assert "response" in result
    stopped = await real_adapter.stop(instance.id)
    assert stopped.status == "stopped"

