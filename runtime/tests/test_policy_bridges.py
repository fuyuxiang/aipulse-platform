from types import SimpleNamespace
import pytest
from runtime.adapter.memory_bridge import MemoryBridge
from runtime.adapter.tool_policy_bridge import ToolPolicyBridge
from runtime.adapter.model_bridge import ModelBridge


class Tools:
    def __init__(self):
        self.tool_names = ['read', 'exec', 'network']

    def unregister(self, name):
        self.tool_names.remove(name)


def test_tool_allowlist_is_enforced_and_deny_wins():
    tools = Tools()
    ToolPolicyBridge().apply(SimpleNamespace(tools=tools), {'allow': ['read', 'exec'], 'deny': ['exec']})
    assert tools.tool_names == ['read']
    tools = Tools()
    ToolPolicyBridge().apply(SimpleNamespace(tools=tools), {})
    assert tools.tool_names == []


def test_disabled_memory_and_unsupported_provider_fail_closed():
    config = SimpleNamespace(memory=SimpleNamespace(enabled=True))
    MemoryBridge().apply_policy(SimpleNamespace(config=config), {'enabled': False})
    assert not config.memory.enabled
    bridge = ModelBridge(SimpleNamespace(ensure_importable=lambda: None))
    with pytest.raises(ValueError, match='unsupported runtime provider'):
        bridge.build_provider({'provider_type': 'unknown-real-provider'})
