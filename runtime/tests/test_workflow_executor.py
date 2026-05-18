from __future__ import annotations

import pytest

from runtime.executors.workflow_executor import WorkflowExecutor, WorkflowValidationError


@pytest.mark.asyncio
async def test_workflow_executor_orders_and_runs_nodes() -> None:
    executor = WorkflowExecutor()
    nodes = [{"id": "a", "type": "default"}, {"id": "b", "type": "default"}]
    edges = [{"source": "a", "target": "b"}]

    async def handler(node, context):
        return {"node": node["id"], "seen": sorted(context)}

    result = await executor.run(nodes, edges, {"default": handler})
    assert result["status"] == "success"
    assert result["results"]["b"]["seen"] == ["a"]


def test_workflow_executor_rejects_cycles() -> None:
    executor = WorkflowExecutor()
    with pytest.raises(WorkflowValidationError):
        executor.validate([{"id": "a"}, {"id": "b"}], [{"source": "a", "target": "b"}, {"source": "b", "target": "a"}])

