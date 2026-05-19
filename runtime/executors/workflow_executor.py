from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Awaitable, Callable


class WorkflowValidationError(ValueError):
    """Raised when workflow DAG validation fails."""


class WorkflowExecutor:
    def validate(self, nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> dict[str, Any]:
        node_ids = {str(node["id"]) for node in nodes}
        if not node_ids:
            raise WorkflowValidationError("workflow requires at least one node")
        graph: dict[str, list[str]] = defaultdict(list)
        indegree = {node_id: 0 for node_id in node_ids}
        outgoing = {node_id: 0 for node_id in node_ids}
        for edge in edges:
            source = str(edge["source"])
            target = str(edge["target"])
            if source not in node_ids or target not in node_ids:
                raise WorkflowValidationError("edge references unknown node")
            graph[source].append(target)
            indegree[target] += 1
            outgoing[source] += 1
        if len(node_ids) > 1:
            isolated = [node_id for node_id in node_ids if indegree[node_id] == 0 and outgoing[node_id] == 0]
            if isolated:
                raise WorkflowValidationError(f"workflow contains isolated nodes: {', '.join(sorted(isolated))}")
        queue = deque([node_id for node_id, degree in indegree.items() if degree == 0])
        ordered: list[str] = []
        while queue:
            current = queue.popleft()
            ordered.append(current)
            for child in graph[current]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
        if len(ordered) != len(node_ids):
            raise WorkflowValidationError("workflow contains a cycle")
        return {"valid": True, "order": ordered}

    async def run(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        handlers: dict[str, Callable[[dict[str, Any], dict[str, Any]], Awaitable[Any]]],
        initial_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        plan = self.validate(nodes, edges)
        by_id = {str(node["id"]): node for node in nodes}
        context = dict(initial_context or {})
        results: dict[str, Any] = {}
        for node_id in plan["order"]:
            node = by_id[node_id]
            node_type = str(node.get("type", "script"))
            handler = handlers.get(node_type) or handlers.get("default")
            if handler is None:
                raise WorkflowValidationError(f"missing handler for node type {node_type}")
            results[node_id] = await handler(node, context)
            context[node_id] = results[node_id]
        return {"status": "success", "results": results, "context": context}
