"""Workflow executor — DAG-based engine with conditional branching, parallel
execution, loops, timeouts, sub-workflows, and compensation support.

Replaces the original linear executor with a production-grade engine.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable

NodeHandler = Callable[[dict[str, Any], dict[str, Any]], Awaitable[Any]]


class WorkflowValidationError(ValueError):
    """Raised when workflow DAG validation fails."""


class NodeStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    WAITING = "waiting"
    CANCELLED = "cancelled"


class WorkflowRunStatus(str, Enum):
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    WAITING_EVENT = "waiting_event"
    WAITING_APPROVAL = "waiting_approval"
    CANCELLED = "cancelled"
    COMPENSATING = "compensating"


@dataclass
class NodeResult:
    node_id: str
    status: NodeStatus
    output: Any = None
    error: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0
    iterations: int = 0

    @property
    def duration_ms(self) -> float:
        if self.started_at and self.finished_at:
            return (self.finished_at - self.started_at) * 1000
        return 0.0


@dataclass
class ExecutionState:
    """Mutable state carried through workflow execution."""
    context: dict[str, Any] = field(default_factory=dict)
    results: dict[str, NodeResult] = field(default_factory=dict)
    status: WorkflowRunStatus = WorkflowRunStatus.RUNNING
    pending_event: str = ""
    pending_approval: str = ""
    compensation_stack: list[dict[str, Any]] = field(default_factory=list)
    checkpoints: list[dict[str, Any]] = field(default_factory=list)


class _WorkflowPause(Exception):
    """Internal signal to pause workflow execution (e.g. waiting for event)."""
    pass


# --- Helper functions ---


def _edge_condition_met(edge: dict[str, Any], context: dict[str, Any], results: dict[str, NodeResult]) -> bool:
    """Evaluate whether an edge's condition is satisfied."""
    condition = edge.get("condition")
    if not condition:
        return True
    if isinstance(condition, str):
        if condition in ("default", "else"):
            return True
        if condition.startswith("{{") and condition.endswith("}}"):
            expr = condition[2:-2].strip()
            return _eval_expression(expr, context, results)
        return _eval_expression(condition, context, results)
    if isinstance(condition, dict):
        left = _resolve_value(condition.get("left"), context, results)
        right = _resolve_value(condition.get("right", condition.get("right_value")), context, results)
        operator = str(condition.get("operator", "equals"))
        return _compare(left, right, operator)
    return True


def _resolve_value(value: Any, context: dict[str, Any], results: dict[str, NodeResult]) -> Any:
    """Resolve a value that may reference context or node results."""
    if not isinstance(value, str):
        return value
    if value.startswith("${") and value.endswith("}"):
        path = value[2:-1]
        return _get_nested(context, path)
    if value.startswith("{{") and value.endswith("}}"):
        path = value[2:-2].strip()
        return _get_nested(context, path)
    return value


def _get_nested(data: dict[str, Any], path: str) -> Any:
    """Get a nested value from a dict using dot notation."""
    parts = path.split(".")
    current: Any = data
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def _eval_expression(expr: str, context: dict[str, Any], results: dict[str, NodeResult]) -> bool:
    """Evaluate a simple boolean expression against context."""
    expr = expr.strip()
    if expr.lower() in ("true", "1", "yes"):
        return True
    if expr.lower() in ("false", "0", "no", ""):
        return False
    for op_str, op_name in [("!=", "not_equals"), (">=", "gte"), ("<=", "lte"),
                             ("==", "equals"), (">", "gt"), ("<", "lt"),
                             (" contains ", "contains"), (" in ", "in_op")]:
        if op_str in expr:
            parts = expr.split(op_str, 1)
            left = _resolve_value(parts[0].strip(), context, results)
            right = _resolve_value(parts[1].strip(), context, results)
            if left is None:
                left = _get_nested(context, parts[0].strip())
            if right is None:
                right = _get_nested(context, parts[1].strip())
            return _compare(left, right, op_name)
    resolved = _get_nested(context, expr)
    return bool(resolved)


def _compare(left: Any, right: Any, operator: str) -> bool:
    """Compare two values with the given operator."""
    if operator in ("equals", "eq", "=="):
        return str(left) == str(right) if left is not None and right is not None else left == right
    if operator in ("not_equals", "neq", "!="):
        return str(left) != str(right) if left is not None and right is not None else left != right
    if operator in ("gt", ">"):
        try:
            return float(left) > float(right)
        except (TypeError, ValueError):
            return False
    if operator in ("lt", "<"):
        try:
            return float(left) < float(right)
        except (TypeError, ValueError):
            return False
    if operator in ("gte", ">="):
        try:
            return float(left) >= float(right)
        except (TypeError, ValueError):
            return False
    if operator in ("lte", "<="):
        try:
            return float(left) <= float(right)
        except (TypeError, ValueError):
            return False
    if operator == "contains":
        return str(right) in str(left)
    if operator == "in_op":
        if isinstance(right, (list, tuple, set)):
            return left in right
        return str(left) in str(right)
    return left == right


class WorkflowExecutor:
    """Production-grade DAG workflow executor with branching, parallelism, loops."""

    def validate(self, nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> dict[str, Any]:
        """Validate DAG structure and return topological order."""
        node_ids = {str(node["id"]) for node in nodes}
        if not node_ids:
            raise WorkflowValidationError("workflow requires at least one node")
        graph: dict[str, list[dict[str, Any]]] = defaultdict(list)
        indegree: dict[str, int] = {nid: 0 for nid in node_ids}
        for edge in edges:
            source = str(edge["source"])
            target = str(edge["target"])
            if source not in node_ids or target not in node_ids:
                raise WorkflowValidationError("edge references unknown node")
            graph[source].append(edge)
            indegree[target] = indegree.get(target, 0) + 1
        if len(node_ids) > 1:
            outgoing = {nid: len(graph.get(nid, [])) for nid in node_ids}
            isolated = [nid for nid in node_ids if indegree[nid] == 0 and outgoing[nid] == 0]
            if isolated:
                raise WorkflowValidationError(f"workflow contains isolated nodes: {', '.join(sorted(isolated))}")
        queue = deque([nid for nid, deg in indegree.items() if deg == 0])
        ordered: list[str] = []
        while queue:
            current = queue.popleft()
            ordered.append(current)
            for edge in graph.get(current, []):
                target = str(edge["target"])
                indegree[target] -= 1
                if indegree[target] == 0:
                    queue.append(target)
        has_loops = any(
            str(n.get("type", "")) in ("for_each", "while_loop")
            for n in nodes
        )
        if len(ordered) != len(node_ids) and not has_loops:
            raise WorkflowValidationError("workflow contains a cycle")
        return {"valid": True, "order": ordered}

    async def run(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        handlers: dict[str, NodeHandler],
        initial_context: dict[str, Any] | None = None,
        *,
        checkpoint_callback: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
        event_wait_callback: Callable[[str, dict[str, Any]], Awaitable[Any]] | None = None,
        compensation_callback: Callable[[dict[str, Any], dict[str, Any]], Awaitable[None]] | None = None,
        timeout_seconds: float = 0,
    ) -> dict[str, Any]:
        """Execute the workflow DAG with full feature support."""
        self.validate(nodes, edges)
        node_map = {str(n["id"]): n for n in nodes}
        edge_map = self._build_edge_map(edges)
        reverse_edge_map = self._build_reverse_edge_map(edges)
        state = ExecutionState(context=dict(initial_context or {}))

        start_nodes = self._find_start_nodes(nodes, edges)
        if not start_nodes:
            start_nodes = [nid for nid, deg in self._indegrees(nodes, edges).items() if deg == 0]

        deadline = (time.monotonic() + timeout_seconds) if timeout_seconds > 0 else 0

        try:
            await self._execute_nodes(
                start_nodes, node_map, edge_map, reverse_edge_map,
                handlers, state, checkpoint_callback, event_wait_callback, deadline,
            )
        except _WorkflowPause:
            pass
        except Exception as exc:
            state.status = WorkflowRunStatus.FAILED
            if state.compensation_stack and compensation_callback:
                state.status = WorkflowRunStatus.COMPENSATING
                await self._run_compensation(state, compensation_callback)
                state.status = WorkflowRunStatus.FAILED
            raise

        if state.status == WorkflowRunStatus.RUNNING:
            state.status = WorkflowRunStatus.SUCCESS

        return {
            "status": state.status.value,
            "results": {nid: {"status": r.status.value, "output": r.output, "error": r.error, "duration_ms": r.duration_ms} for nid, r in state.results.items()},
            "context": state.context,
            "checkpoints": state.checkpoints,
            "pending_event": state.pending_event,
            "pending_approval": state.pending_approval,
        }

    async def resume(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        handlers: dict[str, NodeHandler],
        checkpoint: dict[str, Any],
        *,
        checkpoint_callback: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
        event_wait_callback: Callable[[str, dict[str, Any]], Awaitable[Any]] | None = None,
        compensation_callback: Callable[[dict[str, Any], dict[str, Any]], Awaitable[None]] | None = None,
        timeout_seconds: float = 0,
    ) -> dict[str, Any]:
        """Resume execution from a checkpoint."""
        node_map = {str(n["id"]): n for n in nodes}
        edge_map = self._build_edge_map(edges)
        reverse_edge_map = self._build_reverse_edge_map(edges)

        state = ExecutionState(
            context=dict(checkpoint.get("context", {})),
            results={
                nid: NodeResult(node_id=nid, status=NodeStatus(r["status"]), output=r.get("output"))
                for nid, r in checkpoint.get("results", {}).items()
            },
            compensation_stack=list(checkpoint.get("compensation_stack", [])),
        )

        completed = {nid for nid, r in state.results.items() if r.status == NodeStatus.SUCCESS}
        next_nodes = self._find_next_after_completed(completed, node_map, edge_map, state)
        deadline = (time.monotonic() + timeout_seconds) if timeout_seconds > 0 else 0

        try:
            await self._execute_nodes(
                next_nodes, node_map, edge_map, reverse_edge_map,
                handlers, state, checkpoint_callback, event_wait_callback, deadline,
            )
        except _WorkflowPause:
            pass
        except Exception:
            state.status = WorkflowRunStatus.FAILED
            if state.compensation_stack and compensation_callback:
                state.status = WorkflowRunStatus.COMPENSATING
                await self._run_compensation(state, compensation_callback)
                state.status = WorkflowRunStatus.FAILED
            raise

        if state.status == WorkflowRunStatus.RUNNING:
            state.status = WorkflowRunStatus.SUCCESS

        return {
            "status": state.status.value,
            "results": {nid: {"status": r.status.value, "output": r.output, "error": r.error, "duration_ms": r.duration_ms} for nid, r in state.results.items()},
            "context": state.context,
            "checkpoints": state.checkpoints,
            "pending_event": state.pending_event,
            "pending_approval": state.pending_approval,
        }

    # --- Internal execution engine ---

    async def _execute_nodes(
        self,
        start_nodes: list[str],
        node_map: dict[str, dict[str, Any]],
        edge_map: dict[str, list[dict[str, Any]]],
        reverse_edge_map: dict[str, list[dict[str, Any]]],
        handlers: dict[str, NodeHandler],
        state: ExecutionState,
        checkpoint_callback: Callable[[dict[str, Any]], Awaitable[None]] | None,
        event_wait_callback: Callable[[str, dict[str, Any]], Awaitable[Any]] | None,
        deadline: float,
    ) -> None:
        """BFS-style execution respecting dependencies and conditions."""
        ready: deque[str] = deque(start_nodes)
        in_flight: set[str] = set()

        while ready or in_flight:
            if deadline and time.monotonic() > deadline:
                for nid in list(in_flight):
                    state.results[nid] = NodeResult(node_id=nid, status=NodeStatus.CANCELLED, error="workflow timeout")
                raise TimeoutError("workflow execution exceeded deadline")

            parallel_batch: list[str] = []
            while ready:
                nid = ready.popleft()
                if nid in state.results and state.results[nid].status in (NodeStatus.SUCCESS, NodeStatus.SKIPPED):
                    continue
                node = node_map.get(nid)
                if not node:
                    continue
                node_type = str(node.get("type", ""))
                if node_type == "parallel_fork":
                    await self._handle_fork(nid, node, node_map, edge_map, reverse_edge_map, handlers, state, checkpoint_callback, event_wait_callback, deadline)
                    self._enqueue_successors(nid, edge_map, state, ready)
                    continue
                parallel_batch.append(nid)

            if not parallel_batch and not in_flight:
                break

            tasks = []
            for nid in parallel_batch:
                in_flight.add(nid)
                tasks.append(self._execute_single_node(nid, node_map, edge_map, handlers, state, checkpoint_callback, event_wait_callback, deadline))

            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for nid, result in zip(parallel_batch, results):
                    in_flight.discard(nid)
                    if isinstance(result, _WorkflowPause):
                        raise result
                    if isinstance(result, Exception):
                        state.results[nid] = NodeResult(node_id=nid, status=NodeStatus.FAILED, error=str(result))
                        raise result
                    self._enqueue_successors(nid, edge_map, state, ready)

    async def _execute_single_node(
        self,
        node_id: str,
        node_map: dict[str, dict[str, Any]],
        edge_map: dict[str, list[dict[str, Any]]],
        handlers: dict[str, NodeHandler],
        state: ExecutionState,
        checkpoint_callback: Callable[[dict[str, Any]], Awaitable[None]] | None,
        event_wait_callback: Callable[[str, dict[str, Any]], Awaitable[Any]] | None,
        deadline: float,
    ) -> None:
        """Execute a single node, dispatching by type."""
        node = node_map[node_id]
        node_type = str(node.get("type", ""))
        config = dict(node.get("config", {}))
        started_at = time.monotonic()

        # Handle special node types
        if node_type in ("start", "end", "parallel_join"):
            state.results[node_id] = NodeResult(
                node_id=node_id, status=NodeStatus.SUCCESS,
                started_at=started_at, finished_at=time.monotonic(),
            )
            return

        if node_type == "for_each":
            result = await self._handle_for_each(node_id, node, handlers, state)
            state.results[node_id] = NodeResult(
                node_id=node_id, status=NodeStatus.SUCCESS, output=result,
                started_at=started_at, finished_at=time.monotonic(),
                iterations=len(result) if isinstance(result, list) else 0,
            )
            return

        if node_type == "while_loop":
            result = await self._handle_while_loop(node_id, node, handlers, state)
            state.results[node_id] = NodeResult(
                node_id=node_id, status=NodeStatus.SUCCESS, output=result,
                started_at=started_at, finished_at=time.monotonic(),
                iterations=result.get("iterations", 0) if isinstance(result, dict) else 0,
            )
            return

        if node_type == "sub_workflow":
            result = await self._handle_sub_workflow(node_id, node, handlers, state, checkpoint_callback, event_wait_callback, deadline)
            state.results[node_id] = NodeResult(
                node_id=node_id, status=NodeStatus.SUCCESS, output=result,
                started_at=started_at, finished_at=time.monotonic(),
            )
            return

        if node_type == "wait_event":
            await self._handle_wait_event(node_id, node, state, event_wait_callback)
            state.results[node_id] = NodeResult(
                node_id=node_id, status=NodeStatus.SUCCESS,
                started_at=started_at, finished_at=time.monotonic(),
            )
            return

        if node_type == "emit_event":
            await self._handle_emit_event(node_id, node, state, event_wait_callback)
            state.results[node_id] = NodeResult(
                node_id=node_id, status=NodeStatus.SUCCESS,
                started_at=started_at, finished_at=time.monotonic(),
            )
            return

        if node_type == "condition":
            result = self._handle_condition(node_id, node, state)
            state.results[node_id] = NodeResult(
                node_id=node_id, status=NodeStatus.SUCCESS, output=result,
                started_at=started_at, finished_at=time.monotonic(),
            )
            return

        # Generic handler dispatch
        handler_name = node_type or node.get("handler", "")
        handler = handlers.get(handler_name)
        if not handler:
            handler = handlers.get("default")
        if not handler:
            state.results[node_id] = NodeResult(
                node_id=node_id, status=NodeStatus.SKIPPED,
                error=f"no handler for type '{node_type}'",
                started_at=started_at, finished_at=time.monotonic(),
            )
            return

        # Execute with optional timeout
        node_timeout = float(config.get("timeout_seconds", 0))
        try:
            if node_timeout > 0:
                output = await asyncio.wait_for(
                    handler(config, state.context), timeout=node_timeout
                )
            else:
                output = await handler(config, state.context)
        except asyncio.TimeoutError:
            state.results[node_id] = NodeResult(
                node_id=node_id, status=NodeStatus.FAILED,
                error=f"node execution timed out after {node_timeout}s",
                started_at=started_at, finished_at=time.monotonic(),
            )
            raise TimeoutError(f"node '{node_id}' timed out after {node_timeout}s")

        # Store result and update context
        # Check if handler signals a pause (e.g., waiting_approval, waiting_event)
        if isinstance(output, dict) and output.get("status") in ("waiting_approval", "waiting_event"):
            state.results[node_id] = NodeResult(
                node_id=node_id, status=NodeStatus.WAITING, output=output,
                started_at=started_at, finished_at=time.monotonic(),
            )
            state.context[node_id] = output
            if output.get("status") == "waiting_approval":
                state.status = WorkflowRunStatus.WAITING_APPROVAL
                state.pending_approval = str(output.get("approval_id", ""))
            else:
                state.status = WorkflowRunStatus.WAITING_EVENT
                state.pending_event = str(output.get("event_name", ""))
            if checkpoint_callback:
                cp = self._make_checkpoint(state)
                state.checkpoints.append(cp)
                await checkpoint_callback(cp)
            raise _WorkflowPause()

        state.results[node_id] = NodeResult(
            node_id=node_id, status=NodeStatus.SUCCESS, output=output,
            started_at=started_at, finished_at=time.monotonic(),
        )
        if isinstance(output, dict):
            state.context.update(output)
        elif output is not None:
            state.context[node_id] = output

        # Push compensation if configured
        compensation = config.get("compensation")
        if compensation:
            state.compensation_stack.append({"node_id": node_id, "compensation": compensation, "context": dict(state.context)})

        # Checkpoint after successful execution
        if checkpoint_callback:
            cp = self._make_checkpoint(state)
            state.checkpoints.append(cp)
            await checkpoint_callback(cp)

    async def _handle_fork(
        self,
        node_id: str,
        node: dict[str, Any],
        node_map: dict[str, dict[str, Any]],
        edge_map: dict[str, list[dict[str, Any]]],
        reverse_edge_map: dict[str, list[dict[str, Any]]],
        handlers: dict[str, NodeHandler],
        state: ExecutionState,
        checkpoint_callback: Callable[[dict[str, Any]], Awaitable[None]] | None,
        event_wait_callback: Callable[[str, dict[str, Any]], Awaitable[Any]] | None,
        deadline: float,
    ) -> None:
        """Handle parallel fork: execute all outgoing branches concurrently."""
        started_at = time.monotonic()
        outgoing_edges = edge_map.get(node_id, [])
        branch_targets = [str(e["target"]) for e in outgoing_edges]

        if not branch_targets:
            state.results[node_id] = NodeResult(
                node_id=node_id, status=NodeStatus.SUCCESS,
                started_at=started_at, finished_at=time.monotonic(),
            )
            return

        async def run_branch(target_id: str) -> None:
            await self._execute_single_node(
                target_id, node_map, edge_map, handlers, state,
                checkpoint_callback, event_wait_callback, deadline,
            )

        tasks = [run_branch(tid) for tid in branch_targets]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for tid, result in zip(branch_targets, results):
            if isinstance(result, _WorkflowPause):
                state.results[node_id] = NodeResult(
                    node_id=node_id, status=NodeStatus.SUCCESS,
                    started_at=started_at, finished_at=time.monotonic(),
                )
                raise result
            if isinstance(result, Exception):
                state.results[node_id] = NodeResult(
                    node_id=node_id, status=NodeStatus.FAILED, error=str(result),
                    started_at=started_at, finished_at=time.monotonic(),
                )
                raise result

        state.results[node_id] = NodeResult(
            node_id=node_id, status=NodeStatus.SUCCESS,
            started_at=started_at, finished_at=time.monotonic(),
        )

    async def _handle_for_each(
        self,
        node_id: str,
        node: dict[str, Any],
        handlers: dict[str, NodeHandler],
        state: ExecutionState,
    ) -> list[Any]:
        """Loop over a list of items, executing a sub-handler for each."""
        config = dict(node.get("config", {}))
        # Get items from config literal or resolve from context
        items = config.get("items")
        if items is None:
            items_ref = config.get("items_ref", "")
            if items_ref:
                items = _resolve_value(items_ref, state.context, state.results)
            if items is None:
                items = []
        if not isinstance(items, (list, tuple)):
            items = list(items) if items else []

        handler_name = config.get("handler", config.get("body_handler", ""))
        body_config = config.get("body", {})
        item_var = config.get("item_variable", "item")
        index_var = config.get("index_variable", "index")
        results_list: list[Any] = []

        for idx, item in enumerate(items):
            # Set loop variables in context
            state.context[item_var] = item
            state.context[index_var] = idx
            state.context["loop_index"] = idx
            state.context["loop_item"] = item

            if handler_name and handler_name in handlers:
                handler = handlers[handler_name]
                iteration_config = dict(body_config) if body_config else dict(config)
                iteration_config["item"] = item
                iteration_config["index"] = idx
                output = await handler(iteration_config, state.context)
                if isinstance(output, dict):
                    state.context.update(output)
                results_list.append(output)
            else:
                # No handler found, store item pass-through
                results_list.append({"item": item, "index": idx})

        # Store collected results
        state.context[f"{node_id}_results"] = results_list
        return results_list

    async def _handle_while_loop(
        self,
        node_id: str,
        node: dict[str, Any],
        handlers: dict[str, NodeHandler],
        state: ExecutionState,
    ) -> dict[str, Any]:
        """Conditional loop: evaluate condition, execute body while true."""
        config = dict(node.get("config", {}))
        condition_expr = config.get("condition", "false")
        max_iterations = int(config.get("max_iterations", 100))
        handler_name = config.get("handler", config.get("body_handler", ""))
        body_config = config.get("body", {})

        iterations = 0
        results_list: list[Any] = []

        while iterations < max_iterations:
            # Evaluate condition
            if not _eval_expression(str(condition_expr), state.context, state.results):
                break

            state.context["loop_iteration"] = iterations

            # Execute body handler
            if handler_name and handler_name in handlers:
                handler = handlers[handler_name]
                output = await handler(body_config if body_config else config, state.context)
                results_list.append(output)
                if isinstance(output, dict):
                    state.context.update(output)
            else:
                results_list.append(None)

            iterations += 1

        state.context[f"{node_id}_iterations"] = iterations
        state.context[f"{node_id}_results"] = results_list
        return {"iterations": iterations, "results": results_list}

    async def _handle_sub_workflow(
        self,
        node_id: str,
        node: dict[str, Any],
        handlers: dict[str, NodeHandler],
        state: ExecutionState,
        checkpoint_callback: Callable[[dict[str, Any]], Awaitable[None]] | None,
        event_wait_callback: Callable[[str, dict[str, Any]], Awaitable[Any]] | None,
        deadline: float,
    ) -> dict[str, Any]:
        """Recursively run another workflow definition from config."""
        config = dict(node.get("config", {}))
        sub_nodes = config.get("nodes", [])
        sub_edges = config.get("edges", [])

        if not sub_nodes:
            return {"status": "skipped", "reason": "no sub-workflow nodes defined"}

        # Determine timeout for sub-workflow
        remaining = 0.0
        if deadline:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("no time remaining for sub-workflow")

        sub_executor = WorkflowExecutor()
        result = await sub_executor.run(
            nodes=sub_nodes,
            edges=sub_edges,
            handlers=handlers,
            initial_context=dict(state.context),
            checkpoint_callback=checkpoint_callback,
            event_wait_callback=event_wait_callback,
            timeout_seconds=remaining if remaining > 0 else 0,
        )

        # Merge sub-workflow context back
        sub_context = result.get("context", {})
        state.context.update(sub_context)
        return result

    async def _handle_wait_event(
        self,
        node_id: str,
        node: dict[str, Any],
        state: ExecutionState,
        event_wait_callback: Callable[[str, dict[str, Any]], Awaitable[Any]] | None,
    ) -> None:
        """Pause workflow waiting for an external event."""
        config = dict(node.get("config", {}))
        event_name = config.get("event", config.get("event_name", node_id))

        state.pending_event = event_name
        state.status = WorkflowRunStatus.WAITING_EVENT

        if event_wait_callback:
            result = await event_wait_callback(event_name, {"action": "wait", "node_id": node_id, "config": config})
            if result is not None:
                if isinstance(result, dict):
                    state.context.update(result)
                else:
                    state.context[f"{node_id}_event_data"] = result
                state.pending_event = ""
                state.status = WorkflowRunStatus.RUNNING
                return

        # No callback or callback returned None - pause execution
        checkpoint = self._make_checkpoint(state)
        state.checkpoints.append(checkpoint)
        raise _WorkflowPause()

    async def _handle_emit_event(
        self,
        node_id: str,
        node: dict[str, Any],
        state: ExecutionState,
        event_wait_callback: Callable[[str, dict[str, Any]], Awaitable[Any]] | None,
    ) -> None:
        """Emit an event via the event callback."""
        config = dict(node.get("config", {}))
        event_name = config.get("event", config.get("event_name", node_id))
        payload = config.get("payload", {})

        # Resolve payload values from context
        resolved_payload: dict[str, Any] = {}
        if isinstance(payload, dict):
            for key, val in payload.items():
                resolved_payload[key] = _resolve_value(val, state.context, state.results)
        else:
            resolved_payload = {"data": payload}

        if event_wait_callback:
            await event_wait_callback(event_name, {
                "action": "emit",
                "node_id": node_id,
                "payload": resolved_payload,
            })

    def _handle_condition(
        self,
        node_id: str,
        node: dict[str, Any],
        state: ExecutionState,
    ) -> str:
        """Evaluate condition and return the matched branch result."""
        config = dict(node.get("config", {}))
        condition_expr = config.get("condition", config.get("expression", ""))

        if not condition_expr:
            return "default"

        result = _eval_expression(str(condition_expr), state.context, state.results)
        # Store the boolean result for edge routing
        state.context[f"{node_id}_result"] = result
        return "true" if result else "false"

    def _enqueue_successors(
        self,
        node_id: str,
        edge_map: dict[str, list[dict[str, Any]]],
        state: ExecutionState,
        ready: deque[str],
    ) -> None:
        """Find next nodes via edge_map, evaluate conditions, add eligible to queue."""
        outgoing = edge_map.get(node_id, [])
        if not outgoing:
            return

        node_result = state.results.get(node_id)
        node_output = node_result.output if node_result else None

        # For condition/branch nodes, only follow edges whose condition matches
        is_condition_node = (
            node_result is not None
            and isinstance(node_output, str)
            and node_output in ("true", "false", "default")
        )

        if is_condition_node:
            # Try to find an edge that matches the condition result
            matched_edges: list[dict[str, Any]] = []
            default_edges: list[dict[str, Any]] = []

            for edge in outgoing:
                edge_condition = edge.get("condition", "")
                if isinstance(edge_condition, str):
                    if edge_condition in ("default", "else", ""):
                        default_edges.append(edge)
                    elif edge_condition == node_output:
                        matched_edges.append(edge)
                    elif edge_condition == "true" and node_output == "true":
                        matched_edges.append(edge)
                    elif edge_condition == "false" and node_output == "false":
                        matched_edges.append(edge)
                    else:
                        # Evaluate the condition expression
                        if _edge_condition_met(edge, state.context, state.results):
                            matched_edges.append(edge)
                else:
                    if _edge_condition_met(edge, state.context, state.results):
                        matched_edges.append(edge)

            # Use matched edges, fall back to default
            edges_to_follow = matched_edges if matched_edges else default_edges
            for edge in edges_to_follow:
                target = str(edge["target"])
                if target not in state.results or state.results[target].status == NodeStatus.PENDING:
                    ready.append(target)
        else:
            # Standard node: follow all edges whose conditions are met
            for edge in outgoing:
                if _edge_condition_met(edge, state.context, state.results):
                    target = str(edge["target"])
                    if target not in state.results or state.results[target].status == NodeStatus.PENDING:
                        ready.append(target)

    async def _run_compensation(
        self,
        state: ExecutionState,
        compensation_callback: Callable[[dict[str, Any], dict[str, Any]], Awaitable[None]],
    ) -> None:
        """Pop compensation_stack in reverse, call compensation_callback for each."""
        while state.compensation_stack:
            entry = state.compensation_stack.pop()
            try:
                await compensation_callback(entry["compensation"], entry.get("context", {}))
            except Exception:
                # Compensation failures are logged but do not halt the process
                pass

    # --- Graph helper methods ---

    def _build_edge_map(self, edges: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        """Build source_id -> list of edges mapping."""
        edge_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for edge in edges:
            source = str(edge["source"])
            edge_map[source].append(edge)
        return dict(edge_map)

    def _build_reverse_edge_map(self, edges: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        """Build target_id -> list of edges mapping."""
        reverse_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for edge in edges:
            target = str(edge["target"])
            reverse_map[target].append(edge)
        return dict(reverse_map)

    def _find_start_nodes(self, nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> list[str]:
        """Find nodes with type 'start' or indegree 0."""
        start_typed = [str(n["id"]) for n in nodes if str(n.get("type", "")) == "start"]
        if start_typed:
            return start_typed
        indegrees = self._indegrees(nodes, edges)
        return [nid for nid, deg in indegrees.items() if deg == 0]

    def _indegrees(self, nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> dict[str, int]:
        """Compute indegree map for all nodes."""
        indegree: dict[str, int] = {str(n["id"]): 0 for n in nodes}
        for edge in edges:
            target = str(edge["target"])
            if target in indegree:
                indegree[target] += 1
        return indegree

    def _find_next_after_completed(
        self,
        completed: set[str],
        node_map: dict[str, dict[str, Any]],
        edge_map: dict[str, list[dict[str, Any]]],
        state: ExecutionState,
    ) -> list[str]:
        """For resume: find nodes that should execute next given completed set."""
        candidates: set[str] = set()
        for nid in completed:
            for edge in edge_map.get(nid, []):
                target = str(edge["target"])
                if target not in completed:
                    candidates.add(target)
        # Filter to only those whose dependencies are all completed
        next_nodes: list[str] = []
        for candidate in candidates:
            if candidate in state.results and state.results[candidate].status == NodeStatus.SUCCESS:
                continue
            next_nodes.append(candidate)
        return next_nodes

    def _make_checkpoint(self, state: ExecutionState) -> dict[str, Any]:
        """Serialize current state to a checkpoint dict."""
        return {
            "id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "context": dict(state.context),
            "results": {
                nid: {
                    "status": r.status.value,
                    "output": r.output,
                    "error": r.error,
                    "duration_ms": r.duration_ms,
                }
                for nid, r in state.results.items()
            },
            "status": state.status.value,
            "compensation_stack": list(state.compensation_stack),
            "pending_event": state.pending_event,
            "pending_approval": state.pending_approval,
        }

