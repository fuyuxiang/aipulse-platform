"""Workflow engine — DAG-based multi-step orchestration on top of TaskManager.

Supports: dependency resolution, conditional steps, parallel execution,
loops (for_each/while), compensation on failure, and checkpoint/resume.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from loguru import logger

from echo_agent.tasks.models import (
    WorkflowRecord,
    WorkflowStatus,
    StepDefinition,
    TaskStatus,
    VALID_WORKFLOW_TRANSITIONS,
    TERMINAL_TASK_STATUSES,
    TERMINAL_WORKFLOW_STATUSES,
    _now,
)
from echo_agent.tasks.manager import TaskManager


class WorkflowEngine:
    """Manages workflow lifecycle with DAG resolution, conditions, parallelism, and compensation."""

    def __init__(self, storage: Any, task_manager: TaskManager):
        self._storage = storage
        self._tasks = task_manager

    async def create(
        self,
        name: str,
        steps: list[dict[str, Any]],
        description: str = "",
    ) -> WorkflowRecord:
        step_defs = []
        for i, s in enumerate(steps):
            sd = StepDefinition.from_dict(s)
            if not sd.id:
                sd.id = f"step_{i}"
            if not sd.name:
                sd.name = sd.tool_name or sd.id
            step_defs.append(sd)
        wf = WorkflowRecord(name=name, description=description, steps=step_defs)
        await self._storage.store_workflow(wf.id, wf.to_dict())
        logger.info("Workflow created: {} '{}'", wf.id, name)
        return wf

    async def get(self, workflow_id: str) -> WorkflowRecord | None:
        data = await self._storage.load_workflow(workflow_id)
        if not data:
            return None
        return WorkflowRecord.from_dict(data)

    async def _save(self, wf: WorkflowRecord) -> None:
        wf.updated_at = _now()
        await self._storage.store_workflow(wf.id, wf.to_dict())

    def _transition(self, wf: WorkflowRecord, new_status: WorkflowStatus) -> None:
        allowed = VALID_WORKFLOW_TRANSITIONS.get(wf.status, set())
        if new_status not in allowed:
            raise ValueError(f"Invalid workflow transition: {wf.status.value} → {new_status.value}")
        wf.status = new_status

    async def start(self, workflow_id: str) -> WorkflowRecord:
        wf = await self.get(workflow_id)
        if not wf:
            raise ValueError(f"Workflow '{workflow_id}' not found")
        self._transition(wf, WorkflowStatus.RUNNING)
        await self._queue_eligible_steps(wf)
        await self._save(wf)
        logger.info("Workflow {} started", workflow_id)
        return wf

    async def advance(self, workflow_id: str) -> WorkflowRecord:
        """Advance workflow state: check completed tasks, queue next steps."""
        wf = await self.get(workflow_id)
        if not wf:
            raise ValueError(f"Workflow '{workflow_id}' not found")
        if wf.status in TERMINAL_WORKFLOW_STATUSES:
            return wf

        tasks = await self._tasks.list_by_workflow(workflow_id)
        task_map = {t.id: t for t in tasks}
        step_status: dict[str, TaskStatus | None] = {}
        for step in wf.steps:
            tid = wf.step_tasks.get(step.id)
            step_status[step.id] = task_map[tid].status if tid and tid in task_map else None

        any_failed = any(s == TaskStatus.FAILED for s in step_status.values())
        all_done = all(
            s in TERMINAL_TASK_STATUSES
            for s in step_status.values()
            if s is not None
        ) and len(step_status) > 0 and all(s is not None for s in step_status.values())

        if any_failed:
            failed_steps = [sid for sid, s in step_status.items() if s == TaskStatus.FAILED]
            retried = await self._retry_failed_steps(wf, failed_steps, task_map)
            if not retried:
                await self._run_compensation(wf, task_map)
                self._transition(wf, WorkflowStatus.FAILED)
                await self._save(wf)
                return wf

        if all_done:
            self._transition(wf, WorkflowStatus.SUCCESS)
            await self._save(wf)
            logger.info("Workflow {} completed successfully", workflow_id)
            return wf

        await self._queue_eligible_steps(wf)
        await self._save(wf)
        return wf

    async def on_task_complete(self, task_id: str) -> None:
        task = await self._tasks.get(task_id)
        if not task or not task.workflow_id:
            return
        wf = await self.get(task.workflow_id)
        if wf:
            wf.state[f"result_{task_id}"] = task.result
            await self._save(wf)
        await self.advance(task.workflow_id)

    async def pause(self, workflow_id: str) -> WorkflowRecord:
        wf = await self.get(workflow_id)
        if not wf:
            raise ValueError(f"Workflow '{workflow_id}' not found")
        self._transition(wf, WorkflowStatus.WAITING)
        await self._save(wf)
        return wf

    async def resume(self, workflow_id: str) -> WorkflowRecord:
        wf = await self.get(workflow_id)
        if not wf:
            raise ValueError(f"Workflow '{workflow_id}' not found")
        self._transition(wf, WorkflowStatus.RUNNING)
        await self._queue_eligible_steps(wf)
        await self._save(wf)
        return wf

    async def cancel(self, workflow_id: str) -> WorkflowRecord:
        wf = await self.get(workflow_id)
        if not wf:
            raise ValueError(f"Workflow '{workflow_id}' not found")
        self._transition(wf, WorkflowStatus.CANCELLED)
        for tid in wf.step_tasks.values():
            t = await self._tasks.get(tid)
            if t and t.status not in TERMINAL_TASK_STATUSES:
                try:
                    await self._tasks.cancel(tid)
                except ValueError:
                    pass
        await self._save(wf)
        return wf

    async def list_all(self, status: str | None = None) -> list[WorkflowRecord]:
        rows = await self._storage.list_workflows(status=status)
        return [WorkflowRecord.from_dict(r) for r in rows]

    # ── Condition evaluation ───────────────────────────────────────────────

    def _evaluate_condition(self, condition: str, wf: WorkflowRecord) -> bool:
        """Evaluate a step condition expression against workflow state."""
        if not condition:
            return True
        condition = condition.strip()
        if condition.lower() in ("true", "1", "yes"):
            return True
        if condition.lower() in ("false", "0", "no"):
            return False

        for op_str in ("!=", ">=", "<=", "==", ">", "<", " contains ", " in "):
            if op_str in condition:
                parts = condition.split(op_str, 1)
                left = self._resolve_state_value(parts[0].strip(), wf)
                right = self._resolve_state_value(parts[1].strip(), wf)
                return self._compare(left, right, op_str.strip())

        resolved = self._resolve_state_value(condition, wf)
        return bool(resolved)

    def _resolve_state_value(self, ref: str, wf: WorkflowRecord) -> Any:
        """Resolve a reference to a value in workflow state."""
        if ref.startswith("${") and ref.endswith("}"):
            path = ref[2:-1]
        elif ref.startswith("state."):
            path = ref[6:]
        else:
            return ref

        parts = path.split(".")
        current: Any = wf.state
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None
        return current

    @staticmethod
    def _compare(left: Any, right: Any, operator: str) -> bool:
        if operator in ("==", "equals"):
            return str(left) == str(right)
        if operator in ("!=", "not_equals"):
            return str(left) != str(right)
        if operator in (">", "gt"):
            try:
                return float(left) > float(right)
            except (TypeError, ValueError):
                return False
        if operator in ("<", "lt"):
            try:
                return float(left) < float(right)
            except (TypeError, ValueError):
                return False
        if operator in (">=",):
            try:
                return float(left) >= float(right)
            except (TypeError, ValueError):
                return False
        if operator in ("<=",):
            try:
                return float(left) <= float(right)
            except (TypeError, ValueError):
                return False
        if operator == "contains":
            return str(right) in str(left)
        if operator == "in":
            return str(left) in str(right)
        return left == right

    # ── Step queuing with parallelism ──────────────────────────────────────

    async def _queue_eligible_steps(self, wf: WorkflowRecord) -> None:
        """Queue all steps whose dependencies are met and conditions pass."""
        tasks = await self._tasks.list_by_workflow(wf.id)
        task_map = {t.id: t for t in tasks}
        completed_steps: set[str] = set()
        active_steps: set[str] = set()
        for step in wf.steps:
            tid = wf.step_tasks.get(step.id)
            if tid and tid in task_map:
                t = task_map[tid]
                if t.status == TaskStatus.SUCCESS:
                    completed_steps.add(step.id)
                elif t.status not in TERMINAL_TASK_STATUSES:
                    active_steps.add(step.id)

        queued_this_round: list[str] = []
        for step in wf.steps:
            if step.id in completed_steps or step.id in active_steps:
                continue
            if wf.step_tasks.get(step.id):
                continue
            deps_met = all(d in completed_steps for d in step.depends_on)
            if not deps_met:
                continue
            if step.condition and not self._evaluate_condition(step.condition, wf):
                wf.step_tasks[step.id] = "skipped"
                continue

            loop_config = step.tool_params.get("loop_config")
            if loop_config:
                await self._handle_loop_step(wf, step, loop_config, completed_steps)
                continue

            task = await self._tasks.create(
                title=step.name,
                description=f"Workflow step: {step.tool_name}({step.tool_params})",
                workflow_id=wf.id,
                max_retries=step.retry_max,
                metadata={
                    "step_id": step.id,
                    "tool_name": step.tool_name,
                    "tool_params": step.tool_params,
                    "timeout_seconds": step.timeout_seconds,
                    "compensation": step.tool_params.get("compensation"),
                },
            )
            wf.step_tasks[step.id] = task.id
            wf.current_step = step.id
            queued_this_round.append(step.id)

        if queued_this_round:
            logger.info("Workflow {} queued steps: {}", wf.id, queued_this_round)

    # ── Loop handling ──────────────────────────────────────────────────────

    async def _handle_loop_step(
        self, wf: WorkflowRecord, step: StepDefinition,
        loop_config: dict[str, Any], completed_steps: set[str],
    ) -> None:
        """Handle for_each and while loop steps."""
        loop_type = loop_config.get("type", "for_each")
        max_iterations = int(loop_config.get("max_iterations", 100))

        if loop_type == "for_each":
            items_ref = loop_config.get("items_ref", "")
            items = loop_config.get("items", [])
            if items_ref:
                resolved = self._resolve_state_value(f"${{{items_ref}}}", wf)
                if isinstance(resolved, list):
                    items = resolved

            results = []
            for i, item in enumerate(items[:max_iterations]):
                wf.state[f"loop_{step.id}_item"] = item
                wf.state[f"loop_{step.id}_index"] = i
                task = await self._tasks.create(
                    title=f"{step.name} [{i}]",
                    description=f"Loop iteration {i}: {step.tool_name}",
                    workflow_id=wf.id,
                    max_retries=step.retry_max,
                    metadata={
                        "step_id": step.id,
                        "tool_name": step.tool_name,
                        "tool_params": {**step.tool_params, "item": item, "index": i},
                        "loop_iteration": i,
                    },
                )
                results.append(task.id)

            wf.step_tasks[step.id] = results[0] if results else "skipped"
            wf.state[f"loop_{step.id}_tasks"] = results

        elif loop_type == "while":
            condition = loop_config.get("condition", "false")
            iteration = int(wf.state.get(f"while_{step.id}_iteration", 0))

            if iteration >= max_iterations:
                wf.step_tasks[step.id] = "skipped"
                logger.warning("While loop {} hit max iterations", step.id)
                return

            if not self._evaluate_condition(condition, wf):
                wf.step_tasks[step.id] = "skipped"
                return

            wf.state[f"while_{step.id}_iteration"] = iteration + 1
            task = await self._tasks.create(
                title=f"{step.name} [iter {iteration}]",
                description=f"While loop iteration {iteration}: {step.tool_name}",
                workflow_id=wf.id,
                max_retries=step.retry_max,
                metadata={
                    "step_id": step.id,
                    "tool_name": step.tool_name,
                    "tool_params": {**step.tool_params, "iteration": iteration},
                    "loop_iteration": iteration,
                },
            )
            wf.step_tasks[step.id] = task.id

    # ── Compensation (Saga rollback) ──────────────────────────────────────

    async def _run_compensation(self, wf: WorkflowRecord, task_map: dict[str, Any]) -> None:
        """Run compensation actions in reverse order for completed steps."""
        completed_steps_with_compensation = []
        for step in wf.steps:
            tid = wf.step_tasks.get(step.id)
            if not tid or tid == "skipped":
                continue
            task = task_map.get(tid)
            if task and task.status == TaskStatus.SUCCESS:
                compensation = step.tool_params.get("compensation")
                if compensation:
                    completed_steps_with_compensation.append((step, compensation))

        for step, compensation in reversed(completed_steps_with_compensation):
            try:
                comp_task = await self._tasks.create(
                    title=f"compensate_{step.name}",
                    description=f"Compensation for {step.id}: {compensation}",
                    workflow_id=wf.id,
                    metadata={
                        "step_id": step.id,
                        "is_compensation": True,
                        "compensation_config": compensation,
                    },
                )
                logger.info("Compensation task created for step {}: {}", step.id, comp_task.id)
            except Exception as exc:
                logger.error("Failed to create compensation for step {}: {}", step.id, exc)

    # ── Retry logic ────────────────────────────────────────────────────────

    async def _retry_failed_steps(
        self, wf: WorkflowRecord, failed_steps: list[str], task_map: dict[str, Any],
    ) -> bool:
        """Attempt to retry failed steps. Returns True if any were retried."""
        retried_any = False
        for step_id in failed_steps:
            step = next((s for s in wf.steps if s.id == step_id), None)
            if not step:
                continue
            tid = wf.step_tasks.get(step_id)
            if not tid:
                continue
            task = task_map.get(tid)
            if not task:
                continue
            if task.retry_count < task.max_retries:
                try:
                    await self._tasks.retry(tid)
                    retried_any = True
                    logger.info("Retrying step {} (attempt {})", step_id, task.retry_count + 1)
                except ValueError:
                    pass
        return retried_any
