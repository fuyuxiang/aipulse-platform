"""Coordinator agent — orchestrates multi-agent collaboration with intelligent
task decomposition, result aggregation, and quality checking.

The coordinator sits above worker agents and manages:
- Task analysis and decomposition into sub-tasks
- Intelligent assignment based on worker capabilities
- Result collection and quality validation
- Conflict detection and resolution
- Dynamic worker scaling
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from loguru import logger

from echo_agent.agent.multi_agent.models import WorkerProfile, WorkerResult


@dataclass
class SubTask:
    id: str = field(default_factory=lambda: f"st_{uuid.uuid4().hex[:8]}")
    description: str = ""
    required_capabilities: list[str] = field(default_factory=list)
    assigned_worker_id: str = ""
    priority: int = 5
    timeout_seconds: float = 120.0
    status: str = "pending"
    result: WorkerResult | None = None


@dataclass
class CoordinationPlan:
    tasks: list[SubTask] = field(default_factory=list)
    strategy: str = "parallel"
    max_rounds: int = 3
    require_consensus: bool = False
    conflict_resolution: str = "priority"


WorkerInvoker = Callable[[str, str, list[dict[str, Any]]], Awaitable[WorkerResult]]


class CoordinatorEngine:
    """Manages multi-worker coordination with intelligent task routing."""

    def __init__(
        self,
        workers: list[WorkerProfile],
        invoker: WorkerInvoker,
        *,
        max_rounds: int = 5,
        timeout_seconds: float = 300.0,
    ):
        self._workers = {w.id: w for w in workers}
        self._invoker = invoker
        self._max_rounds = max_rounds
        self._timeout = timeout_seconds
        self._health: dict[str, _WorkerHealth] = {
            w.id: _WorkerHealth(worker_id=w.id) for w in workers
        }

    async def execute_plan(
        self,
        plan: CoordinationPlan,
        tool_defs: list[dict[str, Any]],
    ) -> CoordinationResult:
        """Execute a coordination plan, dispatching tasks to workers."""
        started = time.monotonic()
        results: list[SubTask] = []

        if plan.strategy == "sequential":
            results = await self._execute_sequential(plan.tasks, tool_defs)
        elif plan.strategy == "parallel":
            results = await self._execute_parallel(plan.tasks, tool_defs)
        elif plan.strategy == "round_robin":
            results = await self._execute_round_robin(plan, tool_defs)
        else:
            results = await self._execute_parallel(plan.tasks, tool_defs)

        successful = [t for t in results if t.status == "completed"]
        failed = [t for t in results if t.status == "failed"]

        if plan.require_consensus and len(successful) > 1:
            final_output = self._resolve_consensus(successful, plan.conflict_resolution)
        elif successful:
            final_output = successful[-1].result.output if successful[-1].result else ""
        else:
            final_output = ""

        return CoordinationResult(
            tasks=results,
            final_output=final_output,
            total_duration=time.monotonic() - started,
            successful_count=len(successful),
            failed_count=len(failed),
        )

    def decompose_task(
        self,
        goal: str,
        context: str = "",
    ) -> CoordinationPlan:
        """Decompose a goal into sub-tasks based on available worker capabilities."""
        tasks: list[SubTask] = []
        available_workers = [w for w in self._workers.values() if self._is_healthy(w.id)]

        if not available_workers:
            tasks.append(SubTask(description=goal, priority=10))
            return CoordinationPlan(tasks=tasks, strategy="sequential")

        if len(available_workers) == 1:
            tasks.append(SubTask(
                description=goal,
                assigned_worker_id=available_workers[0].id,
                priority=10,
            ))
            return CoordinationPlan(tasks=tasks, strategy="sequential")

        for worker in available_workers:
            if worker.description and any(
                cap.lower() in goal.lower()
                for cap in (worker.description.split(",") + list(worker.default_tools))
            ):
                tasks.append(SubTask(
                    description=goal,
                    required_capabilities=list(worker.default_tools),
                    assigned_worker_id=worker.id,
                    priority=5,
                ))

        if not tasks:
            best_worker = self._select_best_worker(goal, available_workers)
            tasks.append(SubTask(
                description=goal,
                assigned_worker_id=best_worker.id,
                priority=5,
            ))

        strategy = "parallel" if len(tasks) > 1 else "sequential"
        return CoordinationPlan(tasks=tasks, strategy=strategy)

    async def _execute_sequential(
        self, tasks: list[SubTask], tool_defs: list[dict[str, Any]],
    ) -> list[SubTask]:
        """Execute tasks one by one, passing context forward."""
        context_accumulator = ""
        for task in tasks:
            worker_id = task.assigned_worker_id or self._select_worker_for_task(task)
            task.assigned_worker_id = worker_id
            task.status = "running"

            prompt = task.description
            if context_accumulator:
                prompt = f"{task.description}\n\nPrevious context:\n{context_accumulator}"

            try:
                result = await asyncio.wait_for(
                    self._invoker(worker_id, prompt, tool_defs),
                    timeout=task.timeout_seconds,
                )
                task.result = result
                task.status = result.status
                self._record_success(worker_id, result.duration_seconds)
                if result.output:
                    context_accumulator = result.output
            except asyncio.TimeoutError:
                task.status = "failed"
                task.result = WorkerResult(task_index=0, status="timeout", error="timeout")
                self._record_failure(worker_id)
            except Exception as exc:
                task.status = "failed"
                task.result = WorkerResult(task_index=0, status="failed", error=str(exc))
                self._record_failure(worker_id)

        return tasks

    async def _execute_parallel(
        self, tasks: list[SubTask], tool_defs: list[dict[str, Any]],
    ) -> list[SubTask]:
        """Execute all tasks in parallel."""
        async def _run_task(task: SubTask) -> SubTask:
            worker_id = task.assigned_worker_id or self._select_worker_for_task(task)
            task.assigned_worker_id = worker_id
            task.status = "running"
            try:
                result = await asyncio.wait_for(
                    self._invoker(worker_id, task.description, tool_defs),
                    timeout=task.timeout_seconds,
                )
                task.result = result
                task.status = result.status
                self._record_success(worker_id, result.duration_seconds)
            except asyncio.TimeoutError:
                task.status = "failed"
                task.result = WorkerResult(task_index=0, status="timeout", error="timeout")
                self._record_failure(worker_id)
            except Exception as exc:
                task.status = "failed"
                task.result = WorkerResult(task_index=0, status="failed", error=str(exc))
                self._record_failure(worker_id)
            return task

        await asyncio.gather(*[_run_task(t) for t in tasks])
        return tasks

    async def _execute_round_robin(
        self, plan: CoordinationPlan, tool_defs: list[dict[str, Any]],
    ) -> list[SubTask]:
        """Execute in rounds: each round all workers process, then results feed next round."""
        current_tasks = list(plan.tasks)
        for round_num in range(plan.max_rounds):
            current_tasks = await self._execute_parallel(current_tasks, tool_defs)
            completed = [t for t in current_tasks if t.status == "completed"]
            if completed:
                break
            for task in current_tasks:
                if task.status == "failed" and task.result:
                    task.status = "pending"
                    task.description += f"\n\nPrevious attempt failed: {task.result.error}"
        return current_tasks

    def _select_worker_for_task(self, task: SubTask) -> str:
        """Select the best available worker for a task."""
        available = [
            w for w in self._workers.values()
            if self._is_healthy(w.id)
        ]
        if not available:
            return list(self._workers.keys())[0] if self._workers else ""

        if task.required_capabilities:
            for worker in available:
                if any(cap in worker.default_tools for cap in task.required_capabilities):
                    return worker.id

        return self._select_best_worker(task.description, available).id

    def _select_best_worker(self, goal: str, workers: list[WorkerProfile]) -> WorkerProfile:
        """Select worker with best health score and lowest load."""
        scored = []
        for w in workers:
            health = self._health.get(w.id, _WorkerHealth(worker_id=w.id))
            score = health.success_rate * 0.6 + (1.0 / max(1, health.active_tasks + 1)) * 0.4
            scored.append((w, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0]

    def _resolve_consensus(self, tasks: list[SubTask], strategy: str) -> str:
        """Resolve multiple successful results into a single output."""
        outputs = [t.result.output for t in tasks if t.result and t.result.output]
        if not outputs:
            return ""
        if strategy == "longest":
            return max(outputs, key=len)
        if strategy == "latest":
            return outputs[-1]
        if strategy == "merge":
            return "\n\n".join(outputs)
        return outputs[0]

    def _is_healthy(self, worker_id: str) -> bool:
        health = self._health.get(worker_id)
        if not health:
            return True
        return health.consecutive_failures < 3

    def _record_success(self, worker_id: str, duration: float) -> None:
        health = self._health.setdefault(worker_id, _WorkerHealth(worker_id=worker_id))
        health.total_success += 1
        health.consecutive_failures = 0
        health.last_response_time = duration

    def _record_failure(self, worker_id: str) -> None:
        health = self._health.setdefault(worker_id, _WorkerHealth(worker_id=worker_id))
        health.total_failures += 1
        health.consecutive_failures += 1


@dataclass
class _WorkerHealth:
    worker_id: str = ""
    total_success: int = 0
    total_failures: int = 0
    consecutive_failures: int = 0
    active_tasks: int = 0
    last_response_time: float = 0.0

    @property
    def success_rate(self) -> float:
        total = self.total_success + self.total_failures
        if total == 0:
            return 1.0
        return self.total_success / total


@dataclass
class CoordinationResult:
    tasks: list[SubTask] = field(default_factory=list)
    final_output: str = ""
    total_duration: float = 0.0
    successful_count: int = 0
    failed_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_output": self.final_output,
            "total_duration": self.total_duration,
            "successful_count": self.successful_count,
            "failed_count": self.failed_count,
            "tasks": [
                {
                    "id": t.id,
                    "description": t.description[:200],
                    "assigned_worker_id": t.assigned_worker_id,
                    "status": t.status,
                    "output": t.result.output[:500] if t.result else "",
                    "error": t.result.error if t.result else "",
                }
                for t in self.tasks
            ],
        }
