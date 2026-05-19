"""Intelligent task delegation for multi-agent coordination.

Replaces naive keyword matching with capability-based scoring and optional
LLM-powered task analysis for intelligent agent selection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

LLMInvoker = Callable[[str, str], Awaitable[str]]


@dataclass
class AgentCapability:
    agent_id: str
    name: str = ""
    role: str = "worker"
    capabilities: list[str] = field(default_factory=list)
    task_types: list[str] = field(default_factory=list)
    max_concurrent_tasks: int = 3
    current_load: int = 0
    priority: int = 0
    performance_score: float = 1.0
    is_healthy: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def available_capacity(self) -> int:
        return max(0, self.max_concurrent_tasks - self.current_load)

    @property
    def is_available(self) -> bool:
        return self.is_healthy and self.available_capacity > 0


@dataclass
class DelegationTask:
    task_description: str
    required_capabilities: list[str] = field(default_factory=list)
    preferred_agent_id: str = ""
    priority: int = 5
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DelegationDecision:
    agent_id: str
    task_description: str
    score: float = 0.0
    reason: str = ""
    fallback_agent_ids: list[str] = field(default_factory=list)


class DelegationEngine:
    """Intelligent task delegation using capability matching and optional LLM analysis.

    Scoring algorithm:
    1. Capability match score (0-1): how many required capabilities the agent has
    2. Load score (0-1): preference for agents with more available capacity
    3. Performance score (0-1): historical success rate
    4. Priority bonus: agents with higher priority get a boost
    5. Preferred agent bonus: if explicitly preferred
    """

    def __init__(
        self,
        llm_invoker: LLMInvoker | None = None,
        capability_weight: float = 0.4,
        load_weight: float = 0.2,
        performance_weight: float = 0.3,
        priority_weight: float = 0.1,
    ):
        self._llm = llm_invoker
        self._weights = {
            "capability": capability_weight,
            "load": load_weight,
            "performance": performance_weight,
            "priority": priority_weight,
        }

    async def delegate(
        self,
        task: DelegationTask,
        agents: list[AgentCapability],
    ) -> DelegationDecision | None:
        """Select the best agent for a task using scoring algorithm."""
        available = [a for a in agents if a.is_available and a.role != "coordinator"]
        if not available:
            return None

        scored = []
        for agent in available:
            score = self._score_agent(agent, task)
            scored.append((agent, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        best_agent, best_score = scored[0]
        fallbacks = [a.agent_id for a, _ in scored[1:4]]

        return DelegationDecision(
            agent_id=best_agent.agent_id,
            task_description=task.task_description,
            score=best_score,
            reason=self._explain_score(best_agent, task, best_score),
            fallback_agent_ids=fallbacks,
        )

    async def delegate_multiple(
        self,
        tasks: list[DelegationTask],
        agents: list[AgentCapability],
    ) -> list[DelegationDecision]:
        """Assign multiple tasks to agents, balancing load across the team."""
        available = [a for a in agents if a.is_available and a.role != "coordinator"]
        if not available:
            return []

        decisions: list[DelegationDecision] = []
        load_tracker: dict[str, int] = {a.agent_id: a.current_load for a in available}

        sorted_tasks = sorted(tasks, key=lambda t: t.priority, reverse=True)

        for task in sorted_tasks:
            best_agent = None
            best_score = -1.0

            for agent in available:
                simulated_load = load_tracker.get(agent.agent_id, 0)
                if simulated_load >= agent.max_concurrent_tasks:
                    continue
                agent_copy = AgentCapability(
                    agent_id=agent.agent_id,
                    capabilities=agent.capabilities,
                    task_types=agent.task_types,
                    max_concurrent_tasks=agent.max_concurrent_tasks,
                    current_load=simulated_load,
                    priority=agent.priority,
                    performance_score=agent.performance_score,
                    is_healthy=agent.is_healthy,
                )
                score = self._score_agent(agent_copy, task)
                if score > best_score:
                    best_score = score
                    best_agent = agent

            if best_agent:
                load_tracker[best_agent.agent_id] = load_tracker.get(best_agent.agent_id, 0) + 1
                decisions.append(DelegationDecision(
                    agent_id=best_agent.agent_id,
                    task_description=task.task_description,
                    score=best_score,
                    reason=self._explain_score(best_agent, task, best_score),
                ))

        return decisions

    async def analyze_and_decompose(
        self,
        coordinator_response: str,
        agents: list[AgentCapability],
    ) -> list[DelegationTask]:
        """Use LLM to analyze coordinator response and extract structured tasks.

        Falls back to capability-based extraction if LLM is unavailable.
        """
        if self._llm:
            return await self._llm_decompose(coordinator_response, agents)
        return self._heuristic_decompose(coordinator_response, agents)

    def _score_agent(self, agent: AgentCapability, task: DelegationTask) -> float:
        """Compute a composite score for an agent-task pair."""
        cap_score = self._capability_score(agent, task)
        load_score = agent.available_capacity / max(1, agent.max_concurrent_tasks)
        perf_score = agent.performance_score
        priority_score = min(1.0, agent.priority / 10.0) if agent.priority > 0 else 0.5

        score = (
            cap_score * self._weights["capability"]
            + load_score * self._weights["load"]
            + perf_score * self._weights["performance"]
            + priority_score * self._weights["priority"]
        )

        if task.preferred_agent_id and task.preferred_agent_id == agent.agent_id:
            score += 0.2

        return min(1.0, score)

    def _capability_score(self, agent: AgentCapability, task: DelegationTask) -> float:
        """Score how well an agent's capabilities match the task requirements."""
        if not task.required_capabilities:
            if agent.task_types:
                return 0.7
            return 0.5

        agent_caps = set(c.lower() for c in agent.capabilities + agent.task_types)
        required = set(c.lower() for c in task.required_capabilities)

        if not required:
            return 0.5

        matched = required & agent_caps
        return len(matched) / len(required)

    def _explain_score(self, agent: AgentCapability, task: DelegationTask, score: float) -> str:
        parts = [f"score={score:.2f}"]
        if task.required_capabilities:
            agent_caps = set(c.lower() for c in agent.capabilities + agent.task_types)
            matched = set(c.lower() for c in task.required_capabilities) & agent_caps
            parts.append(f"capabilities_matched={len(matched)}/{len(task.required_capabilities)}")
        parts.append(f"load={agent.current_load}/{agent.max_concurrent_tasks}")
        parts.append(f"perf={agent.performance_score:.2f}")
        return ", ".join(parts)

    async def _llm_decompose(
        self,
        coordinator_response: str,
        agents: list[AgentCapability],
    ) -> list[DelegationTask]:
        """Use LLM to extract tasks from coordinator response."""
        agent_descriptions = "\n".join(
            f"- {a.agent_id}: capabilities={a.capabilities}, task_types={a.task_types}"
            for a in agents if a.role != "coordinator"
        )
        prompt = (
            "Analyze the following coordinator response and extract discrete tasks "
            "that should be delegated to worker agents. For each task, identify "
            "required capabilities.\n\n"
            f"Available agents:\n{agent_descriptions}\n\n"
            f"Coordinator response:\n{coordinator_response}\n\n"
            "Return tasks as a JSON array with objects containing: "
            '"task_description", "required_capabilities" (list of strings), "priority" (1-10).\n'
            "Return ONLY the JSON array, no other text."
        )
        try:
            result = await self._llm("system", prompt)
            return self._parse_llm_tasks(result)
        except Exception:
            return self._heuristic_decompose(coordinator_response, agents)

    def _heuristic_decompose(
        self,
        coordinator_response: str,
        agents: list[AgentCapability],
    ) -> list[DelegationTask]:
        """Extract tasks using heuristic matching against agent capabilities."""
        tasks: list[DelegationTask] = []
        all_capabilities: set[str] = set()
        for agent in agents:
            all_capabilities.update(c.lower() for c in agent.capabilities + agent.task_types)

        sentences = re.split(r'[.。!！?？\n]+', coordinator_response)
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence) < 10:
                continue
            matched_caps = [cap for cap in all_capabilities if cap.lower() in sentence.lower()]
            if matched_caps:
                tasks.append(DelegationTask(
                    task_description=sentence,
                    required_capabilities=matched_caps,
                    priority=5,
                ))

        if not tasks and coordinator_response.strip():
            tasks.append(DelegationTask(
                task_description=coordinator_response[:500],
                priority=5,
            ))

        return tasks

    @staticmethod
    def _parse_llm_tasks(llm_output: str) -> list[DelegationTask]:
        """Parse LLM output into DelegationTask objects."""
        import json
        try:
            start = llm_output.find("[")
            end = llm_output.rfind("]") + 1
            if start < 0 or end <= start:
                return []
            items = json.loads(llm_output[start:end])
            tasks = []
            for item in items:
                if isinstance(item, dict):
                    tasks.append(DelegationTask(
                        task_description=str(item.get("task_description", "")),
                        required_capabilities=list(item.get("required_capabilities", [])),
                        priority=int(item.get("priority", 5)),
                    ))
            return tasks
        except (json.JSONDecodeError, ValueError):
            return []
