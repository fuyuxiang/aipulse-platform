from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.services.coordination.conflict import AgentResponse, ConflictResolver, ResolutionStrategy
from app.services.coordination.consensus import ConsensusEngine, Vote, VoteChoice, VotingStrategy
from app.services.coordination.delegation import AgentCapability, DelegationEngine, DelegationTask
from app.services.coordination.health import HealthMonitor
from app.services.coordination.protocol import CoordinationMessage, MessageBroker, MessageType
from app.services.resource_service import ResourceService


class MultiAgentService:
    def __init__(self, db: Session):
        self.db = db
        self.resources = ResourceService(db)
        self._broker = MessageBroker()
        self._health = HealthMonitor(heartbeat_interval=30.0, unhealthy_threshold=3)
        self._consensus = ConsensusEngine()
        self._delegation = DelegationEngine()
        self._conflict = ConflictResolver()

    def create_team(self, tenant_id: str, user_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        name = str(payload.get("name", ""))
        topology = str(payload.get("topology", "star"))
        coordinator_agent_id = str(payload.get("coordinator_agent_id", ""))
        team = self.resources.create("agent_teams", tenant_id, user_id, {
            "name": name,
            "code": f"team-{uuid.uuid4().hex[:8]}",
            "status": "active",
            "spec": {
                "topology": topology,
                "coordinator_agent_id": coordinator_agent_id,
                "delegation_strategy": str(payload.get("delegation_strategy", "auto")),
                "consensus_strategy": str(payload.get("consensus_strategy", "majority")),
                "conflict_strategy": str(payload.get("conflict_strategy", "priority")),
                "max_rounds": int(payload.get("max_rounds", 10)),
                "timeout_seconds": int(payload.get("timeout_seconds", 300)),
                "communication_protocol": str(payload.get("communication_protocol", "a2a")),
                "shared_memory_enabled": bool(payload.get("shared_memory_enabled", True)),
                "shared_knowledge_base_ids": payload.get("shared_knowledge_base_ids", []),
                "fallback_agent_id": str(payload.get("fallback_agent_id", "")),
                "enable_failover": bool(payload.get("enable_failover", True)),
                "enable_consensus": bool(payload.get("enable_consensus", False)),
                "description": str(payload.get("description", "")),
            },
        })
        members = payload.get("members", [])
        for member in members:
            self._add_member(tenant_id, user_id, team.id, member)
        return ResourceService.to_dict(team)

    def update_team(self, tenant_id: str, user_id: str, team_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        row = self.resources.update("agent_teams", tenant_id, user_id, team_id, payload)
        return ResourceService.to_dict(row)

    def delete_team(self, tenant_id: str, user_id: str, team_id: str) -> dict[str, str]:
        return self.resources.delete("agent_teams", tenant_id, user_id, team_id)

    def get_team(self, tenant_id: str, team_id: str) -> dict[str, Any]:
        team = ResourceService.to_dict(self.resources.get("agent_teams", tenant_id, team_id))
        members, _ = self.resources.list("agent_team_members", tenant_id, 1, 100, {"parent_id": team_id})
        team["members"] = [ResourceService.to_dict(m) for m in members]
        return team

    def list_teams(self, tenant_id: str, page: int, page_size: int) -> tuple[list[dict[str, Any]], int]:
        rows, total = self.resources.list("agent_teams", tenant_id, page, page_size)
        return [ResourceService.to_dict(row) for row in rows], total

    def add_member(self, tenant_id: str, user_id: str, team_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return ResourceService.to_dict(self._add_member(tenant_id, user_id, team_id, payload))

    def _add_member(self, tenant_id: str, user_id: str, team_id: str, payload: dict[str, Any]) -> Any:
        agent_id = str(payload.get("agent_id", ""))
        role = str(payload.get("role", "worker"))
        return self.resources.create("agent_team_members", tenant_id, user_id, {
            "name": str(payload.get("name", f"member-{agent_id[:8]}")),
            "code": f"member-{uuid.uuid4().hex[:8]}",
            "status": "active",
            "parent_id": team_id,
            "agent_id": agent_id,
            "spec": {
                "role": role,
                "capabilities": payload.get("capabilities", []),
                "task_types": payload.get("task_types", []),
                "priority": int(payload.get("priority", 0)),
                "max_concurrent_tasks": int(payload.get("max_concurrent_tasks", 3)),
                "delegation_rules": payload.get("delegation_rules", []),
                "can_delegate_to": payload.get("can_delegate_to", []),
                "description": str(payload.get("description", "")),
            },
        })

    def remove_member(self, tenant_id: str, user_id: str, team_id: str, member_id: str) -> dict[str, str]:
        return self.resources.delete("agent_team_members", tenant_id, user_id, member_id)

    def update_member(self, tenant_id: str, user_id: str, member_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        row = self.resources.update("agent_team_members", tenant_id, user_id, member_id, payload)
        return ResourceService.to_dict(row)

    async def run_team(self, tenant_id: str, user_id: str, team_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        team = self.resources.get("agent_teams", tenant_id, team_id)
        spec = dict(team.spec or {})
        prompt = str(payload.get("prompt", ""))
        session_id = str(payload.get("session_id", f"team-run-{uuid.uuid4().hex[:8]}"))

        members, _ = self.resources.list("agent_team_members", tenant_id, 1, 100, {"parent_id": team_id})
        topology = spec.get("topology", "star")
        coordinator_id = spec.get("coordinator_agent_id", "")
        max_rounds = spec.get("max_rounds", 10)
        timeout_seconds = float(spec.get("timeout_seconds", 300))

        run = self.resources.create("agent_team_runs", tenant_id, user_id, {
            "name": f"team-run-{team.name}",
            "code": f"trun-{uuid.uuid4().hex[:8]}",
            "status": "running",
            "parent_id": team_id,
            "session_id": session_id,
            "spec": {
                "prompt": prompt,
                "topology": topology,
                "coordinator_agent_id": coordinator_id,
                "max_rounds": max_rounds,
                "member_count": len(members),
                "started_at": datetime.now(timezone.utc).isoformat(),
            },
        })

        agent_caps = self._build_agent_capabilities(members)
        for cap in agent_caps:
            self._health.register_agent(cap.agent_id, max_capacity=cap.max_concurrent_tasks)

        messages: list[dict[str, Any]] = []
        final_response = ""

        try:
            if topology == "pipeline":
                final_response = await self._run_pipeline(
                    tenant_id, user_id, team_id, run.id,
                    members, prompt, session_id, messages, timeout_seconds,
                )
            elif topology == "mesh":
                final_response = await self._run_mesh(
                    tenant_id, user_id, team_id, run.id,
                    members, agent_caps, prompt, session_id,
                    max_rounds, messages, spec, timeout_seconds,
                )
            else:
                final_response = await self._run_star(
                    tenant_id, user_id, team_id, run.id,
                    coordinator_id, members, agent_caps, prompt, session_id,
                    max_rounds, messages, spec, timeout_seconds,
                )
        except Exception as exc:
            run_spec = dict(run.spec or {})
            run_spec["finished_at"] = datetime.now(timezone.utc).isoformat()
            run_spec["message_count"] = len(messages)
            self.resources.update("agent_team_runs", tenant_id, user_id, run.id, {"status": "failed", "spec": run_spec, "error_message": str(exc)})
            raise

        run_spec = dict(run.spec or {})
        run_spec["finished_at"] = datetime.now(timezone.utc).isoformat()
        run_spec["final_response"] = final_response
        run_spec["message_count"] = len(messages)
        self.resources.update("agent_team_runs", tenant_id, user_id, run.id, {"status": "completed", "spec": run_spec})

        return {
            "run_id": run.id,
            "team_id": team_id,
            "status": "completed",
            "response": final_response,
            "messages": messages,
        }

    async def _run_star(
        self, tenant_id: str, user_id: str, team_id: str, run_id: str,
        coordinator_id: str, members: list[Any], agent_caps: list[AgentCapability],
        prompt: str, session_id: str, max_rounds: int,
        messages: list[dict[str, Any]], spec: dict[str, Any], timeout: float,
    ) -> str:
        """Star topology: coordinator delegates tasks to workers intelligently."""
        worker_caps = [c for c in agent_caps if c.role != "coordinator"]
        current_prompt = prompt
        fallback_agent_id = spec.get("fallback_agent_id", "")

        for round_num in range(max_rounds):
            coordinator_response = await self._invoke_agent_safe(
                tenant_id, user_id, coordinator_id, current_prompt, session_id, fallback_agent_id
            )
            self._record_message(tenant_id, user_id, run_id, coordinator_id, "coordinator", coordinator_response, round_num)
            messages.append({"round": round_num, "agent_id": coordinator_id, "role": "coordinator", "content": coordinator_response})
            self._health.record_heartbeat(coordinator_id)

            tasks = await self._delegation.analyze_and_decompose(coordinator_response, worker_caps)
            if not tasks:
                return coordinator_response

            decisions = await self._delegation.delegate_multiple(tasks, worker_caps)
            if not decisions:
                return coordinator_response

            worker_results: list[AgentResponse] = []
            for decision in decisions:
                agent_id = decision.agent_id
                task_desc = decision.task_description
                self._health.assign_task(agent_id, {"id": f"t_{round_num}_{agent_id}", "description": task_desc})

                started = time.time()
                try:
                    worker_response = await self._invoke_agent_safe(
                        tenant_id, user_id, agent_id, task_desc, session_id, fallback_agent_id
                    )
                    elapsed_ms = (time.time() - started) * 1000
                    self._health.record_task_success(agent_id, elapsed_ms)
                except Exception as exc:
                    self._health.record_task_failure(agent_id, str(exc))
                    if decision.fallback_agent_ids:
                        worker_response = await self._invoke_agent_safe(
                            tenant_id, user_id, decision.fallback_agent_ids[0], task_desc, session_id, ""
                        )
                    else:
                        worker_response = f"[Error from {agent_id}: {exc}]"
                finally:
                    self._health.complete_task(agent_id)

                self._record_message(tenant_id, user_id, run_id, agent_id, "worker", worker_response, round_num)
                messages.append({"round": round_num, "agent_id": agent_id, "role": "worker", "task": task_desc, "content": worker_response})
                worker_results.append(AgentResponse(
                    agent_id=agent_id, content=worker_response,
                    priority=int(decision.score * 10), timestamp=time.time(),
                ))

            if len(worker_results) > 1 and spec.get("enable_consensus"):
                conflict_report = await self._conflict.detect_conflict(worker_results)
                if conflict_report.severity.value in ("medium", "high"):
                    resolution = await self._conflict.resolve(
                        worker_results,
                        strategy=ResolutionStrategy(spec.get("conflict_strategy", "priority")),
                        conflict_report=conflict_report,
                    )
                    current_prompt = (
                        f"Workers completed. Resolved result:\n{resolution.resolved_content}\n\n"
                        f"Original request: {prompt}"
                    )
                    continue

            results_summary = "\n".join(f"Worker {r.agent_id}: {r.content}" for r in worker_results)
            current_prompt = (
                f"Workers completed:\n{results_summary}\n\nOriginal request: {prompt}\n"
                f"Synthesize the results into a final response."
            )

        final = await self._invoke_agent_safe(tenant_id, user_id, coordinator_id, current_prompt, session_id, fallback_agent_id)
        return final

    async def _run_pipeline(
        self, tenant_id: str, user_id: str, team_id: str, run_id: str,
        members: list[Any], prompt: str, session_id: str,
        messages: list[dict[str, Any]], timeout: float,
    ) -> str:
        """Pipeline topology: sequential processing with failover."""
        sorted_members = sorted(members, key=lambda m: (m.spec or {}).get("priority", 0))
        current_input = prompt

        for i, member in enumerate(sorted_members):
            agent_id = member.agent_id
            self._health.record_heartbeat(agent_id)
            self._health.assign_task(agent_id, {"id": f"pipe_{i}", "description": current_input[:100]})

            started = time.time()
            try:
                response = await self._invoke_agent_safe(
                    tenant_id, user_id, agent_id, current_input, session_id,
                    self._find_fallback_in_members(members, agent_id),
                )
                self._health.record_task_success(agent_id, (time.time() - started) * 1000)
            except Exception as exc:
                self._health.record_task_failure(agent_id, str(exc))
                fallback_id = self._find_fallback_in_members(members, agent_id)
                if fallback_id:
                    response = await self._invoke_agent_safe(tenant_id, user_id, fallback_id, current_input, session_id, "")
                else:
                    raise
            finally:
                self._health.complete_task(agent_id)

            self._record_message(tenant_id, user_id, run_id, agent_id, "pipeline", response, i)
            messages.append({"step": i, "agent_id": agent_id, "content": response})
            current_input = response

        return current_input

    async def _run_mesh(
        self, tenant_id: str, user_id: str, team_id: str, run_id: str,
        members: list[Any], agent_caps: list[AgentCapability],
        prompt: str, session_id: str, max_rounds: int,
        messages: list[dict[str, Any]], spec: dict[str, Any], timeout: float,
    ) -> str:
        """Mesh topology: parallel execution with consensus and conflict resolution."""
        conflict_strategy = ResolutionStrategy(spec.get("conflict_strategy", "priority"))

        for round_num in range(min(max_rounds, 3)):
            round_prompt = prompt if round_num == 0 else f"{prompt}\n\nPrevious round results available."

            tasks = [
                self._invoke_agent_safe(tenant_id, user_id, m.agent_id, round_prompt, session_id, "")
                for m in members
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            round_responses: list[AgentResponse] = []
            for member, result in zip(members, results):
                if isinstance(result, Exception):
                    self._health.record_task_failure(member.agent_id, str(result))
                    content = f"[Error: {result}]"
                else:
                    self._health.record_task_success(member.agent_id)
                    content = str(result)

                self._record_message(tenant_id, user_id, run_id, member.agent_id, "mesh", content, round_num)
                messages.append({"round": round_num, "agent_id": member.agent_id, "content": content})
                member_spec = member.spec or {}
                round_responses.append(AgentResponse(
                    agent_id=member.agent_id, content=content,
                    priority=int(member_spec.get("priority", 0)), timestamp=time.time(),
                ))

            valid_responses = [r for r in round_responses if not r.content.startswith("[Error")]
            if not valid_responses:
                continue

            conflict_report = await self._conflict.detect_conflict(valid_responses)
            if conflict_report.severity.value in ("none", "low"):
                resolution = await self._conflict.resolve(valid_responses, strategy=ResolutionStrategy.MERGE)
                return resolution.resolved_content

            resolution = await self._conflict.resolve(valid_responses, strategy=conflict_strategy, conflict_report=conflict_report)
            return resolution.resolved_content

        if messages:
            return messages[-1].get("content", "")
        return ""

    async def _invoke_agent_safe(self, tenant_id: str, user_id: str, agent_id: str, prompt: str, session_id: str, fallback_id: str) -> str:
        """Invoke an agent with automatic failover to fallback on failure."""
        from app.services.agent_runner_service import AgentRunnerService

        try:
            result = await AgentRunnerService(self.db).run(tenant_id, user_id, agent_id, {
                "prompt": prompt, "session_id": session_id,
                "memory_policy": {"write_scope": "team", "shared": True, "include_shared": True},
            })
            return str(result.get("response", ""))
        except Exception as exc:
            if fallback_id and fallback_id != agent_id:
                result = await AgentRunnerService(self.db).run(tenant_id, user_id, fallback_id, {
                    "prompt": prompt, "session_id": session_id,
                    "memory_policy": {"write_scope": "team", "shared": True, "include_shared": True},
                })
                return str(result.get("response", ""))
            raise

    def _record_message(self, tenant_id: str, user_id: str, run_id: str, agent_id: str, role: str, content: str, step: int) -> None:
        self.resources.create("agent_team_messages", tenant_id, user_id, {
            "name": f"msg-{role}-{step}",
            "code": f"tmsg-{uuid.uuid4().hex[:8]}",
            "status": "sent",
            "parent_id": run_id,
            "agent_id": agent_id,
            "spec": {"role": role, "content": content, "step": step},
        })

    def _build_agent_capabilities(self, members: list[Any]) -> list[AgentCapability]:
        """Convert team members to AgentCapability objects for delegation engine."""
        caps = []
        for member in members:
            spec = member.spec or {}
            caps.append(AgentCapability(
                agent_id=member.agent_id,
                name=member.name,
                role=str(spec.get("role", "worker")),
                capabilities=list(spec.get("capabilities", [])),
                task_types=list(spec.get("task_types", [])),
                max_concurrent_tasks=int(spec.get("max_concurrent_tasks", 3)),
                priority=int(spec.get("priority", 0)),
            ))
        return caps

    def _find_fallback_in_members(self, members: list[Any], exclude_agent_id: str) -> str:
        """Find a fallback agent from team members."""
        for member in members:
            if member.agent_id != exclude_agent_id:
                spec = member.spec or {}
                if spec.get("role") == "worker":
                    return member.agent_id
        return ""

    # ── Dynamic orchestration ──────────────────────────────────────────────

    def add_runtime_member(self, tenant_id: str, user_id: str, team_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Dynamically add a member to a running team."""
        member = self._add_member(tenant_id, user_id, team_id, payload)
        spec = member.spec or {}
        self._health.register_agent(
            member.agent_id,
            max_capacity=int(spec.get("max_concurrent_tasks", 3)),
        )
        self._broker.send(CoordinationMessage(
            sender_agent_id="system",
            message_type=MessageType.CAPABILITY_ANNOUNCE,
            payload={"agent_id": member.agent_id, "capabilities": spec.get("capabilities", []), "action": "joined"},
        ))
        return ResourceService.to_dict(member)

    def remove_runtime_member(self, tenant_id: str, user_id: str, team_id: str, member_id: str) -> dict[str, str]:
        """Dynamically remove a member from a running team."""
        member = self.resources.get("agent_team_members", tenant_id, member_id)
        self._health.unregister_agent(member.agent_id)
        return self.resources.delete("agent_team_members", tenant_id, user_id, member_id)

    async def get_team_health(self, tenant_id: str, team_id: str) -> dict[str, Any]:
        """Get health status of all agents in a team."""
        await self._health.check_health()
        return self._health.get_team_status()

    def get_team_run(self, tenant_id: str, run_id: str) -> dict[str, Any]:
        run = ResourceService.to_dict(self.resources.get("agent_team_runs", tenant_id, run_id))
        msgs, _ = self.resources.list("agent_team_messages", tenant_id, 1, 200, {"parent_id": run_id})
        run["messages"] = [ResourceService.to_dict(m) for m in msgs]
        return run

    def list_team_runs(self, tenant_id: str, team_id: str, page: int, page_size: int) -> tuple[list[dict[str, Any]], int]:
        rows, total = self.resources.list("agent_team_runs", tenant_id, page, page_size, {"parent_id": team_id})
        return [ResourceService.to_dict(row) for row in rows], total
