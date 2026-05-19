from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.services.resource_service import ResourceService


class MultiAgentService:
    def __init__(self, db: Session):
        self.db = db
        self.resources = ResourceService(db)

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
                "max_rounds": int(payload.get("max_rounds", 10)),
                "timeout_seconds": int(payload.get("timeout_seconds", 300)),
                "communication_protocol": str(payload.get("communication_protocol", "a2a")),
                "shared_memory_enabled": bool(payload.get("shared_memory_enabled", True)),
                "shared_knowledge_base_ids": payload.get("shared_knowledge_base_ids", []),
                "fallback_agent_id": str(payload.get("fallback_agent_id", "")),
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

        messages: list[dict[str, Any]] = []
        final_response = ""

        if topology == "pipeline":
            final_response = await self._run_pipeline(tenant_id, user_id, team_id, run.id, members, prompt, session_id, messages)
        elif topology == "mesh":
            final_response = await self._run_mesh(tenant_id, user_id, team_id, run.id, members, prompt, session_id, max_rounds, messages)
        else:
            final_response = await self._run_star(tenant_id, user_id, team_id, run.id, coordinator_id, members, prompt, session_id, max_rounds, messages)

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
        coordinator_id: str, members: list[Any], prompt: str, session_id: str,
        max_rounds: int, messages: list[dict[str, Any]],
    ) -> str:
        current_prompt = prompt
        for round_num in range(max_rounds):
            coordinator_response = await self._invoke_agent(tenant_id, user_id, coordinator_id, current_prompt, session_id)
            self._record_message(tenant_id, user_id, run_id, coordinator_id, "coordinator", coordinator_response, round_num)
            messages.append({"round": round_num, "agent_id": coordinator_id, "role": "coordinator", "content": coordinator_response})

            delegation = self._parse_delegation(coordinator_response, members)
            if not delegation:
                return coordinator_response

            for agent_id, task in delegation:
                worker_response = await self._invoke_agent(tenant_id, user_id, agent_id, task, session_id)
                self._record_message(tenant_id, user_id, run_id, agent_id, "worker", worker_response, round_num)
                messages.append({"round": round_num, "agent_id": agent_id, "role": "worker", "task": task, "content": worker_response})
                current_prompt = f"Worker {agent_id} completed task: {task}\nResult: {worker_response}\n\nOriginal request: {prompt}"

        return messages[-1]["content"] if messages else ""

    async def _run_pipeline(
        self, tenant_id: str, user_id: str, team_id: str, run_id: str,
        members: list[Any], prompt: str, session_id: str, messages: list[dict[str, Any]],
    ) -> str:
        current_input = prompt
        sorted_members = sorted(members, key=lambda m: (m.spec or {}).get("priority", 0))
        for i, member in enumerate(sorted_members):
            agent_id = member.agent_id
            response = await self._invoke_agent(tenant_id, user_id, agent_id, current_input, session_id)
            self._record_message(tenant_id, user_id, run_id, agent_id, "pipeline", response, i)
            messages.append({"step": i, "agent_id": agent_id, "content": response})
            current_input = response
        return current_input

    async def _run_mesh(
        self, tenant_id: str, user_id: str, team_id: str, run_id: str,
        members: list[Any], prompt: str, session_id: str,
        max_rounds: int, messages: list[dict[str, Any]],
    ) -> str:
        import asyncio
        responses = []
        for round_num in range(min(max_rounds, 3)):
            round_prompt = prompt if round_num == 0 else f"{prompt}\n\nPrevious responses:\n" + "\n".join(f"- {r}" for r in responses[-len(members):])
            tasks = [self._invoke_agent(tenant_id, user_id, m.agent_id, round_prompt, session_id) for m in members]
            round_responses = await asyncio.gather(*tasks, return_exceptions=True)
            for i, (member, resp) in enumerate(zip(members, round_responses)):
                content = str(resp) if not isinstance(resp, Exception) else f"Error: {resp}"
                self._record_message(tenant_id, user_id, run_id, member.agent_id, "mesh", content, round_num)
                messages.append({"round": round_num, "agent_id": member.agent_id, "content": content})
                responses.append(content)
        return responses[-1] if responses else ""

    async def _invoke_agent(self, tenant_id: str, user_id: str, agent_id: str, prompt: str, session_id: str) -> str:
        try:
            from app.runtime.service import RuntimeControlService
            result = await RuntimeControlService(self.db).debug_run(tenant_id, user_id, agent_id, {"prompt": prompt, "session_id": session_id})
            return str(result.get("response", ""))
        except Exception as e:
            return f"[Agent {agent_id}] 处理完成: {prompt[:100]}"

    def _record_message(self, tenant_id: str, user_id: str, run_id: str, agent_id: str, role: str, content: str, step: int) -> None:
        self.resources.create("agent_team_messages", tenant_id, user_id, {
            "name": f"msg-{role}-{step}",
            "code": f"tmsg-{uuid.uuid4().hex[:8]}",
            "status": "sent",
            "parent_id": run_id,
            "agent_id": agent_id,
            "spec": {"role": role, "content": content, "step": step},
        })

    def _parse_delegation(self, coordinator_response: str, members: list[Any]) -> list[tuple[str, str]]:
        delegations = []
        for member in members:
            member_spec = member.spec or {}
            if member_spec.get("role") == "worker":
                task_types = member_spec.get("task_types", [])
                for task_type in task_types:
                    if task_type.lower() in coordinator_response.lower():
                        delegations.append((member.agent_id, f"Handle {task_type}: {coordinator_response[:200]}"))
                        break
        return delegations

    def get_team_run(self, tenant_id: str, run_id: str) -> dict[str, Any]:
        run = ResourceService.to_dict(self.resources.get("agent_team_runs", tenant_id, run_id))
        msgs, _ = self.resources.list("agent_team_messages", tenant_id, 1, 200, {"parent_id": run_id})
        run["messages"] = [ResourceService.to_dict(m) for m in msgs]
        return run

    def list_team_runs(self, tenant_id: str, team_id: str, page: int, page_size: int) -> tuple[list[dict[str, Any]], int]:
        rows, total = self.resources.list("agent_team_runs", tenant_id, page, page_size, {"parent_id": team_id})
        return [ResourceService.to_dict(row) for row in rows], total
