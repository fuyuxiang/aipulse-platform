"""Workflow checkpoint — persistence and recovery for workflow execution state.

Provides checkpoint creation, storage, and resume capabilities so workflows
can recover from failures or continue after approval/event waits.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Checkpoint:
    id: str = field(default_factory=lambda: f"cp_{uuid.uuid4().hex[:12]}")
    workflow_run_id: str = ""
    created_at: float = field(default_factory=time.time)
    completed_nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    compensation_stack: list[dict[str, Any]] = field(default_factory=list)
    pending_node_id: str = ""
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "workflow_run_id": self.workflow_run_id,
            "created_at": self.created_at,
            "completed_nodes": self.completed_nodes,
            "context": self.context,
            "compensation_stack": self.compensation_stack,
            "pending_node_id": self.pending_node_id,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Checkpoint:
        return cls(
            id=data.get("id", ""),
            workflow_run_id=data.get("workflow_run_id", ""),
            created_at=data.get("created_at", 0.0),
            completed_nodes=data.get("completed_nodes", {}),
            context=data.get("context", {}),
            compensation_stack=data.get("compensation_stack", []),
            pending_node_id=data.get("pending_node_id", ""),
            reason=data.get("reason", ""),
        )


class CheckpointStore:
    """File-based checkpoint storage for workflow runs."""

    def __init__(self, base_dir: str | Path):
        self._base = Path(base_dir) / "workflow_checkpoints"
        self._base.mkdir(parents=True, exist_ok=True)

    def _run_dir(self, run_id: str) -> Path:
        d = self._base / run_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    async def save(self, checkpoint: Checkpoint) -> str:
        path = self._run_dir(checkpoint.workflow_run_id) / f"{checkpoint.id}.json"
        path.write_text(json.dumps(checkpoint.to_dict(), ensure_ascii=False, default=str), encoding="utf-8")
        return checkpoint.id

    async def load(self, run_id: str, checkpoint_id: str) -> Checkpoint | None:
        path = self._run_dir(run_id) / f"{checkpoint_id}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return Checkpoint.from_dict(data)

    async def load_latest(self, run_id: str) -> Checkpoint | None:
        run_dir = self._run_dir(run_id)
        files = sorted(run_dir.glob("cp_*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
        if not files:
            return None
        data = json.loads(files[0].read_text(encoding="utf-8"))
        return Checkpoint.from_dict(data)

    async def list_checkpoints(self, run_id: str) -> list[Checkpoint]:
        run_dir = self._run_dir(run_id)
        results = []
        for f in sorted(run_dir.glob("cp_*.json"), key=lambda p: p.stat().st_mtime):
            data = json.loads(f.read_text(encoding="utf-8"))
            results.append(Checkpoint.from_dict(data))
        return results

    async def delete(self, run_id: str, checkpoint_id: str) -> bool:
        path = self._run_dir(run_id) / f"{checkpoint_id}.json"
        if path.exists():
            path.unlink()
            return True
        return False

    async def delete_all(self, run_id: str) -> int:
        run_dir = self._run_dir(run_id)
        count = 0
        for f in run_dir.glob("cp_*.json"):
            f.unlink()
            count += 1
        return count


def build_checkpoint(
    run_id: str,
    results: dict[str, Any],
    context: dict[str, Any],
    compensation_stack: list[dict[str, Any]],
    pending_node_id: str = "",
    reason: str = "",
) -> Checkpoint:
    """Create a checkpoint from current execution state."""
    completed = {}
    for node_id, result in results.items():
        if isinstance(result, dict):
            completed[node_id] = result
        else:
            completed[node_id] = {"status": getattr(result, "status", "success"), "output": getattr(result, "output", result)}
    return Checkpoint(
        workflow_run_id=run_id,
        completed_nodes=completed,
        context=context,
        compensation_stack=compensation_stack,
        pending_node_id=pending_node_id,
        reason=reason,
    )
