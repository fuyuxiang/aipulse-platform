"""Consensus voting mechanism for multi-agent decision making.

Supports multiple voting strategies: majority, unanimous, weighted, and quorum.
Handles timeout for non-responsive agents and provides structured vote results.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable


class VotingStrategy(str, Enum):
    MAJORITY = "majority"
    UNANIMOUS = "unanimous"
    WEIGHTED = "weighted"
    QUORUM = "quorum"
    FIRST_RESPONSE = "first_response"


class VoteChoice(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"


@dataclass
class Vote:
    agent_id: str
    choice: VoteChoice
    weight: float = 1.0
    reasoning: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VotingSession:
    id: str = field(default_factory=lambda: f"vote_{uuid.uuid4().hex[:12]}")
    topic: str = ""
    description: str = ""
    strategy: VotingStrategy = VotingStrategy.MAJORITY
    eligible_voters: list[str] = field(default_factory=list)
    votes: list[Vote] = field(default_factory=list)
    quorum_threshold: float = 0.5
    weight_map: dict[str, float] = field(default_factory=dict)
    timeout_seconds: float = 60.0
    created_at: float = field(default_factory=time.time)
    closed_at: float = 0.0
    result: str = ""
    decided: bool = False

    @property
    def participation_rate(self) -> float:
        if not self.eligible_voters:
            return 0.0
        non_abstain = [v for v in self.votes if v.choice != VoteChoice.ABSTAIN]
        return len(non_abstain) / len(self.eligible_voters)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "topic": self.topic,
            "description": self.description,
            "strategy": self.strategy.value,
            "eligible_voters": self.eligible_voters,
            "votes": [{"agent_id": v.agent_id, "choice": v.choice.value, "weight": v.weight, "reasoning": v.reasoning} for v in self.votes],
            "quorum_threshold": self.quorum_threshold,
            "timeout_seconds": self.timeout_seconds,
            "created_at": self.created_at,
            "closed_at": self.closed_at,
            "result": self.result,
            "decided": self.decided,
            "participation_rate": self.participation_rate,
        }


VoteCollector = Callable[[str, str, list[str]], Awaitable[list[Vote]]]


class ConsensusEngine:
    """Manages voting sessions and computes consensus results.

    Usage:
        engine = ConsensusEngine()
        session = engine.create_session(
            topic="Should we use approach A or B?",
            voters=["agent_1", "agent_2", "agent_3"],
            strategy=VotingStrategy.MAJORITY,
        )
        # Collect votes (via message broker or direct invocation)
        engine.cast_vote(session.id, Vote(agent_id="agent_1", choice=VoteChoice.APPROVE))
        engine.cast_vote(session.id, Vote(agent_id="agent_2", choice=VoteChoice.APPROVE))
        result = engine.tally(session.id)
    """

    def __init__(self):
        self._sessions: dict[str, VotingSession] = {}

    def create_session(
        self,
        topic: str,
        voters: list[str],
        strategy: VotingStrategy = VotingStrategy.MAJORITY,
        description: str = "",
        quorum_threshold: float = 0.5,
        weight_map: dict[str, float] | None = None,
        timeout_seconds: float = 60.0,
    ) -> VotingSession:
        """Create a new voting session."""
        session = VotingSession(
            topic=topic,
            description=description,
            strategy=strategy,
            eligible_voters=voters,
            quorum_threshold=quorum_threshold,
            weight_map=weight_map or {},
            timeout_seconds=timeout_seconds,
        )
        self._sessions[session.id] = session
        return session

    def cast_vote(self, session_id: str, vote: Vote) -> bool:
        """Cast a vote in a session. Returns False if session is closed or voter ineligible."""
        session = self._sessions.get(session_id)
        if not session or session.decided:
            return False
        if vote.agent_id not in session.eligible_voters:
            return False
        if any(v.agent_id == vote.agent_id for v in session.votes):
            return False

        if session.weight_map and vote.agent_id in session.weight_map:
            vote.weight = session.weight_map[vote.agent_id]

        session.votes.append(vote)

        if session.strategy == VotingStrategy.FIRST_RESPONSE:
            self._close_session(session, vote.choice.value)

        return True

    def tally(self, session_id: str) -> dict[str, Any]:
        """Compute the voting result for a session."""
        session = self._sessions.get(session_id)
        if not session:
            return {"error": "session not found"}

        if session.decided:
            return {"result": session.result, "decided": True, "session": session.to_dict()}

        strategy_fn = {
            VotingStrategy.MAJORITY: self._tally_majority,
            VotingStrategy.UNANIMOUS: self._tally_unanimous,
            VotingStrategy.WEIGHTED: self._tally_weighted,
            VotingStrategy.QUORUM: self._tally_quorum,
            VotingStrategy.FIRST_RESPONSE: self._tally_first,
        }

        result = strategy_fn[session.strategy](session)
        if result is not None:
            self._close_session(session, result)

        return {"result": session.result, "decided": session.decided, "session": session.to_dict()}

    async def run_vote(
        self,
        topic: str,
        voters: list[str],
        vote_collector: VoteCollector,
        strategy: VotingStrategy = VotingStrategy.MAJORITY,
        timeout_seconds: float = 60.0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run a complete voting session: create, collect votes, tally."""
        session = self.create_session(
            topic=topic,
            voters=voters,
            strategy=strategy,
            timeout_seconds=timeout_seconds,
            **kwargs,
        )

        try:
            collected_votes = await asyncio.wait_for(
                vote_collector(session.id, topic, voters),
                timeout=timeout_seconds,
            )
            for vote in collected_votes:
                self.cast_vote(session.id, vote)
        except asyncio.TimeoutError:
            for voter_id in voters:
                if not any(v.agent_id == voter_id for v in session.votes):
                    self.cast_vote(session.id, Vote(agent_id=voter_id, choice=VoteChoice.ABSTAIN, reasoning="timeout"))

        return self.tally(session.id)

    def get_session(self, session_id: str) -> VotingSession | None:
        return self._sessions.get(session_id)

    def _close_session(self, session: VotingSession, result: str) -> None:
        session.decided = True
        session.result = result
        session.closed_at = time.time()

    def _tally_majority(self, session: VotingSession) -> str | None:
        """Simple majority: more approves than rejects wins."""
        votes = [v for v in session.votes if v.choice != VoteChoice.ABSTAIN]
        total_voters = len(session.eligible_voters)

        if len(votes) < total_voters and not self._is_timed_out(session):
            return None

        approves = sum(1 for v in votes if v.choice == VoteChoice.APPROVE)
        rejects = sum(1 for v in votes if v.choice == VoteChoice.REJECT)

        if approves > rejects:
            return "approve"
        if rejects > approves:
            return "reject"
        return "tie"

    def _tally_unanimous(self, session: VotingSession) -> str | None:
        """All voters must agree."""
        votes = [v for v in session.votes if v.choice != VoteChoice.ABSTAIN]
        total_voters = len(session.eligible_voters)

        if len(votes) < total_voters and not self._is_timed_out(session):
            return None

        if all(v.choice == VoteChoice.APPROVE for v in votes) and len(votes) == total_voters:
            return "approve"
        if any(v.choice == VoteChoice.REJECT for v in votes):
            return "reject"
        if self._is_timed_out(session):
            return "reject"
        return None

    def _tally_weighted(self, session: VotingSession) -> str | None:
        """Weighted voting: each vote counts proportional to its weight."""
        votes = [v for v in session.votes if v.choice != VoteChoice.ABSTAIN]
        total_voters = len(session.eligible_voters)

        if len(votes) < total_voters and not self._is_timed_out(session):
            return None

        approve_weight = sum(v.weight for v in votes if v.choice == VoteChoice.APPROVE)
        reject_weight = sum(v.weight for v in votes if v.choice == VoteChoice.REJECT)

        if approve_weight > reject_weight:
            return "approve"
        if reject_weight > approve_weight:
            return "reject"
        return "tie"

    def _tally_quorum(self, session: VotingSession) -> str | None:
        """Quorum: need minimum participation, then majority decides."""
        votes = [v for v in session.votes if v.choice != VoteChoice.ABSTAIN]
        total_voters = len(session.eligible_voters)
        participation = len(votes) / max(1, total_voters)

        if participation < session.quorum_threshold and not self._is_timed_out(session):
            return None

        if participation < session.quorum_threshold:
            return "no_quorum"

        approves = sum(1 for v in votes if v.choice == VoteChoice.APPROVE)
        rejects = sum(1 for v in votes if v.choice == VoteChoice.REJECT)

        if approves > rejects:
            return "approve"
        if rejects > approves:
            return "reject"
        return "tie"

    def _tally_first(self, session: VotingSession) -> str | None:
        """First response wins."""
        if session.votes:
            return session.votes[0].choice.value
        if self._is_timed_out(session):
            return "no_response"
        return None

    @staticmethod
    def _is_timed_out(session: VotingSession) -> bool:
        return (time.time() - session.created_at) > session.timeout_seconds
