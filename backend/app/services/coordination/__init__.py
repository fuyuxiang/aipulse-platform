"""Coordination package — multi-agent collaboration infrastructure.

Provides structured communication protocols, intelligent task delegation,
consensus voting, conflict resolution, and health monitoring for
multi-agent team coordination.
"""

from app.services.coordination.protocol import (
    CoordinationMessage,
    MessageType,
    MessageBroker,
)
from app.services.coordination.delegation import DelegationEngine
from app.services.coordination.consensus import ConsensusEngine, VotingStrategy
from app.services.coordination.conflict import ConflictResolver, ResolutionStrategy
from app.services.coordination.health import HealthMonitor, AgentHealth

__all__ = [
    "CoordinationMessage",
    "MessageType",
    "MessageBroker",
    "DelegationEngine",
    "ConsensusEngine",
    "VotingStrategy",
    "ConflictResolver",
    "ResolutionStrategy",
    "HealthMonitor",
    "AgentHealth",
]
