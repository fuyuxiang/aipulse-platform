"""Conflict detection and resolution for multi-agent outputs.

Detects contradictions between agent responses and applies resolution
strategies: merge, vote, escalate, or priority-based selection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable

LLMInvoker = Callable[[str, str], Awaitable[str]]


class ResolutionStrategy(str, Enum):
    MERGE = "merge"
    VOTE = "vote"
    ESCALATE = "escalate"
    PRIORITY = "priority"
    LONGEST = "longest"
    LATEST = "latest"


class ConflictSeverity(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class AgentResponse:
    agent_id: str
    content: str
    priority: int = 0
    timestamp: float = 0.0
    confidence: float = 0.8
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConflictReport:
    severity: ConflictSeverity = ConflictSeverity.NONE
    conflicting_pairs: list[tuple[str, str]] = field(default_factory=list)
    details: str = ""
    indicators: list[str] = field(default_factory=list)


@dataclass
class ResolutionResult:
    strategy_used: ResolutionStrategy
    resolved_content: str
    selected_agent_id: str = ""
    confidence: float = 0.0
    explanation: str = ""
    all_responses: list[AgentResponse] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_used": self.strategy_used.value,
            "resolved_content": self.resolved_content,
            "selected_agent_id": self.selected_agent_id,
            "confidence": self.confidence,
            "explanation": self.explanation,
        }


CONTRADICTION_INDICATORS = [
    (r"\b(yes|agree|correct|true)\b", r"\b(no|disagree|incorrect|false|wrong)\b"),
    (r"\b(should|must|recommend)\b", r"\b(should not|must not|avoid|don't)\b"),
    (r"\b(increase|more|higher|up)\b", r"\b(decrease|less|lower|down|reduce)\b"),
    (r"\b(enable|activate|turn on)\b", r"\b(disable|deactivate|turn off)\b"),
    (r"\b(approve|accept|allow)\b", r"\b(reject|deny|block|refuse)\b"),
    (r"(可以|应该|建议|推荐)", r"(不可以|不应该|不建议|不推荐|避免)"),
    (r"(增加|提高|扩大)", r"(减少|降低|缩小)"),
    (r"(同意|赞成|支持)", r"(反对|否决|拒绝)"),
]


class ConflictResolver:
    """Detects and resolves conflicts between multiple agent responses."""

    def __init__(
        self,
        llm_invoker: LLMInvoker | None = None,
        default_strategy: ResolutionStrategy = ResolutionStrategy.PRIORITY,
    ):
        self._llm = llm_invoker
        self._default_strategy = default_strategy

    async def detect_conflict(self, responses: list[AgentResponse]) -> ConflictReport:
        """Detect contradictions between agent responses."""
        if len(responses) < 2:
            return ConflictReport(severity=ConflictSeverity.NONE)

        if self._llm and len(responses) <= 5:
            return await self._llm_detect(responses)

        return self._heuristic_detect(responses)

    async def resolve(
        self,
        responses: list[AgentResponse],
        strategy: ResolutionStrategy | None = None,
        conflict_report: ConflictReport | None = None,
    ) -> ResolutionResult:
        """Resolve conflicts between responses using the specified strategy."""
        effective_strategy = strategy or self._default_strategy

        if not responses:
            return ResolutionResult(
                strategy_used=effective_strategy,
                resolved_content="",
                explanation="no responses to resolve",
            )

        if len(responses) == 1:
            return ResolutionResult(
                strategy_used=effective_strategy,
                resolved_content=responses[0].content,
                selected_agent_id=responses[0].agent_id,
                confidence=responses[0].confidence,
                explanation="single response, no conflict",
            )

        if conflict_report is None:
            conflict_report = await self.detect_conflict(responses)

        if conflict_report.severity == ConflictSeverity.NONE:
            return await self._resolve_merge(responses)

        resolver_fn = {
            ResolutionStrategy.MERGE: self._resolve_merge,
            ResolutionStrategy.PRIORITY: self._resolve_priority,
            ResolutionStrategy.LONGEST: self._resolve_longest,
            ResolutionStrategy.LATEST: self._resolve_latest,
            ResolutionStrategy.ESCALATE: self._resolve_escalate,
            ResolutionStrategy.VOTE: self._resolve_vote,
        }

        resolver = resolver_fn.get(effective_strategy, self._resolve_priority)
        result = await resolver(responses)
        result.all_responses = responses
        return result

    def _heuristic_detect(self, responses: list[AgentResponse]) -> ConflictReport:
        """Detect conflicts using pattern matching heuristics."""
        conflicting_pairs: list[tuple[str, str]] = []
        indicators: list[str] = []

        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                text_a = responses[i].content.lower()
                text_b = responses[j].content.lower()

                for pos_pattern, neg_pattern in CONTRADICTION_INDICATORS:
                    has_pos_a = bool(re.search(pos_pattern, text_a, re.IGNORECASE))
                    has_neg_a = bool(re.search(neg_pattern, text_a, re.IGNORECASE))
                    has_pos_b = bool(re.search(pos_pattern, text_b, re.IGNORECASE))
                    has_neg_b = bool(re.search(neg_pattern, text_b, re.IGNORECASE))

                    if (has_pos_a and has_neg_b) or (has_neg_a and has_pos_b):
                        pair = (responses[i].agent_id, responses[j].agent_id)
                        if pair not in conflicting_pairs:
                            conflicting_pairs.append(pair)
                            indicators.append(f"{pos_pattern} vs {neg_pattern}")

        if not conflicting_pairs:
            similarity = self._text_similarity(
                responses[0].content, responses[1].content
            )
            if similarity < 0.2:
                severity = ConflictSeverity.LOW
            else:
                severity = ConflictSeverity.NONE
        elif len(conflicting_pairs) == 1:
            severity = ConflictSeverity.MEDIUM
        else:
            severity = ConflictSeverity.HIGH

        return ConflictReport(
            severity=severity,
            conflicting_pairs=conflicting_pairs,
            details=f"Found {len(conflicting_pairs)} contradicting pairs",
            indicators=indicators,
        )

    async def _llm_detect(self, responses: list[AgentResponse]) -> ConflictReport:
        """Use LLM to detect semantic conflicts."""
        response_text = "\n\n".join(
            f"Agent {r.agent_id}:\n{r.content}" for r in responses
        )
        prompt = (
            "Analyze the following agent responses for contradictions or conflicts. "
            "Rate the conflict severity as: none, low, medium, or high.\n\n"
            f"{response_text}\n\n"
            "Respond with ONLY one word: none, low, medium, or high."
        )
        try:
            result = await self._llm("system", prompt)
            result = result.strip().lower()
            severity_map = {
                "none": ConflictSeverity.NONE,
                "low": ConflictSeverity.LOW,
                "medium": ConflictSeverity.MEDIUM,
                "high": ConflictSeverity.HIGH,
            }
            severity = severity_map.get(result, ConflictSeverity.LOW)
            return ConflictReport(severity=severity, details=f"LLM assessment: {result}")
        except Exception:
            return self._heuristic_detect(responses)

    async def _resolve_merge(self, responses: list[AgentResponse]) -> ResolutionResult:
        """Merge non-conflicting parts of responses."""
        if self._llm:
            response_text = "\n\n".join(
                f"Agent {r.agent_id}:\n{r.content}" for r in responses
            )
            prompt = (
                "Merge the following agent responses into a single coherent response. "
                "Keep all non-contradicting information. For contradictions, prefer "
                "the response with higher confidence.\n\n"
                f"{response_text}\n\n"
                "Merged response:"
            )
            try:
                merged = await self._llm("system", prompt)
                return ResolutionResult(
                    strategy_used=ResolutionStrategy.MERGE,
                    resolved_content=merged.strip(),
                    confidence=max(r.confidence for r in responses) * 0.9,
                    explanation="LLM-merged responses",
                )
            except Exception:
                pass

        combined = "\n\n".join(r.content for r in responses)
        return ResolutionResult(
            strategy_used=ResolutionStrategy.MERGE,
            resolved_content=combined,
            confidence=0.6,
            explanation="concatenated responses (LLM unavailable for smart merge)",
        )

    async def _resolve_priority(self, responses: list[AgentResponse]) -> ResolutionResult:
        """Select response from highest priority agent."""
        sorted_responses = sorted(responses, key=lambda r: (r.priority, r.confidence), reverse=True)
        best = sorted_responses[0]
        return ResolutionResult(
            strategy_used=ResolutionStrategy.PRIORITY,
            resolved_content=best.content,
            selected_agent_id=best.agent_id,
            confidence=best.confidence,
            explanation=f"selected by priority ({best.priority}) and confidence ({best.confidence:.2f})",
        )

    async def _resolve_longest(self, responses: list[AgentResponse]) -> ResolutionResult:
        """Select the longest (most detailed) response."""
        best = max(responses, key=lambda r: len(r.content))
        return ResolutionResult(
            strategy_used=ResolutionStrategy.LONGEST,
            resolved_content=best.content,
            selected_agent_id=best.agent_id,
            confidence=best.confidence,
            explanation=f"selected longest response ({len(best.content)} chars)",
        )

    async def _resolve_latest(self, responses: list[AgentResponse]) -> ResolutionResult:
        """Select the most recent response."""
        best = max(responses, key=lambda r: r.timestamp)
        return ResolutionResult(
            strategy_used=ResolutionStrategy.LATEST,
            resolved_content=best.content,
            selected_agent_id=best.agent_id,
            confidence=best.confidence,
            explanation="selected most recent response",
        )

    async def _resolve_escalate(self, responses: list[AgentResponse]) -> ResolutionResult:
        """Mark as needing escalation to coordinator."""
        summary = "; ".join(f"{r.agent_id}: {r.content[:100]}" for r in responses)
        return ResolutionResult(
            strategy_used=ResolutionStrategy.ESCALATE,
            resolved_content=f"[ESCALATION NEEDED] Conflicting responses: {summary}",
            confidence=0.0,
            explanation="conflict requires coordinator intervention",
        )

    async def _resolve_vote(self, responses: list[AgentResponse]) -> ResolutionResult:
        """Placeholder for vote-based resolution (requires ConsensusEngine integration)."""
        return await self._resolve_priority(responses)

    @staticmethod
    def _text_similarity(text_a: str, text_b: str) -> float:
        """Simple word-overlap similarity."""
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union)
