"""Typed contracts for guardrail classification and policy decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


GuardrailAction = Literal["continue", "abstain", "clarify", "refuse"]
GuardrailLabel = Literal[
    "in_scope", "out_of_scope", "sensitive_or_disallowed", "ambiguous", "unknown",
]


@dataclass(frozen=True, slots=True)
class QueryClassification:
    label: GuardrailLabel
    confidence: float
    scores: dict[str, float] = field(default_factory=dict)
    source: str = "rules"
    matched_rules: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GuardrailDecision:
    action: GuardrailAction
    category: str
    reason: str
    confidence: float | None = None
    handoff_eligible: bool = False
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def allows_pipeline(self) -> bool:
        return self.action == "continue"

    def as_dict(self) -> dict[str, object]:
        return {
            "action": self.action,
            "category": self.category,
            "reason": self.reason,
            "confidence": self.confidence,
            "handoff_eligible": self.handoff_eligible,
            **self.metadata,
        }
