"""Data contracts for the generation layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from labor_rag.models.rag import EvidenceItem


GenerationStatus = Literal[
    "generated",
    "generated_with_warnings",
    "abstained",
]


@dataclass(slots=True, frozen=True)
class GenerationConfig:
    """Model-independent generation policy."""

    temperature: float = 0.2
    max_output_tokens: int = 1200
    require_citations: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0 and 2.")
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be greater than 0.")


@dataclass(slots=True, frozen=True)
class GenerationRequest:
    """Generation input produced directly by the retrieval/context pipeline."""

    query: str
    context: str
    evidence: tuple[EvidenceItem, ...] = ()
    source_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.query, str) or not self.query.strip():
            raise ValueError("query must be a non-empty string.")


@dataclass(slots=True)
class GenerationResult:
    """Structured output ready for a CLI or later FastAPI layer."""

    status: GenerationStatus
    query: str
    answer: str
    provider: str
    model: str
    latency_ms: float
    usage: dict[str, Any]
    evidence: list[EvidenceItem]
    citations_used: list[str]
    citation_warnings: list[str]
    source_metadata: dict[str, Any]
