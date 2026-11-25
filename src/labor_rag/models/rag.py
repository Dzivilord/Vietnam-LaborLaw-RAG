"""Shared RAG schemas used across context, generation, CLI, and future APIs.

These dataclasses are intentionally framework-agnostic.  FastAPI/Pydantic models
can later mirror them at the HTTP boundary without forcing the core pipeline to
depend on FastAPI.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from labor_rag.generation.schemas import GenerationResult
    from labor_rag.query_understanding.ambiguity_classifier import AmbiguityDecision, Suggestion
    from labor_rag.retrieval.base import RetrievalResult


RankSource = Literal["reranker", "retrieval"]
RAGStatus = Literal[
    "ready",
    "generated",
    "generated_with_warnings",
    "clarification_required",
    "insufficient_context",
    "abstained",
    "out_of_scope",
    "refused",
]


@dataclass(slots=True, frozen=True)
class EvidenceItem:
    """One UI/generation-ready evidence unit with provenance.

    Unlike the old ``evidence_manifest`` dictionary hidden inside diagnostics,
    this is a first-class schema.  It keeps the original chunk content together
    with legal hierarchy and retrieval/reranking provenance so the UI can show
    an answer next to its exact source without parsing the rendered context.
    """

    evidence_id: str
    chunk_id: str
    content: str
    rank_source: RankSource
    rank: int
    score: float
    estimated_tokens: int

    retrieval_score: float | None = None
    reranker_score: float | None = None
    dense_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None

    chunk_level: str | None = None
    parent_id: str | None = None
    document_id: str | None = None
    document_title: str | None = None
    document_number: str | None = None
    document_type: str | None = None
    issued_date: str | None = None
    effective_date: str | None = None
    source_file: str | None = None

    chapter_id: str | None = None
    chapter_number: str | None = None
    chapter_title: str | None = None
    section_id: str | None = None
    section_number: str | None = None
    section_title: str | None = None
    article_id: str | None = None
    article_number: str | None = None
    article_title: str | None = None
    clause_id: str | None = None
    clause_number: str | None = None
    point_id: str | None = None
    point_number: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RAGRunResult:
    """Unified end-to-end result suitable for CLI now and FastAPI later."""

    status: RAGStatus
    query: str
    answer: str | None
    context: str
    evidence: list[EvidenceItem]

    retrieval_results: list[RetrievalResult] = field(default_factory=list)
    reranked_results: list[RetrievalResult] = field(default_factory=list)
    suggestions: list[Suggestion] = field(default_factory=list)
    ambiguity_decision: AmbiguityDecision | None = None
    generation: GenerationResult | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def should_generate(self) -> bool:
        return self.status == "ready" and bool(self.context)


def json_safe(value: Any) -> Any:
    """Recursively convert core dataclasses to JSON-safe Python values."""

    if is_dataclass(value) and not isinstance(value, type):
        return json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
