"""Contracts for immutable golden cases and captured evaluation results."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GoldenCase:
    """A human-labelled benchmark case; it must never include run output.

    ``expected_legal_references`` uses stable legal metadata rather than E1,
    E2… because evidence IDs are created afresh by each context-building run.
    ``relevant_chunk_ids`` is intentionally retained for retrieval/reranking
    metrics but contains no chunk text.
    """

    id: str
    query: str
    expected_case_type: str
    expected_route: str
    expected_action: str
    reference_answer: str | None = None
    expected_legal_references: tuple[dict[str, Any], ...] = ()
    relevant_chunk_ids: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    difficulty: str | None = None
    notes: str | None = None
    source_corpus: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "GoldenCase":
        required = ("id", "query", "expected_case_type", "expected_route", "expected_action")
        for key in required:
            if not raw.get(key):
                raise ValueError(f"Golden case is missing required field '{key}'.")
        prohibited = {key for key in raw if key.startswith("actual_") or key in {
            "answer", "citation_ids", "evidence_ids", "retrieval_context", "latency_ms", "error",
            "retrieved_chunk_ids", "reranked_chunk_ids",
        }}
        if prohibited:
            raise ValueError(f"Golden case must not contain captured output fields: {sorted(prohibited)}")
        values = dict(raw)
        for field_name in ("expected_legal_references", "relevant_chunk_ids", "tags"):
            values[field_name] = tuple(values.get(field_name) or ())
        known = set(cls.__dataclass_fields__)
        metadata = dict(values.get("metadata") or {})
        metadata.update({key: value for key, value in values.items() if key not in known})
        values["metadata"] = metadata
        return cls(**{key: value for key, value in values.items() if key in known})


@dataclass(frozen=True)
class EvaluationCase:
    id: str
    query: str
    expected_route: str | None = None
    actual_route: str | None = None
    expected_action: str | None = None
    actual_action: str | None = None
    relevant_chunk_ids: tuple[str, ...] = ()
    retrieved_chunk_ids: tuple[str, ...] = ()
    reranked_chunk_ids: tuple[str, ...] = ()
    expected_citation_ids: tuple[str, ...] = ()
    citation_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    answer: str | None = None
    reference_answer: str | None = None
    retrieval_context: tuple[str, ...] = ()
    latency_ms: float | None = None
    error: str | None = None
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "EvaluationCase":
        for key in ("id", "query"):
            if not raw.get(key):
                raise ValueError(f"Evaluation case is missing required field '{key}'.")

        tuple_fields = (
            "relevant_chunk_ids", "retrieved_chunk_ids", "reranked_chunk_ids",
            "expected_citation_ids", "citation_ids", "evidence_ids",
            "retrieval_context", "tags",
        )
        values = dict(raw)
        for field_name in tuple_fields:
            values[field_name] = tuple(values.get(field_name) or ())
        known = {field.name for field in cls.__dataclass_fields__.values()}
        metadata = dict(values.get("metadata") or {})
        metadata.update({key: value for key, value in values.items() if key not in known})
        values["metadata"] = metadata
        return cls(**{key: value for key, value in values.items() if key in known})


def load_cases(path: str | Path) -> list[EvaluationCase]:
    dataset_path = Path(path)
    cases: list[EvaluationCase] = []
    with dataset_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            try:
                raw = json.loads(line)
                cases.append(EvaluationCase.from_dict(raw))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid case at {dataset_path}:{line_number}: {exc}") from exc
    if not cases:
        raise ValueError(f"Dataset '{dataset_path}' has no evaluation cases.")
    return cases


def load_golden_cases(path: str | Path) -> list[GoldenCase]:
    """Load a labelled JSONL benchmark without any system output fields."""
    dataset_path = Path(path)
    cases: list[GoldenCase] = []
    with dataset_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            try:
                cases.append(GoldenCase.from_dict(json.loads(line)))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid golden case at {dataset_path}:{line_number}: {exc}") from exc
    if not cases:
        raise ValueError(f"Golden dataset '{dataset_path}' has no cases.")
    return cases
