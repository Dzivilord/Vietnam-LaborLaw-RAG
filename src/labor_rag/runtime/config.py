"""Runtime configuration for composing the labor-law RAG stack.

The goal is to keep model/top-k/reranker/ambiguity/context/generation choices
outside scripts so the same configuration can later be constructed from a
FastAPI request or application settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml


RetrievalMethod = Literal["dense", "bm25", "hybrid"]
EmbeddingModelKey = Literal["bge", "dek21"]
RerankerType = Literal["bge", "qwen3"]
GenerationProvider = Literal["ollama", "openrouter"]
GuardrailClassifierProvider = Literal["rules", "jev", "openrouter_jev"]
AmbiguityMode = Literal["clarify", "clarify_after_retrieval", "broad_context"]
UnknownAction = Literal["abstain", "continue"]


@dataclass(slots=True, frozen=True)
class RetrievalRuntimeConfig:
    method: RetrievalMethod = "hybrid"
    top_k: int = 20
    candidate_k: int = 20

    data_path: Path | None = None
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    embedding_model: EmbeddingModelKey = "bge"
    collection: str | None = None
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    # Embedded local Qdrant directory; when set, qdrant_url is ignored.
    qdrant_path: Path | None = None
    device: str | None = None
    dek21_dimension: int = 768

    rrf_k: int = 60
    dense_weight: float = 1.0
    bm25_weight: float = 1.0


@dataclass(slots=True, frozen=True)
class QueryUnderstandingRuntimeConfig:
    """Offline query rewriting and direct ``Điều N`` lookup (off by default)."""

    enabled: bool = False
    article_lookup: bool = True
    expand_terms: bool = True
    restore_diacritics: bool = True
    expand_abbreviations: bool = True


@dataclass(slots=True, frozen=True)
class RerankerRuntimeConfig:
    enabled: bool = True
    type: RerankerType = "bge"
    model: str | None = None
    device: str | None = None
    top_k: int = 10
    no_rerank_context_k: int = 5
    batch_size: int = 8
    max_length: int | None = None
    qwen3_instruction: str | None = None


@dataclass(slots=True, frozen=True)
class ContextRuntimeConfig:
    min_score: float = 0.60
    relative_to_top: float | None = None
    score_gap: float | None = None
    max_chunks: int = 8
    max_tokens: int | None = 3500
    order_by: Literal["legal", "relevance"] = "legal"


@dataclass(slots=True, frozen=True)
class AmbiguityRuntimeConfig:
    enabled: bool = False
    min_relevant_chunks: int = 4
    model: str = "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"
    device: str | None = None
    min_decision_score: float = 0.45
    min_decision_margin: float = 0.08


@dataclass(slots=True, frozen=True)
class GenerationRuntimeConfig:
    enabled: bool = False
    provider: GenerationProvider = "ollama"
    model: str | None = None
    fallback_models: tuple[str, ...] = ()
    temperature: float = 0.2
    max_output_tokens: int = 1200
    require_citations: bool = True

    ollama_url: str | None = None
    ollama_think: bool = False
    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_referer: str | None = None
    openrouter_title: str = "Labor Law RAG"
    timeout: float = 180.0


@dataclass(slots=True, frozen=True)
class GuardrailClassifierRuntimeConfig:
    provider: GuardrailClassifierProvider = "rules"
    model: str | None = None
    device: str | None = None
    threshold: float = 0.80
    # Per-label overrides, e.g. {"unknown": 0.35, "ambiguous": 0.35}.
    label_thresholds: dict[str, float] = field(default_factory=dict)
    base_url: str = "https://openrouter.ai/api/alpha/decisions"
    timeout: float = 15.0
    referer: str | None = None
    title: str = "Labor Law RAG Guardrails"


@dataclass(slots=True, frozen=True)
class GuardrailRuntimeConfig:
    enabled: bool = True
    scope_enabled: bool = True
    sensitive_enabled: bool = True
    rules_enabled: bool = True
    ambiguity_enabled: bool = True
    ambiguity_mode: AmbiguityMode = "broad_context"
    unknown_action: UnknownAction = "abstain"
    evidence_quality_enabled: bool = True
    handoff_eligible: bool = True
    output_enabled: bool = True
    require_citations: bool = True
    reject_invalid_citations: bool = True
    classifier: GuardrailClassifierRuntimeConfig = field(default_factory=GuardrailClassifierRuntimeConfig)


@dataclass(slots=True, frozen=True)
class RAGRuntimeConfig:
    retrieval: RetrievalRuntimeConfig = field(default_factory=RetrievalRuntimeConfig)
    query_understanding: QueryUnderstandingRuntimeConfig = field(default_factory=QueryUnderstandingRuntimeConfig)
    reranker: RerankerRuntimeConfig = field(default_factory=RerankerRuntimeConfig)
    context: ContextRuntimeConfig = field(default_factory=ContextRuntimeConfig)
    ambiguity: AmbiguityRuntimeConfig = field(default_factory=AmbiguityRuntimeConfig)
    generation: GenerationRuntimeConfig = field(default_factory=GenerationRuntimeConfig)
    guardrails: GuardrailRuntimeConfig = field(default_factory=GuardrailRuntimeConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RAGRuntimeConfig":
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as stream:
            values = yaml.safe_load(stream) or {}

        project_root = config_path.resolve().parent.parent
        retrieval_values = dict(values.get("retrieval", {}))
        if "type" in retrieval_values:
            retrieval_values["method"] = retrieval_values.pop("type")
        data_path = retrieval_values.get("data_path")
        if data_path:
            data_path = Path(data_path)
            if not data_path.is_absolute():
                data_path = project_root / data_path
            retrieval_values["data_path"] = data_path
        qdrant_path = retrieval_values.get("qdrant_path")
        if qdrant_path:
            qdrant_path = Path(qdrant_path)
            if not qdrant_path.is_absolute():
                qdrant_path = project_root / qdrant_path
            retrieval_values["qdrant_path"] = qdrant_path

        embedding_values = values.get("embedding", {})
        embedding_model = embedding_values.get("model", "bge")
        if embedding_model != "bge":
            embedding_model = "dek21"
        retrieval_values["embedding_model"] = embedding_model

        reranker_values = dict(values.get("reranker", {}))
        reranker_values.setdefault("enabled", True)
        reranker_values.setdefault("type", "bge")

        generation_values = dict(values.get("generation", {}))
        generation_values.setdefault("enabled", True)
        if "fallback_models" in generation_values:
            generation_values["fallback_models"] = tuple(generation_values["fallback_models"] or ())

        guardrail_values = dict(values.get("guardrails", {}))
        guardrail_classifier_values = dict(guardrail_values.pop("classifier", {}))

        return cls(
            retrieval=RetrievalRuntimeConfig(
                **retrieval_values,
            ),
            query_understanding=QueryUnderstandingRuntimeConfig(**values.get("query_understanding", {})),
            reranker=RerankerRuntimeConfig(**reranker_values),
            context=ContextRuntimeConfig(**values.get("context", {})),
            ambiguity=AmbiguityRuntimeConfig(**values.get("ambiguity", {})),
            generation=GenerationRuntimeConfig(**generation_values),
            guardrails=GuardrailRuntimeConfig(
                **guardrail_values,
                classifier=GuardrailClassifierRuntimeConfig(**guardrail_classifier_values),
            ),
        )
