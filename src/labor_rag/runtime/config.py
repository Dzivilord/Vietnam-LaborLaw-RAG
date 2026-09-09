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
GenerationProvider = Literal["gemini", "ollama"]


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
    device: str | None = None
    dek21_dimension: int = 768

    rrf_k: int = 60
    dense_weight: float = 1.0
    bm25_weight: float = 1.0


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
    temperature: float = 0.2
    max_output_tokens: int = 1200
    require_citations: bool = True

    ollama_url: str | None = None
    ollama_think: bool = False
    timeout: float = 180.0


@dataclass(slots=True, frozen=True)
class RAGRuntimeConfig:
    retrieval: RetrievalRuntimeConfig = field(default_factory=RetrievalRuntimeConfig)
    reranker: RerankerRuntimeConfig = field(default_factory=RerankerRuntimeConfig)
    context: ContextRuntimeConfig = field(default_factory=ContextRuntimeConfig)
    ambiguity: AmbiguityRuntimeConfig = field(default_factory=AmbiguityRuntimeConfig)
    generation: GenerationRuntimeConfig = field(default_factory=GenerationRuntimeConfig)

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

        return cls(
            retrieval=RetrievalRuntimeConfig(
                **retrieval_values,
            ),
            reranker=RerankerRuntimeConfig(**reranker_values),
            context=ContextRuntimeConfig(**values.get("context", {})),
            ambiguity=AmbiguityRuntimeConfig(**values.get("ambiguity", {})),
            generation=GenerationRuntimeConfig(**generation_values),
        )
