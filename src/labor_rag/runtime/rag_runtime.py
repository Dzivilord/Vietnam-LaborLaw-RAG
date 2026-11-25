"""Application runtime layer for serving the Labor Law RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from .config import RAGRuntimeConfig
from labor_rag.generation import GenerationConfig
from labor_rag.generation.base import BaseLLMProvider
from labor_rag.generation.providers import OllamaProvider, OpenRouterProvider
from labor_rag.observability.logfire_spans import span as logfire_span


@dataclass
class RuntimeResult:
    answer: Any
    metadata: dict
    evidence: list[Any]


class RAGRuntime:
    """Stable entry point between API/CLI and the RAG pipeline.

    Runtime owns orchestration concerns:
    - pipeline lifecycle
    - request execution
    - response normalization
    - error boundary
    """

    def __init__(
        self,
        pipeline,
        config: RAGRuntimeConfig,
    ):
        self.pipeline = pipeline
        self.config = config

    def chat(self, query: str, **kwargs) -> RuntimeResult:
        started = perf_counter()
        try:
            generation_overrides = kwargs.pop("generation_overrides", None) or {}
            generation_provider, generation_config = self._generation_overrides(generation_overrides)
            with logfire_span("rag.request", **{"rag.query_length": len(query)}) as span:
                result = self.pipeline.run(
                    query,
                    generation_provider=generation_provider,
                    generation_config=generation_config,
                    **kwargs,
                )
                span.set_attribute("rag.status", result.status)

            return RuntimeResult(
                answer=result.answer,
                metadata={
                    "status": result.status,
                    "citations": [
                        evidence.evidence_id for evidence in result.evidence
                    ],
                    "diagnostics": result.diagnostics,
                    "total_latency_ms": round((perf_counter() - started) * 1000.0, 2),
                },
                evidence=list(result.evidence),
            )

        except Exception as exc:
            return RuntimeResult(
                answer=None,
                metadata={
                    "status": "error",
                    "error": str(exc),
                    "total_latency_ms": round((perf_counter() - started) * 1000.0, 2),
                },
                evidence=[],
            )

    def _generation_overrides(self, overrides: dict[str, Any]) -> tuple[BaseLLMProvider | None, GenerationConfig | None]:
        if not overrides:
            return None, None
        runtime_config = self.config.generation
        model = overrides.get("model") or runtime_config.model
        if runtime_config.provider == "ollama":
            provider: BaseLLMProvider = OllamaProvider(
                model=model, base_url=runtime_config.ollama_url,
                timeout=runtime_config.timeout, think=runtime_config.ollama_think,
            )
        else:
            fallback_models = tuple(overrides.get("fallback_models") or runtime_config.fallback_models)
            provider = OpenRouterProvider(
                model=model, fallback_models=fallback_models,
                api_key=runtime_config.openrouter_api_key, base_url=runtime_config.openrouter_base_url,
                referer=runtime_config.openrouter_referer, title=runtime_config.openrouter_title,
                timeout=runtime_config.timeout,
            )
        config = GenerationConfig(
            temperature=overrides.get("temperature", runtime_config.temperature),
            max_output_tokens=overrides.get("max_output_tokens", runtime_config.max_output_tokens),
            require_citations=runtime_config.require_citations,
        )
        return provider, config
