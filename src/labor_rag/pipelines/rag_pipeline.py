"""End-to-end retrieve -> rerank -> context orchestration.

The optional ambiguity feature is activation-based, not an always-on pipeline
stage.  By default it is disabled.  Even when enabled, it only runs when the
context builder has retained at least ``ambiguity_check_min_chunks`` relevant
chunks.  This preserves the normal fast path for focused questions.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Literal

from labor_rag.retrieval.base import BaseRetriever, RetrievalResult
from labor_rag.observability.logfire_spans import set_attributes as set_span_attributes
from labor_rag.observability.logfire_spans import span as logfire_span
from labor_rag.reranking.base import BaseReranker
from labor_rag.reranking.context_builder import (
    ContextBuildResult,
    RerankedContextBuilder,
)
from labor_rag.query_understanding.ambiguity_classifier import (
    AmbiguityDecision,
    QueryAmbiguityClassifier,
    Suggestion,
)


PipelineStatus = Literal[
    "ready",
    "clarification_required",
    "insufficient_context",
]


@dataclass(slots=True, frozen=True)
class RetrievalContextPipelineConfig:
    """Runtime policy for retrieval, reranking, and optional clarification."""

    retrieval_top_k: int = 20
    rerank_top_k: int = 10
    enable_rerank: bool = True
    no_rerank_context_k: int = 5
    enable_ambiguity_check: bool = False
    ambiguity_check_min_chunks: int = 4

    def __post_init__(self) -> None:
        if self.retrieval_top_k <= 0:
            raise ValueError("retrieval_top_k must be greater than 0.")
        if self.rerank_top_k <= 0:
            raise ValueError("rerank_top_k must be greater than 0.")
        if self.no_rerank_context_k <= 0:
            raise ValueError("no_rerank_context_k must be greater than 0.")
        if self.no_rerank_context_k > self.retrieval_top_k:
            raise ValueError("no_rerank_context_k cannot exceed retrieval_top_k.")
        if self.ambiguity_check_min_chunks <= 0:
            raise ValueError("ambiguity_check_min_chunks must be greater than 0.")


@dataclass(slots=True)
class RetrievalContextPipelineResult:
    """Structured pipeline output ready for a generation/API layer."""

    status: PipelineStatus
    query: str
    retrieval_results: list[RetrievalResult]
    reranked_results: list[RetrievalResult]
    context_result: ContextBuildResult
    context: str
    ambiguity_check_triggered: bool
    ambiguity_decision: AmbiguityDecision | None
    suggestions: list[Suggestion]
    diagnostics: dict[str, Any]

    @property
    def should_generate(self) -> bool:
        return self.status == "ready" and bool(self.context)


class RetrievalContextPipeline:
    """Compose an existing retriever, reranker, and context builder.

    Dependencies are injected rather than constructed here so this orchestration
    layer works unchanged with dense, BM25, hybrid RRF, or future retrievers.
    """

    def __init__(
        self,
        *,
        retriever: BaseRetriever,
        reranker: BaseReranker | None,
        context_builder: RerankedContextBuilder,
        ambiguity_classifier: QueryAmbiguityClassifier | None = None,
        config: RetrievalContextPipelineConfig | None = None,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.context_builder = context_builder
        self.ambiguity_classifier = ambiguity_classifier
        self.config = config or RetrievalContextPipelineConfig()

        if self.config.enable_rerank and self.reranker is None:
            raise ValueError("enable_rerank=True requires a reranker.")

        if (
            self.config.enable_ambiguity_check
            and self.ambiguity_classifier is None
        ):
            raise ValueError(
                "enable_ambiguity_check=True requires ambiguity_classifier."
            )

    def run(
        self,
        query: str,
        *,
        check_ambiguity: bool | None = None,
        enable_rerank: bool | None = None,
        event_callback: Callable[[str, str, str], None] | None = None,
    ) -> RetrievalContextPipelineResult:
        """Run retrieval -> reranking -> context -> optional ambiguity check.

        ``check_ambiguity`` is a per-request override:
        * ``None``  -> use pipeline config;
        * ``True``  -> activate feature for this request;
        * ``False`` -> bypass feature for this request.
        """
        started = perf_counter()
        timings_ms: dict[str, float] = {}
        query = self._validate_query(query)

        self._emit(event_callback, "retrieval", "started", "Searching dense and lexical evidence.")
        with logfire_span("rag.retrieve", **{"rag.top_k": self.config.retrieval_top_k}) as span:
            retrieved = self.retriever.retrieve(query, top_k=self.config.retrieval_top_k)
            set_span_attributes(span, **{
                "rag.retrieved_count": len(retrieved),
                "rag.top_chunk_ids": [item.chunk_id for item in retrieved[:5]],
            })
        timings_ms["retrieval"] = self._elapsed_ms(started)
        self._emit(event_callback, "retrieval", "completed", f"Retrieved {len(retrieved)} candidates.")
        rerank_enabled = self.config.enable_rerank if enable_rerank is None else bool(enable_rerank)
        if rerank_enabled:
            self._emit(event_callback, "reranking", "started", "Scoring candidate relevance.")
            if self.reranker is None:
                raise RuntimeError("Reranking is enabled but no reranker is configured.")
            with logfire_span("rag.rerank", **{"rag.candidates": len(retrieved), "rag.top_k": self.config.rerank_top_k}) as span:
                reranked = self.reranker.rerank(query, retrieved, top_k=self.config.rerank_top_k)
                set_span_attributes(span, **{
                    "rag.reranker": getattr(self.reranker, "model_name", type(self.reranker).__name__),
                    "rag.top_score": round(float(reranked[0].score), 4) if reranked else None,
                    "rag.top_chunk_ids": [item.chunk_id for item in reranked[:5]],
                    "rag.top_scores": [round(float(item.score), 4) for item in reranked[:5]],
                })
            timings_ms["reranking"] = self._elapsed_ms(started) - timings_ms["retrieval"]
            self._emit(event_callback, "reranking", "completed", f"Reranked {len(reranked)} candidates; context building is next.")
            self._emit(event_callback, "context", "started", "Selecting legal context and evidence.")
            with logfire_span("rag.context", **{"rag.source": "reranker"}) as span:
                context_result = self.context_builder.build(query, reranked)
                _annotate_context_span(span, context_result)
            timings_ms["context_building"] = self._elapsed_ms(started) - sum(timings_ms.values())
            context_source = "reranker"
        else:
            self._emit(event_callback, "context", "started", "Rerank skipped; selecting context from retrieval results.")
            reranked = []
            with logfire_span("rag.context", **{"rag.source": "retrieval"}) as span:
                context_result = self.context_builder.build_from_retrieval(query, retrieved, top_k=self.config.no_rerank_context_k)
                _annotate_context_span(span, context_result)
            timings_ms["context_building"] = self._elapsed_ms(started) - timings_ms["retrieval"]
            context_source = "retrieval_rank"
        self._emit(event_callback, "context", "completed", f"Selected {len(context_result.selected)} evidence chunks.")

        if context_result.status == "insufficient_context":
            return self._result(
                status="insufficient_context",
                query=query,
                retrieved=retrieved,
                reranked=reranked,
                context_result=context_result,
                context="",
                ambiguity_check_triggered=False,
                ambiguity_decision=None,
                suggestions=[],
                reason="no_chunk_survived_context_selection",
                context_source=context_source,
                timings_ms=timings_ms, rerank_enabled=rerank_enabled,
            )

        feature_enabled = (
            self.config.enable_ambiguity_check
            if check_ambiguity is None
            else bool(check_ambiguity)
        )
        selected_results = [item.result for item in context_result.selected]
        enough_chunks = (
            len(selected_results) >= self.config.ambiguity_check_min_chunks
        )

        # Fast path: normal specific query, feature disabled, or only a small
        # number of relevant chunks.  No ambiguity model is loaded or called.
        if not feature_enabled or not enough_chunks:
            reason = (
                "ambiguity_feature_disabled"
                if not feature_enabled
                else "too_few_relevant_chunks_for_ambiguity_check"
            )
            return self._result(
                status="ready",
                query=query,
                retrieved=retrieved,
                reranked=reranked,
                context_result=context_result,
                context=context_result.context,
                ambiguity_check_triggered=False,
                ambiguity_decision=None,
                suggestions=[],
                reason=reason,
                context_source=context_source,
                timings_ms=timings_ms, rerank_enabled=rerank_enabled,
            )

        if self.ambiguity_classifier is None:
            raise RuntimeError(
                "Ambiguity checking was requested but no classifier was configured."
            )

        decision = self.ambiguity_classifier.classify(
            query,
            selected_results,
        )
        timings_ms["ambiguity_check"] = self._elapsed_ms(started) - sum(timings_ms.values())

        if decision.requires_clarification:
            # Do not pass a perfectly valid but semantically broad context into
            # generation.  Return grounded titles so the UI/user can refine the
            # query first.
            return self._result(
                status="clarification_required",
                query=query,
                retrieved=retrieved,
                reranked=reranked,
                context_result=context_result,
                context="",
                ambiguity_check_triggered=True,
                ambiguity_decision=decision,
                suggestions=list(decision.suggestions),
                reason="ambiguous_query_over_multiple_relevant_facets",
                context_source=context_source,
                timings_ms=timings_ms, rerank_enabled=rerank_enabled,
            )

        # Both SPECIFIC and COMPREHENSIVE are answerable.  In the comprehensive
        # case, keeping all context-builder-selected evidence is intentional.
        return self._result(
            status="ready",
            query=query,
            retrieved=retrieved,
            reranked=reranked,
            context_result=context_result,
            context=context_result.context,
            ambiguity_check_triggered=True,
            ambiguity_decision=decision,
            suggestions=[],
            reason=f"query_classified_as_{decision.label}",
            context_source=context_source,
            timings_ms=timings_ms, rerank_enabled=rerank_enabled,
        )

    def _result(
        self,
        *,
        status: PipelineStatus,
        query: str,
        retrieved: list[RetrievalResult],
        reranked: list[RetrievalResult],
        context_result: ContextBuildResult,
        context: str,
        ambiguity_check_triggered: bool,
        ambiguity_decision: AmbiguityDecision | None,
        suggestions: list[Suggestion],
        reason: str,
        context_source: str,
        timings_ms: dict[str, float],
        rerank_enabled: bool,
    ) -> RetrievalContextPipelineResult:
        diagnostics: dict[str, Any] = {
            "reason": reason,
            "rerank_enabled": rerank_enabled,
            "context_source": context_source,
            "no_rerank_context_k": (
                self.config.no_rerank_context_k
                if not self.config.enable_rerank
                else None
            ),
            "retrieved_count": len(retrieved),
            "reranked_count": len(reranked),
            "relevant_context_count": len(context_result.selected),
            "ambiguity_feature_enabled_by_default": (
                self.config.enable_ambiguity_check
            ),
            "ambiguity_check_min_chunks": self.config.ambiguity_check_min_chunks,
            "ambiguity_check_triggered": ambiguity_check_triggered,
            "timings_ms": {
                stage: round(max(duration, 0.0), 2)
                for stage, duration in timings_ms.items()
            },
        }
        if ambiguity_decision is not None:
            diagnostics["ambiguity"] = {
                "label": ambiguity_decision.label,
                "score": ambiguity_decision.score,
                "predicted_label": ambiguity_decision.predicted_label,
                "predicted_score": ambiguity_decision.predicted_score,
                "decision_margin": ambiguity_decision.decision_margin,
                "raw_scores": dict(ambiguity_decision.raw_scores),
                "topics": list(ambiguity_decision.topics),
                "model": ambiguity_decision.model_name,
                "low_confidence": ambiguity_decision.low_confidence,
            }

        return RetrievalContextPipelineResult(
            status=status,
            query=query,
            retrieval_results=list(retrieved),
            reranked_results=list(reranked),
            context_result=context_result,
            context=context,
            ambiguity_check_triggered=ambiguity_check_triggered,
            ambiguity_decision=ambiguity_decision,
            suggestions=suggestions,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _validate_query(query: str) -> str:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")
        return query.strip()

    @staticmethod
    def _elapsed_ms(started: float) -> float:
        return (perf_counter() - started) * 1000.0

    @staticmethod
    def _emit(callback: Callable[[str, str, str], None] | None, stage: str, status: str, message: str) -> None:
        if callback is not None:
            callback(stage, status, message)


def _annotate_context_span(span: Any, context_result: ContextBuildResult) -> None:
    """Attach what context selection kept and why it dropped the rest."""
    rejected: dict[str, int] = {}
    for item in context_result.rejected:
        reason = str(item.get("reason", "unknown"))
        rejected[reason] = rejected.get(reason, 0) + 1
    set_span_attributes(span, **{
        "rag.context_status": context_result.status,
        "rag.selected_count": len(context_result.selected),
        "rag.selected_chunk_ids": [item.chunk_id for item in context_result.evidence],
        "rag.rejected_by_reason": rejected,
        "rag.context_tokens": context_result.diagnostics.get("estimated_context_tokens"),
    })
