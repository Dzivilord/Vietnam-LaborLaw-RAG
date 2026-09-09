"""End-to-end in-memory labor-law RAG application pipeline.

This is the boundary a future FastAPI route should call.  No JSONL file is used
to hand data from retrieval/context building to generation.
"""

from __future__ import annotations

from labor_rag.generation import GenerationRequest, RAGGenerator
from labor_rag.models.rag import RAGRunResult
from labor_rag.pipelines.rag_pipeline import RetrievalContextPipeline


class LaborLawRAGPipeline:
    """Compose retrieval/context orchestration with optional answer generation."""

    def __init__(
        self,
        *,
        retrieval_pipeline: RetrievalContextPipeline,
        generator: RAGGenerator | None = None,
    ) -> None:
        self.retrieval_pipeline = retrieval_pipeline
        self.generator = generator

    def run(
        self,
        query: str,
        *,
        generate: bool = True,
        check_ambiguity: bool | None = None,
    ) -> RAGRunResult:
        upstream = self.retrieval_pipeline.run(
            query,
            check_ambiguity=check_ambiguity,
        )

        base = dict(
            query=upstream.query,
            answer=None,
            context=upstream.context,
            evidence=list(upstream.context_result.evidence),
            retrieval_results=list(upstream.retrieval_results),
            reranked_results=list(upstream.reranked_results),
            suggestions=list(upstream.suggestions),
            ambiguity_decision=upstream.ambiguity_decision,
            diagnostics={
                "pipeline": dict(upstream.diagnostics),
                "context": dict(upstream.context_result.diagnostics),
            },
        )

        if upstream.status != "ready":
            return RAGRunResult(status=upstream.status, generation=None, **base)

        if not generate or self.generator is None:
            return RAGRunResult(status="ready", generation=None, **base)

        generation = self.generator.generate(
            GenerationRequest(
                query=upstream.query,
                context=upstream.context,
                evidence=tuple(upstream.context_result.evidence),
                source_metadata={
                    "pipeline_status": upstream.status,
                    "context_status": upstream.context_result.status,
                },
            )
        )

        return RAGRunResult(
            status=generation.status,
            query=upstream.query,
            answer=generation.answer,
            context=upstream.context,
            evidence=list(generation.evidence),
            retrieval_results=list(upstream.retrieval_results),
            reranked_results=list(upstream.reranked_results),
            suggestions=list(upstream.suggestions),
            ambiguity_decision=upstream.ambiguity_decision,
            generation=generation,
            diagnostics={
                "pipeline": dict(upstream.diagnostics),
                "context": dict(upstream.context_result.diagnostics),
                "generation": {
                    "provider": generation.provider,
                    "model": generation.model,
                    "latency_ms": generation.latency_ms,
                    "usage": dict(generation.usage),
                    "citations_used": list(generation.citations_used),
                    "citation_warnings": list(generation.citation_warnings),
                },
            },
        )
