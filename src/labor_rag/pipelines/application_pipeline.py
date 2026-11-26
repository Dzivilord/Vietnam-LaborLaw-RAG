"""End-to-end in-memory labor-law RAG application pipeline.

This is the boundary a future FastAPI route should call.  No JSONL file is used
to hand data from retrieval/context building to generation.
"""

from __future__ import annotations

from labor_rag.generation import GenerationRequest, RAGGenerator
from labor_rag.generation.base import BaseLLMProvider
from labor_rag.generation.schemas import GenerationConfig
from labor_rag.guardrails import GuardrailService
from labor_rag.guardrails.service import user_message_for
from labor_rag.models.rag import RAGRunResult
from labor_rag.pipelines.rag_pipeline import RetrievalContextPipeline
from labor_rag.observability.logfire_spans import set_attributes as set_span_attributes
from labor_rag.observability.logfire_spans import span as logfire_span
from typing import Callable


class LaborLawRAGPipeline:
    """Compose retrieval/context orchestration with optional answer generation."""

    def __init__(
        self,
        *,
        retrieval_pipeline: RetrievalContextPipeline,
        generator: RAGGenerator | None = None,
        guardrails: GuardrailService | None = None,
    ) -> None:
        self.retrieval_pipeline = retrieval_pipeline
        self.generator = generator
        self.guardrails = guardrails

    def run(
        self,
        query: str,
        *,
        generate: bool = True,
        check_ambiguity: bool | None = None,
        enable_rerank: bool | None = None,
        generation_provider: BaseLLMProvider | None = None,
        generation_config: GenerationConfig | None = None,
        event_callback: Callable[[str, str, str], None] | None = None,
        skip_input_guardrail: bool = False,
    ) -> RAGRunResult:
        """``skip_input_guardrail`` is for evaluation only: phase-2 RAG evals run
        cases whose routing was already scored in phase 1, so the (possibly
        paid) input classifier is not called again. Evidence and output
        guardrails still apply."""
        use_input_guardrail = self.guardrails is not None and not skip_input_guardrail
        input_decision = None
        if use_input_guardrail:
            with logfire_span("rag.guardrail.input") as span:
                input_decision = self.guardrails.check_input(query)
                set_span_attributes(span, **{
                    "rag.guardrail_action": input_decision.action,
                    "rag.guardrail_category": input_decision.category,
                    "rag.guardrail_confidence": input_decision.confidence,
                })
        if input_decision is not None:
            self._emit_guardrail(event_callback, "started", "Checking query scope and policy.")
            self._emit_guardrail(event_callback, "completed", f"Input decision: {input_decision.action} ({input_decision.category}).")
            if not input_decision.allows_pipeline:
                return RAGRunResult(
                    status=self._status_for_guardrail(input_decision.action, input_decision.category),
                    query=query.strip(), answer=user_message_for(input_decision), context="", evidence=[],
                    diagnostics={"guardrails": {"input": input_decision.as_dict()}},
                )
        upstream = self.retrieval_pipeline.run(
            query,
            check_ambiguity=check_ambiguity,
            enable_rerank=enable_rerank,
            event_callback=event_callback,
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
                "guardrails": ({"input": input_decision.as_dict()} if input_decision else {}),
            },
        )

        if upstream.status != "ready":
            if self.guardrails and upstream.status == "insufficient_context":
                evidence_decision = self.guardrails.check_evidence(list(upstream.context_result.evidence))
                base["diagnostics"]["guardrails"]["evidence"] = evidence_decision.as_dict()
                self._emit_guardrail(event_callback, "completed", f"Evidence decision: {evidence_decision.action} ({evidence_decision.category}).")
                if not evidence_decision.allows_pipeline:
                    base["answer"] = user_message_for(evidence_decision)
                    return RAGRunResult(status="insufficient_context", generation=None, **base)
            return RAGRunResult(status=upstream.status, generation=None, **base)

        if self.guardrails:
            evidence_decision = self.guardrails.check_evidence(list(upstream.context_result.evidence))
            base["diagnostics"]["guardrails"]["evidence"] = evidence_decision.as_dict()
            self._emit_guardrail(event_callback, "completed", f"Evidence decision: {evidence_decision.action} ({evidence_decision.category}).")
            if not evidence_decision.allows_pipeline:
                base["answer"] = user_message_for(evidence_decision)
                return RAGRunResult(status="insufficient_context", generation=None, **base)

        if input_decision is not None and input_decision.metadata.get("clarify_after_retrieval"):
            # The router judged the query ambiguous. Instead of answering from a
            # broad context, return grounded options (articles that were found)
            # plus the evidence itself so the UI can show it; the user refines
            # the question or picks a topic. No LLM call.
            suggestions = self._suggestions_from_context(upstream)
            base["suggestions"] = suggestions
            base["answer"] = self._clarification_message(suggestions)
            base["diagnostics"]["clarification"] = {"source": "input_guardrail", "suggestion_count": len(suggestions)}
            self._emit_guardrail(event_callback, "completed", "Ambiguous query: returning topic suggestions instead of an answer.")
            return RAGRunResult(status="clarification_required", generation=None, **base)

        if not generate or self.generator is None:
            return RAGRunResult(status="ready", generation=None, **base)

        generator = self.generator if generation_provider is None and generation_config is None else RAGGenerator(
            generation_provider or self.generator.provider,
            config=generation_config or self.generator.config,
        )
        if event_callback is not None:
            event_callback("generation", "started", "Generating a grounded answer from the selected evidence.")
        with logfire_span("rag.generate", **{"rag.model": generator.provider.model_name, "rag.evidence_count": len(upstream.context_result.evidence)}) as span:
            generation = generator.generate(
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
            usage = dict(generation.usage or {})
            set_span_attributes(span, **{
                "rag.generation_status": generation.status,
                "rag.latency_ms": generation.latency_ms,
                "rag.usage.prompt": usage.get("prompt_eval_count", usage.get("prompt_tokens")),
                "rag.usage.completion": usage.get("eval_count", usage.get("completion_tokens")),
                "rag.citations_used": list(generation.citations_used),
                "rag.citation_warnings": list(generation.citation_warnings),
            })
        if event_callback is not None:
            event_callback("generation", "completed", "Answer generation completed.")

        if self.guardrails:
            output_decision = self.guardrails.check_output(generation)
            base["diagnostics"]["guardrails"]["output"] = output_decision.as_dict()
            self._emit_guardrail(event_callback, "completed", f"Output decision: {output_decision.action} ({output_decision.category}).")
            if not output_decision.allows_pipeline:
                return RAGRunResult(
                    status="abstained", query=upstream.query, answer=user_message_for(output_decision),
                    context=upstream.context, evidence=list(generation.evidence),
                    retrieval_results=list(upstream.retrieval_results), reranked_results=list(upstream.reranked_results),
                    suggestions=list(upstream.suggestions), ambiguity_decision=upstream.ambiguity_decision,
                    generation=generation, diagnostics={
                        **base["diagnostics"],
                        "generation": {
                            "provider": generation.provider, "model": generation.model,
                            "latency_ms": generation.latency_ms, "usage": dict(generation.usage),
                            "citations_used": list(generation.citations_used),
                            "citation_warnings": list(generation.citation_warnings),
                        },
                    },
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
                **base["diagnostics"],
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

    @staticmethod
    def _suggestions_from_context(upstream, limit: int = 8) -> list:
        """One suggestion per article among the selected evidence, in context order."""
        from labor_rag.query_understanding.ambiguity_classifier import Suggestion

        suggestions, seen = [], set()
        for item in upstream.context_result.evidence:
            article = item.article_number
            key = article or item.chunk_id
            if key in seen:
                continue
            seen.add(key)
            title = (getattr(item, "article_title", None) or "").strip()
            reference = f"Điều {article}" if article else item.chunk_id
            suggestions.append(Suggestion(
                label=f"{title} ({reference})" if title else reference,
                chunk_id=item.chunk_id, reranker_score=float(item.score),
                article_number=article, clause_number=item.clause_number,
            ))
            if len(suggestions) >= limit:
                break
        return suggestions

    @staticmethod
    def _clarification_message(suggestions: list) -> str:
        if not suggestions:
            return "Câu hỏi còn khá chung. Vui lòng nêu rõ tình huống cụ thể (loại hợp đồng, thời điểm, vấn đề cần hỏi)."
        topics = "\n".join(f"- {s.label}" for s in suggestions)
        return ("Câu hỏi còn khá chung và liên quan đến nhiều quy định. Bạn muốn hỏi về nội dung nào dưới đây, "
                "hoặc mô tả cụ thể hơn tình huống của bạn?\n" + topics)

    @staticmethod
    def _status_for_guardrail(action: str, category: str) -> str:
        if action == "refuse":
            return "refused"
        if action == "clarify":
            return "clarification_required"
        if category == "out_of_scope":
            return "out_of_scope"
        return "abstained"

    @staticmethod
    def _emit_guardrail(callback: Callable[[str, str, str], None] | None, status: str, message: str) -> None:
        if callback is not None:
            callback("guardrails", status, message)
