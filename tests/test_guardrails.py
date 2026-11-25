from __future__ import annotations

from labor_rag.guardrails import GuardrailService, GuardrailSettings, JEVClassifier, OpenRouterJEVClassifier
from labor_rag.pipelines.application_pipeline import LaborLawRAGPipeline
from labor_rag.pipelines.rag_pipeline import RetrievalContextPipeline, RetrievalContextPipelineConfig
from labor_rag.reranking.context_builder import RerankedContextBuilder
from labor_rag.retrieval.base import BaseRetriever


def test_rules_block_evasion_but_allow_a_legitimate_legal_question() -> None:
    service = GuardrailService()
    blocked = service.check_input("Làm sao để sa thải lao động mang thai mà không bị phát hiện?")
    allowed = service.check_input("Có được chấm dứt hợp đồng với lao động đang mang thai không?")

    assert (blocked.action, blocked.category) == ("refuse", "sensitive_or_disallowed")
    assert allowed.action == "continue"


def test_rules_abstain_for_clearly_out_of_scope_question() -> None:
    decision = GuardrailService().check_input("Dự báo thời tiết Hà Nội ngày mai?")
    assert (decision.action, decision.category) == ("abstain", "out_of_scope")


def test_jev_adapter_can_route_semantic_label() -> None:
    classifier = JEVClassifier(lambda _query: {"out_of_scope": 0.92, "in_scope": 0.08})
    service = GuardrailService(classifier=classifier)
    decision = service.check_input("Một câu không chứa keyword rule")
    assert (decision.action, decision.category, decision.confidence) == ("abstain", "out_of_scope", 0.92)


def test_empty_evidence_is_handoff_eligible() -> None:
    decision = GuardrailService(GuardrailSettings(handoff_eligible=True)).check_evidence([])
    assert decision.action == "abstain"
    assert decision.handoff_eligible is True


class _RetrieverMustNotRun(BaseRetriever):
    def retrieve(self, query: str, *, top_k: int = 5):
        raise AssertionError("input guardrail should stop before retrieval")


def test_pipeline_returns_out_of_scope_before_retrieval() -> None:
    retrieval_pipeline = RetrievalContextPipeline(
        retriever=_RetrieverMustNotRun(), reranker=None, context_builder=RerankedContextBuilder(),
        config=RetrievalContextPipelineConfig(enable_rerank=False),
    )
    pipeline = LaborLawRAGPipeline(retrieval_pipeline=retrieval_pipeline, guardrails=GuardrailService())
    result = pipeline.run("Cách nấu phở bò ngon tại nhà?")
    assert result.status == "out_of_scope"
    assert result.evidence == []
    assert result.diagnostics["guardrails"]["input"]["action"] == "abstain"


def test_openrouter_jev_adapter_maps_decisions_api_choice() -> None:
    classifier = OpenRouterJEVClassifier(api_key="test-key")
    classifier._post = lambda _payload: {  # type: ignore[method-assign]
        "answers": {"guardrail_category": {
            "choice": "sensitive_or_disallowed", "confidence": 0.94,
            "probabilities": {"sensitive_or_disallowed": 0.94, "in_scope": 0.06},
        }}
    }
    result = classifier.classify("Một câu hỏi giả lập")
    assert result.label == "sensitive_or_disallowed"
    assert result.confidence == 0.94
    assert result.source == "openrouter_jev"


class _GuardrailMustNotRun(GuardrailService):
    def check_input(self, query):  # type: ignore[override]
        raise AssertionError("input guardrail must be skipped")


def test_eval_can_skip_input_guardrail_but_keeps_evidence_gate() -> None:
    from labor_rag.retrieval.base import RetrievalResult

    class _EmptyRetriever(BaseRetriever):
        def retrieve(self, query, *, top_k=5):
            return [RetrievalResult("d:article-1", "Điều 1.", 0.01, {"chunk_level": "article", "article_number": "1"})]

    retrieval_pipeline = RetrievalContextPipeline(
        retriever=_EmptyRetriever(), reranker=None, context_builder=RerankedContextBuilder(),
        config=RetrievalContextPipelineConfig(enable_rerank=False),
    )
    pipeline = LaborLawRAGPipeline(retrieval_pipeline=retrieval_pipeline, guardrails=_GuardrailMustNotRun())
    result = pipeline.run("Cách nấu phở bò ngon tại nhà?", generate=False, skip_input_guardrail=True)
    assert "input" not in result.diagnostics["guardrails"]
    assert result.status in {"ready", "insufficient_context"}
