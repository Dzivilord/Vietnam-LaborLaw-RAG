from evals.deterministic_metrics import build_report
from evals.schemas import EvaluationCase


def test_offline_report_covers_all_quality_groups() -> None:
    cases = [
        EvaluationCase(
            id="ok", query="q", expected_route="legal", actual_route="legal",
            expected_action="answer", actual_action="answer", relevant_chunk_ids=("a",),
            retrieved_chunk_ids=("x", "a"), reranked_chunk_ids=("a",),
            expected_citation_ids=("E1",), citation_ids=("E1",), evidence_ids=("E1",), latency_ms=12,
        ),
        EvaluationCase(
            id="unsafe", query="q2", expected_route="out_of_scope", actual_route="legal",
            expected_action="abstain", actual_action="answer", latency_ms=30, error="bad route",
        ),
    ]
    report = build_report(cases)
    assert report["routing"]["case_count"] == 2
    assert report["retrieval"]["recall_at_5"] == 1.0
    assert report["reranking"]["mrr_at_1"] == 1.0
    assert report["generation"]["citation_validity"] == 1.0
    assert report["safety_and_abstention"]["unsafe_answer_rate"] == 1.0
    assert report["performance_and_reliability"]["error_rate"] == 0.5
