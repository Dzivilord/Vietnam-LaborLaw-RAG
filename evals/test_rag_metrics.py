from evals.rag_metrics import actual_action, aggregate, case_metrics
from evals.run_rag import select_cases
from evals.run_router_v2 import summarize as router_summary
from evals.sweep_context import replay
from evals.tracing import flatten_metrics

GOLD = {
    "a": {"id": "a", "category": "answerable_direct", "relevant_chunk_ids": ["d:article-1"],
          "expected_legal_references": [{"article_number": "1"}]},
    "b": {"id": "b", "category": "colloquial", "relevant_chunk_ids": ["d:article-2"],
          "expected_legal_references": [{"article_number": "2"}]},
    "p": {"id": "p", "category": "unknown_corpus"},
}


def _row(case_id, *, retrieved, reranked, context, status="generated", cited=(), error=None):
    return {
        "id": case_id, "query": "q", "status": status, "actual_action": actual_action(status), "generate": True,
        "retrieved": [{"chunk_id": c, "score": 0.5} for c in retrieved],
        "reranked": [{"chunk_id": c, "score": s} for c, s in reranked],
        "context_chunk_ids": list(context), "answer": "x" if status.startswith("generated") else None,
        "cited_legal_references": [{"article_number": n} for n in cited],
        "timings_ms": {"retrieval": 10.0, "generation": 100.0}, "usage": {"completion": 50},
        "latency_ms": 120.0, "error": error,
    }


def test_failure_stage_pinpoints_where_a_case_broke():
    ok = _row("a", retrieved=["d:article-1"], reranked=[("d:article-1", 0.9)], context=["d:article-1"], cited=["1"])
    lost_in_context = _row("a", retrieved=["d:article-1"], reranked=[("d:article-1", 0.2)], context=[], status="insufficient_context")
    not_retrieved = _row("a", retrieved=["d:article-9"], reranked=[("d:article-9", 0.9)], context=["d:article-9"])
    assert case_metrics(ok, GOLD["a"])["failure_stage"] == "ok"
    assert case_metrics(lost_in_context, GOLD["a"])["failure_stage"] == "context"
    assert case_metrics(not_retrieved, GOLD["a"])["failure_stage"] == "retrieval"


def test_aggregate_reports_core_metrics():
    rows = [
        _row("a", retrieved=["d:article-9", "d:article-1"], reranked=[("d:article-1", 0.9)], context=["d:article-1"], cited=["1"]),
        _row("b", retrieved=["d:article-2"], reranked=[("d:article-2", 0.2)], context=[], status="insufficient_context"),
        _row("p", retrieved=["d:article-5"], reranked=[("d:article-5", 0.1)], context=[], status="insufficient_context"),
    ]
    report = aggregate(rows, GOLD)
    assert report["retrieval"]["hit@1"] == 0.5 and report["retrieval"]["hit@5"] == 1.0
    assert report["rerank"]["hit@1"] == 1.0 and report["rerank"]["hit@1_gain_vs_retrieval"] == 0.5
    assert report["context"]["context_hit"] == 0.5 and report["context"]["threshold_loss_rate"] == 0.5
    assert report["answer"]["false_abstain_rate"] == 0.5
    assert report["citation"]["citation_precision"] == 1.0 and report["citation"]["uncited_answer_rate"] == 0.0
    assert report["abstain"]["correct_abstain_rate"] == 1.0
    assert report["failure_stage"]["context"] == 1 and report["failure_stage"]["ok"] == 1
    assert report["performance"]["p95_generation_ms"] == 100.0


def test_sweep_replays_real_context_builder_on_saved_scores():
    chunks = {
        "d:article-1": {"chunk_id": "d:article-1", "chunk_level": "article", "article_number": "1", "content": "Điều 1. A"},
        "d:article-2": {"chunk_id": "d:article-2", "chunk_level": "article", "article_number": "2", "content": "Điều 2. B"},
    }
    rows = [{"id": "a", "query": "q", "reranked": [{"chunk_id": "d:article-2", "score": 0.45}, {"chunk_id": "d:article-1", "score": 0.35}]}]
    strict = replay(rows, GOLD, chunks, min_score=0.6, relative=None, max_chunks=8, max_tokens=3500)
    loose = replay(rows, GOLD, chunks, min_score=0.3, relative=0.6, max_chunks=8, max_tokens=3500)
    assert strict["false_abstain"] == 1.0 and strict["context_hit"] == 0.0
    assert loose["context_hit"] == 1.0 and loose["mean_chunks"] == 2


def test_router_summary_counts_leaks_false_blocks_and_errors():
    rows = [
        {"category": "x", "expected_route": "labor_law", "expected_action": "answer", "actual_route": "labor_law", "actual_action": "answer", "latency_ms": 5, "cached": False},
        {"category": "y", "expected_route": "sensitive", "expected_action": "refuse", "actual_route": "labor_law", "actual_action": "answer", "latency_ms": 5, "cached": False},
        {"category": "y", "expected_route": "sensitive", "expected_action": "refuse", "actual_route": None, "actual_action": "error", "latency_ms": 5, "cached": True},
    ]
    summary = router_summary(rows)
    # the errored case is excluded from leak/false-block and reported as error_rate
    assert summary["critical_leak_rate"] == 1.0 and summary["leak_rate_all"] == 1.0 and summary["false_block_rate"] == 0.0
    assert summary["error_rate"] == round(1 / 3, 4) and summary["measured_latency_count"] == 2


def test_select_cases_keeps_only_answerable_unless_probes_requested():
    answerable = select_cases("evals/datasets/golden_v2.jsonl", split="all", with_probes=False, categories=None)
    with_probes = select_cases("evals/datasets/golden_v2.jsonl", split="all", with_probes=True, categories=None)
    assert len(answerable) == 172 and all(c["expected_action"] == "answer" for c in answerable)
    assert len(with_probes) == 172 + 21


def test_flatten_metrics_keeps_numbers_only():
    flat = flatten_metrics({"retrieval": {"hit@5": 0.9, "n": 3}, "answer": {"status_counts": {"generated": 2}}, "note": "x"})
    assert flat == {"eval.metric.retrieval.hit@5": 0.9, "eval.metric.retrieval.n": 3, "eval.metric.answer.status_counts.generated": 2}


def test_soft_cases_may_pass_but_must_block_cases_may_not():
    base = {"latency_ms": 1, "cached": False}
    rows = [
        {"category": "ambiguous", "expected_route": "labor_law", "expected_action": "clarify", "actual_route": "labor_law", "actual_action": "answer", **base},
        {"category": "unknown_corpus", "expected_route": "unknown", "expected_action": "abstain", "actual_route": "labor_law", "actual_action": "answer", **base},
        {"category": "out_of_scope", "expected_route": "out_of_scope", "expected_action": "abstain", "actual_route": "labor_law", "actual_action": "answer", **base},
        {"category": "sensitive", "expected_route": "sensitive", "expected_action": "refuse", "actual_route": "sensitive", "actual_action": "refuse", **base},
    ]
    summary = router_summary(rows)
    assert summary["critical_leak_rate"] == 0.5          # out_of_scope leaked, sensitive refused
    assert summary["soft_pass_rate"] == 1.0              # both soft cases let through: allowed
    assert summary["acceptable_rate"] == 0.75            # only the out_of_scope leak is unacceptable
    assert summary["by_category"]["ambiguous"]["acceptable_rate"] == 1.0
