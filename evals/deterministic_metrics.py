"""No-LLM metrics for routing, retrieval, reranking, citations and safety."""

from __future__ import annotations

from collections import Counter
from math import log2
from statistics import median
from typing import Callable, Iterable

from .schemas import EvaluationCase


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return round(sum(values) / len(values), 4) if values else None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile)
    return round(ordered[index], 2)


def classification_report(cases: list[EvaluationCase], expected: str, actual: str) -> dict:
    pairs = [
        (getattr(case, expected), getattr(case, actual))
        for case in cases
        if getattr(case, expected) is not None and getattr(case, actual) is not None
    ]
    labels = sorted({label for pair in pairs for label in pair})
    if not pairs:
        return {"case_count": 0, "accuracy": None, "macro_f1": None, "per_label": {}, "confusion_matrix": {}}
    per_label = {}
    for label in labels:
        true_positive = sum(want == label and got == label for want, got in pairs)
        false_positive = sum(want != label and got == label for want, got in pairs)
        false_negative = sum(want == label and got != label for want, got in pairs)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4), "support": sum(want == label for want, _ in pairs)}
    confusion = {label: Counter() for label in labels}
    for want, got in pairs:
        confusion[want][got] += 1
    return {
        "case_count": len(pairs),
        "accuracy": round(sum(want == got for want, got in pairs) / len(pairs), 4),
        "macro_f1": _mean(item["f1"] for item in per_label.values()),
        "per_label": per_label,
        "confusion_matrix": {label: dict(counts) for label, counts in confusion.items()},
    }


def rank_report(cases: list[EvaluationCase], field: str, ks: tuple[int, ...] = (1, 5, 10, 20)) -> dict:
    eligible = [case for case in cases if case.relevant_chunk_ids and getattr(case, field)]
    result: dict[str, float | int | None] = {"case_count": len(eligible)}
    for k in ks:
        recall: list[float] = []
        precision: list[float] = []
        hit_rate: list[float] = []
        mrr: list[float] = []
        ndcg: list[float] = []
        for case in eligible:
            relevant = set(case.relevant_chunk_ids)
            ranking = list(getattr(case, field))[:k]
            hits = [index for index, chunk_id in enumerate(ranking, start=1) if chunk_id in relevant]
            hit_count = len(hits)
            recall.append(hit_count / len(relevant))
            precision.append(hit_count / k)
            hit_rate.append(float(bool(hits)))
            mrr.append(1 / hits[0] if hits else 0.0)
            dcg = sum(1 / log2(index + 1) for index in hits)
            ideal = sum(1 / log2(index + 1) for index in range(1, min(len(relevant), k) + 1))
            ndcg.append(dcg / ideal if ideal else 0.0)
        result.update({
            f"recall_at_{k}": _mean(recall), f"precision_at_{k}": _mean(precision),
            f"hit_rate_at_{k}": _mean(hit_rate), f"mrr_at_{k}": _mean(mrr), f"ndcg_at_{k}": _mean(ndcg),
        })
    return result


def citation_report(cases: list[EvaluationCase]) -> dict:
    eligible = [case for case in cases if case.citation_ids or case.expected_citation_ids]
    validity = []
    expected_recall = []
    for case in eligible:
        cited, evidence, expected = set(case.citation_ids), set(case.evidence_ids), set(case.expected_citation_ids)
        validity.append(len(cited & evidence) / len(cited) if cited else 0.0)
        if expected:
            expected_recall.append(len(cited & expected) / len(expected))
    return {"case_count": len(eligible), "citation_validity": _mean(validity), "expected_citation_recall": _mean(expected_recall)}


def abstention_report(cases: list[EvaluationCase]) -> dict:
    report = classification_report(cases, "expected_action", "actual_action")
    pairs = [
        (case.expected_action, case.actual_action)
        for case in cases if case.expected_action and case.actual_action
    ]
    should_not_answer = [actual for expected, actual in pairs if expected in {"abstain", "clarify", "refuse"}]
    report["unsafe_answer_rate"] = _mean(actual == "answer" for actual in should_not_answer)
    report["protected_case_count"] = len(should_not_answer)
    return report


def performance_report(cases: list[EvaluationCase]) -> dict:
    latencies = [case.latency_ms for case in cases if case.latency_ms is not None]
    errors = [case for case in cases if case.error]
    return {
        "case_count": len(cases), "measured_latency_count": len(latencies),
        "p50_latency_ms": round(median(latencies), 2) if latencies else None,
        "p95_latency_ms": _percentile(latencies, 0.95),
        "max_latency_ms": round(max(latencies), 2) if latencies else None,
        "error_rate": round(len(errors) / len(cases), 4) if cases else None,
    }


def build_report(cases: list[EvaluationCase]) -> dict:
    return {
        "routing": classification_report(cases, "expected_route", "actual_route"),
        "retrieval": rank_report(cases, "retrieved_chunk_ids"),
        "reranking": rank_report(cases, "reranked_chunk_ids"),
        "generation": citation_report(cases),
        "safety_and_abstention": abstention_report(cases),
        "performance_and_reliability": performance_report(cases),
    }
