"""Deterministic (no-LLM) metrics for phase-2 RAG runs.

Input: rows written by ``evals.run_rag`` plus the golden cases.  Chunk matching
is by exact chunk ID (the corpus never holds an article chunk and clause chunks
of the same article at once, so exact IDs are unambiguous).

Groups
* retrieval  - hit@K, recall@K, MRR@20 over the fused retrieval list
* rerank     - hit@1, hit@5, recall@5, MRR@10 after rerank; gain vs retrieval
* context    - context_hit, context_recall, chunks per answer, threshold_loss
* answer     - answer_rate, false_abstain_rate, status counts
* citation   - citation precision/recall, uncited_answer_rate
* abstain    - correct_abstain_rate on cases that must not be answered
* performance- p50/p95 total and per stage, error_rate, completion tokens
* failure_stage - first stage at which an answerable case went wrong
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable

from .score_citations import score_case

ANSWERED_STATUSES = {"generated", "generated_with_warnings"}
FAILURE_STAGES = ("error", "retrieval", "rerank", "context", "abstained", "citation", "ok")


def actual_action(status: str, *, generate: bool = True) -> str:
    if status == "error":
        return "error"
    if status == "refused":
        return "refuse"
    if status == "clarification_required":
        return "clarify"
    if status in ANSWERED_STATUSES or (status == "ready" and not generate):
        return "answer"
    return "abstain"


def _mean(values: Iterable[float | None]) -> float | None:
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 4) if values else None


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[round((len(ordered) - 1) * q)], 2)


def first_hit_rank(ids: list[str], relevant: set[str]) -> int | None:
    for rank, chunk_id in enumerate(ids, start=1):
        if chunk_id in relevant:
            return rank
    return None


def case_metrics(row: dict[str, Any], gold: dict[str, Any]) -> dict[str, Any]:
    """Per-case facts used both for aggregation and for the Logfire case span."""
    relevant = set(gold.get("relevant_chunk_ids") or [])
    retrieved = [item["chunk_id"] for item in row.get("retrieved", [])]
    reranked = [item["chunk_id"] for item in row.get("reranked", [])]
    context = list(row.get("context_chunk_ids", []))
    out: dict[str, Any] = {"answerable": bool(relevant), "answered": row.get("actual_action") == "answer"}
    if not relevant:
        out["correct_abstain"] = row.get("actual_action") in {"abstain", "refuse", "clarify"}
        return out
    out.update({
        "retrieval_rank": first_hit_rank(retrieved, relevant),
        "rerank_rank": first_hit_rank(reranked, relevant) if reranked else None,
        "context_hit": bool(relevant & set(context)),
        "context_recall": round(len(relevant & set(context)) / len(relevant), 4),
        "context_chunks": len(context),
        "in_reranked": bool(relevant & set(reranked)),
    })
    if out["answered"] and row.get("answer") is not None and row.get("generate", True):
        cite = score_case(row.get("cited_legal_references") or [], gold.get("expected_legal_references") or [])
        out.update({"citation_precision": cite["citation_precision"], "citation_recall": cite["citation_recall"], "cited": cite["cited"]})
    out["failure_stage"] = _failure_stage(row, out)
    return out


def _failure_stage(row: dict[str, Any], m: dict[str, Any]) -> str:
    if row.get("error"):
        return "error"
    if m["retrieval_rank"] is None:
        return "retrieval"
    if row.get("reranked") and not m["in_reranked"]:
        return "rerank"
    if not m["context_hit"]:
        return "context"
    if not m["answered"]:
        return "abstained"
    if "citation_precision" in m and not m.get("cited"):
        return "citation"
    return "ok"


def _rank_block(ranks: list[int | None], recalls: dict[int, list[float]], ks: tuple[int, ...], mrr_k: int) -> dict[str, Any]:
    n = len(ranks)
    block: dict[str, Any] = {"n": n}
    for k in ks:
        block[f"hit@{k}"] = round(sum(r is not None and r <= k for r in ranks) / n, 4) if n else None
    for k, values in recalls.items():
        block[f"recall@{k}"] = _mean(values)
    block[f"mrr@{mrr_k}"] = round(sum(1 / r for r in ranks if r is not None and r <= mrr_k) / n, 4) if n else None
    return block


def _recall_at(ids: list[str], relevant: set[str], k: int) -> float:
    return len(relevant & set(ids[:k])) / len(relevant)


def aggregate(rows: list[dict[str, Any]], golden: dict[str, dict[str, Any]]) -> dict[str, Any]:
    pairs = [(row, golden[row["id"]]) for row in rows if row["id"] in golden]
    facts = {row["id"]: case_metrics(row, gold) for row, gold in pairs}
    answerable = [(row, gold) for row, gold in pairs if gold.get("relevant_chunk_ids")]
    probes = [(row, gold) for row, gold in pairs if not gold.get("relevant_chunk_ids")]
    report: dict[str, Any] = {"case_count": len(pairs), "answerable_count": len(answerable), "abstain_probe_count": len(probes)}

    if answerable:
        ret_ranks, rer_ranks = [], []
        ret_recall: dict[int, list[float]] = {5: [], 10: [], 20: []}
        rer_recall: dict[int, list[float]] = {5: []}
        has_rerank = any(row.get("reranked") for row, _ in answerable)
        for row, gold in answerable:
            relevant = set(gold["relevant_chunk_ids"])
            retrieved = [i["chunk_id"] for i in row.get("retrieved", [])]
            reranked = [i["chunk_id"] for i in row.get("reranked", [])]
            ret_ranks.append(facts[row["id"]]["retrieval_rank"])
            for k in ret_recall:
                ret_recall[k].append(_recall_at(retrieved, relevant, k))
            if has_rerank:
                rer_ranks.append(facts[row["id"]]["rerank_rank"])
                rer_recall[5].append(_recall_at(reranked, relevant, 5))
        report["retrieval"] = _rank_block(ret_ranks, ret_recall, (1, 5, 10, 20), 20)
        if has_rerank:
            report["rerank"] = _rank_block(rer_ranks, rer_recall, (1, 5), 10)
            report["rerank"]["hit@1_gain_vs_retrieval"] = round(report["rerank"]["hit@1"] - report["retrieval"]["hit@1"], 4)

        f = [facts[row["id"]] for row, _ in answerable]
        reached_rerank = [m for m in f if m["in_reranked"]]
        report["context"] = {
            "context_hit": _mean(float(m["context_hit"]) for m in f),
            "context_recall": _mean(m["context_recall"] for m in f),
            "mean_context_chunks": _mean(m["context_chunks"] for m in f),
            # Gold chunk survived rerank but the context thresholds dropped it.
            "threshold_loss_rate": _mean(float(not m["context_hit"]) for m in reached_rerank) if reached_rerank else None,
        }
        answered = [m for m in f if m["answered"]]
        report["answer"] = {
            "answer_rate": _mean(float(m["answered"]) for m in f),
            "false_abstain_rate": _mean(float(not m["answered"]) for m in f),
            "status_counts": dict(Counter(row.get("status") for row, _ in answerable)),
        }
        cited = [m for m in answered if "citation_precision" in m]
        if cited:
            report["citation"] = {
                "n": len(cited),
                "citation_precision": _mean(m["citation_precision"] for m in cited),
                "citation_recall": _mean(m["citation_recall"] for m in cited),
                "uncited_answer_rate": _mean(float(not m["cited"]) for m in cited),
            }
        report["failure_stage"] = {stage: sum(m["failure_stage"] == stage for m in f) for stage in FAILURE_STAGES}

    if probes:
        report["abstain"] = {"correct_abstain_rate": _mean(float(facts[row["id"]]["correct_abstain"]) for row, _ in probes)}

    report["performance"] = _performance(rows)
    report["by_category"] = _by_category(pairs, facts, has_rerank=any(row.get("reranked") for row, _ in pairs))
    return report


def _performance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [row for row in rows if not row.get("error")]
    perf: dict[str, Any] = {
        "error_rate": round(sum(bool(row.get("error")) for row in rows) / len(rows), 4) if rows else None,
        "p50_latency_ms": _percentile([row["latency_ms"] for row in ok if row.get("latency_ms") is not None], 0.50),
        "p95_latency_ms": _percentile([row["latency_ms"] for row in ok if row.get("latency_ms") is not None], 0.95),
    }
    stages: dict[str, list[float]] = defaultdict(list)
    for row in ok:
        for stage, value in (row.get("timings_ms") or {}).items():
            if value is not None:
                stages[stage].append(float(value))
    for stage, values in sorted(stages.items()):
        perf[f"p95_{stage}_ms"] = _percentile(values, 0.95)
    completion = [row.get("usage", {}).get("completion") for row in ok if row.get("usage", {}).get("completion") is not None]
    perf["mean_completion_tokens"] = _mean(completion)
    return perf


def _by_category(pairs, facts, *, has_rerank: bool) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row, gold in pairs:
        groups[gold.get("category", "unknown")].append(facts[row["id"]])
    out = {}
    for category, items in sorted(groups.items()):
        answerable = [m for m in items if m["answerable"]]
        entry: dict[str, Any] = {"n": len(items)}
        if answerable:
            entry.update({
                "retrieval_hit@5": _mean(float(m["retrieval_rank"] is not None and m["retrieval_rank"] <= 5) for m in answerable),
                "rerank_hit@1": _mean(float(m["rerank_rank"] == 1) for m in answerable) if has_rerank else None,
                "context_hit": _mean(float(m["context_hit"]) for m in answerable),
                "answer_rate": _mean(float(m["answered"]) for m in answerable),
                "citation_precision": _mean(m.get("citation_precision") for m in answerable),
            })
        else:
            entry["correct_abstain_rate"] = _mean(float(m["correct_abstain"]) for m in items)
        out[category] = entry
    return out
