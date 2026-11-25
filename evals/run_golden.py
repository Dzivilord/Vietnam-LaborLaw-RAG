"""Execute the current RAG pipeline against a golden dataset and score it.

The golden JSONL remains immutable. This command writes captured results to
``evals/runs/`` and a merged report to ``evals/reports/``. Routes are taken
from the input-guardrail decision when guardrails are enabled.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from .deterministic_metrics import build_report
from .schemas import EvaluationCase, GoldenCase, load_golden_cases


def _actual_action(status: str) -> str:
    if status == "refused":
        return "refuse"
    if status == "out_of_scope":
        return "abstain"
    if status == "clarification_required":
        return "clarify"
    if status in {"insufficient_context", "abstained"}:
        return "abstain"
    return "answer"


def _actual_route(result: Any) -> str | None:
    input_decision = result.diagnostics.get("guardrails", {}).get("input", {})
    category = input_decision.get("category")
    if category == "out_of_scope":
        return "out_of_scope"
    if category == "sensitive_or_disallowed":
        return "sensitive"
    if category == "unknown":
        return "unknown"
    if category in {"in_scope", "ambiguous"}:
        return "labor_law"
    return None


def _legal_reference(evidence: Any) -> dict[str, str | None]:
    return {
        "document_id": evidence.document_id,
        "document_number": evidence.document_number,
        "article_number": evidence.article_number,
        "clause_number": evidence.clause_number,
        "point_number": evidence.point_number,
    }


def _capture(golden: GoldenCase, result: Any, latency_ms: float, error: str | None = None) -> dict[str, Any]:
    generation = result.generation
    citations = list(generation.citations_used) if generation else []
    evidence_by_id = {evidence.evidence_id: evidence for evidence in result.evidence}
    return {
        "id": golden.id,
        "actual_route": _actual_route(result),
        "actual_action": _actual_action(result.status) if error is None else "error",
        "pipeline_status": result.status if error is None else "error",
        "retrieved_chunk_ids": [item.chunk_id for item in result.retrieval_results],
        "reranked_chunk_ids": [item.chunk_id for item in result.reranked_results],
        "citation_ids": citations,
        "evidence_ids": [item.evidence_id for item in result.evidence],
        "cited_legal_references": [_legal_reference(evidence_by_id[item]) for item in citations if item in evidence_by_id],
        "answer": result.answer,
        "retrieval_context": [item.content for item in result.evidence],
        "latency_ms": round(latency_ms, 2),
        "error": error,
        "metadata": {
            "route_component": "guardrail_input_decision",
            "pipeline_diagnostics": result.diagnostics,
        },
    }


def _merge(golden: GoldenCase, captured: dict[str, Any]) -> EvaluationCase:
    return EvaluationCase.from_dict({
        "id": golden.id,
        "query": golden.query,
        "expected_route": golden.expected_route,
        "actual_route": captured["actual_route"],
        "expected_action": golden.expected_action,
        "actual_action": captured["actual_action"],
        "relevant_chunk_ids": list(golden.relevant_chunk_ids),
        "retrieved_chunk_ids": captured["retrieved_chunk_ids"],
        "reranked_chunk_ids": captured["reranked_chunk_ids"],
        "citation_ids": captured["citation_ids"],
        "evidence_ids": captured["evidence_ids"],
        "answer": captured["answer"],
        "reference_answer": golden.reference_answer,
        "retrieval_context": captured["retrieval_context"],
        "latency_ms": captured["latency_ms"],
        "error": captured["error"],
        "tags": list(golden.tags),
    })


def main() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(".env"), override=False)
    except ImportError:
        pass
    parser = argparse.ArgumentParser(description="Run the RAG pipeline against labelled golden cases.")
    parser.add_argument("--golden", default="evals/datasets/golden_v2.jsonl")
    parser.add_argument("--config", default="configs/labor_rag.yaml")
    parser.add_argument("--split", choices=["all", "dev", "heldout"], default="all")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N cases for a smoke test.")
    parser.add_argument("--no-generate", action="store_true", help="Evaluate retrieval/rerank only; do not call Ollama.")
    parser.add_argument("--run-output", default=None)
    parser.add_argument("--report-output", default=None)
    args = parser.parse_args()

    golden_cases = load_golden_cases(args.golden)
    if args.split != "all":
        golden_cases = [case for case in golden_cases if case.metadata.get("split") == args.split]
    if args.limit is not None:
        golden_cases = golden_cases[:args.limit]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_output = Path(args.run_output or f"evals/runs/golden_{timestamp}.jsonl")
    report_output = Path(args.report_output or f"evals/reports/golden_{timestamp}.json")

    from labor_rag.runtime.config import RAGRuntimeConfig
    from labor_rag.runtime.model_manager import ModelManager

    manager = ModelManager(RAGRuntimeConfig.from_yaml(args.config))
    manager.startup()
    captured: list[dict[str, Any]] = []
    try:
        for index, golden in enumerate(golden_cases, start=1):
            print(f"[{index}/{len(golden_cases)}] {golden.id}")
            started = perf_counter()
            try:
                result = manager.pipeline.run(golden.query, generate=not args.no_generate)
                captured.append(_capture(golden, result, (perf_counter() - started) * 1000))
            except Exception as exc:  # Capture failures as reliability data, then continue the benchmark.
                captured.append({
                    "id": golden.id, "actual_route": None, "actual_action": "error",
                    "pipeline_status": "error", "retrieved_chunk_ids": [], "reranked_chunk_ids": [],
                    "citation_ids": [], "evidence_ids": [], "cited_legal_references": [], "answer": None,
                    "retrieval_context": [], "latency_ms": round((perf_counter() - started) * 1000, 2),
                    "error": str(exc), "metadata": {"route_component": "guardrail_input_decision_unavailable_due_to_error"},
                })
    finally:
        manager.shutdown()

    run_output.parent.mkdir(parents=True, exist_ok=True)
    run_output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in captured) + "\n", encoding="utf-8")
    by_id = {row["id"]: row for row in captured}
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "golden_dataset": str(args.golden), "run_output": str(run_output),
        "pipeline_notes": ["Routing is derived from the enabled input guardrail. Routes are unavailable only for failed pipeline calls."],
        **build_report([_merge(golden, by_id[golden.id]) for golden in golden_cases]),
    }
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nCaptured run: {run_output}\nReport: {report_output}")


if __name__ == "__main__":
    main()
