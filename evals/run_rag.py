"""Phase 2: run retrieval -> rerank -> context -> generation on answerable cases.

Routing was scored in phase 1 (``evals.run_router_v2``), so the input guardrail
is skipped here: no paid classifier calls, and RAG quality is measured on its
own.  Evidence/output guardrails still run (they decide abstentions).

    python -m evals.run_rag --split dev
    python -m evals.run_rag --split heldout --config configs/labor_rag_qwen3.yaml
    python -m evals.run_rag --no-generate                 # retrieval/rerank/context only
    python -m evals.run_rag --with-abstain-probes         # + unknown-corpus cases that must abstain
    # Kaggle (embedded Qdrant, GPU):
    python -m evals.run_rag --qdrant-path /kaggle/working/qdrant --device cuda

The per-case JSONL stores retrieved/reranked IDs with scores, selected context,
answer and citations, so ``evals.sweep_context`` and ``evals.run_ragas`` can
re-score it later without re-running models.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

from . import tracing
from .rag_metrics import actual_action, aggregate, case_metrics


def select_cases(path: str | Path, *, split: str, with_probes: bool, categories: list[str] | None) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = []
    for row in rows:
        if split != "all" and row.get("split") != split:
            continue
        if categories and row["category"] not in categories:
            continue
        answerable = row["expected_action"] == "answer" and row["expected_route"] == "labor_law"
        probe = with_probes and row["expected_route"] == "unknown"
        if answerable or probe:
            selected.append(row)
    return selected


def apply_overrides(config, args):
    retrieval, reranker, generation = config.retrieval, config.reranker, config.generation
    if args.qdrant_path:
        retrieval = dataclasses.replace(retrieval, qdrant_path=Path(args.qdrant_path))
    if args.qdrant_url:
        retrieval = dataclasses.replace(retrieval, qdrant_url=args.qdrant_url)
    if args.device:
        retrieval = dataclasses.replace(retrieval, device=args.device)
        reranker = dataclasses.replace(reranker, device=args.device)
    if args.ollama_url:
        generation = dataclasses.replace(generation, ollama_url=args.ollama_url)
    return dataclasses.replace(config, retrieval=retrieval, reranker=reranker, generation=generation)


def _legal_reference(evidence: Any) -> dict[str, Any]:
    return {
        "document_id": evidence.document_id, "document_number": evidence.document_number,
        "article_number": evidence.article_number, "clause_number": evidence.clause_number,
        "point_number": getattr(evidence, "point_number", None),
    }


def capture(case: dict, result: Any, latency_ms: float, *, generate: bool) -> dict[str, Any]:
    generation = result.generation
    citations = list(generation.citations_used) if generation else []
    by_id = {item.evidence_id: item for item in result.evidence}
    diagnostics = result.diagnostics or {}
    timings = dict(diagnostics.get("pipeline", {}).get("timings_ms", {}))
    usage: dict[str, Any] = {}
    if generation is not None:
        timings["generation"] = generation.latency_ms
        raw = dict(generation.usage or {})
        usage = {"prompt": raw.get("prompt_eval_count", raw.get("prompt_tokens")),
                 "completion": raw.get("eval_count", raw.get("completion_tokens"))}
    answered_status = result.status
    return {
        "id": case["id"], "category": case["category"], "split": case.get("split"),
        "query": case["query"], "status": answered_status,
        "actual_action": actual_action(answered_status, generate=generate), "generate": generate,
        "retrieved": [{"chunk_id": r.chunk_id, "score": round(float(r.score), 6)} for r in result.retrieval_results],
        "reranked": [{"chunk_id": r.chunk_id, "score": round(float(r.score), 6)} for r in result.reranked_results],
        "context_chunk_ids": [item.chunk_id for item in result.evidence],
        "contexts": [item.content for item in result.evidence],
        "answer": result.answer if generation is not None else None,
        "citation_ids": citations,
        "cited_legal_references": [_legal_reference(by_id[c]) for c in citations if c in by_id],
        "citation_warnings": list(generation.citation_warnings) if generation else [],
        "timings_ms": timings, "usage": usage, "latency_ms": round(latency_ms, 2), "error": None,
        "context_diagnostics": {k: diagnostics.get("context", {}).get(k) for k in ("top_score", "lowest_selected_score", "selected_count", "rejected_count", "estimated_context_tokens")},
    }


def error_row(case: dict, exc: Exception, latency_ms: float, *, generate: bool) -> dict[str, Any]:
    return {
        "id": case["id"], "category": case["category"], "split": case.get("split"), "query": case["query"],
        "status": "error", "actual_action": "error", "generate": generate, "retrieved": [], "reranked": [],
        "context_chunk_ids": [], "contexts": [], "answer": None, "citation_ids": [], "cited_legal_references": [],
        "citation_warnings": [], "timings_ms": {}, "usage": {}, "latency_ms": round(latency_ms, 2),
        "error": f"{type(exc).__name__}: {exc}",
    }


def print_report(report: dict[str, Any]) -> None:
    def line(title: str, block: dict[str, Any] | None) -> None:
        if block:
            print(f"{title:12} " + "  ".join(f"{k}={v}" for k, v in block.items() if not isinstance(v, dict)))

    print(f"cases={report['case_count']} answerable={report['answerable_count']} probes={report['abstain_probe_count']}")
    for key in ("retrieval", "rerank", "context", "answer", "citation", "abstain", "performance", "failure_stage"):
        line(key, report.get(key))
    print(f"{'category':26} {'n':>3} {'hit@5':>6} {'rr@1':>6} {'ctx':>6} {'ans':>6} {'cit_p':>6}")
    for cat, m in report.get("by_category", {}).items():
        if "context_hit" in m:
            print(f"{cat:26} {m['n']:>3} {m['retrieval_hit@5']!s:>6} {m['rerank_hit@1']!s:>6} {m['context_hit']!s:>6} {m['answer_rate']!s:>6} {m['citation_precision']!s:>6}")
        else:
            print(f"{cat:26} {m['n']:>3} correct_abstain_rate={m['correct_abstain_rate']}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(".env"), override=False)
    except ImportError:
        pass
    parser = argparse.ArgumentParser(description="Phase 2: RAG evaluation on answerable golden_v2 cases.")
    parser.add_argument("--golden", default="evals/datasets/golden_v2.jsonl")
    parser.add_argument("--config", default="configs/labor_rag.yaml")
    parser.add_argument("--split", choices=["all", "dev", "heldout"], default="dev")
    parser.add_argument("--categories", nargs="*", default=None, help="Restrict to these categories.")
    parser.add_argument("--with-abstain-probes", action="store_true", help="Also run unknown-route cases (must abstain).")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-generate", action="store_true", help="Stop after context selection (no LLM).")
    parser.add_argument("--keep-input-guardrail", action="store_true", help="Run the input guardrail too (costs API calls).")
    parser.add_argument("--qdrant-path", default=None, help="Embedded Qdrant directory (Kaggle).")
    parser.add_argument("--qdrant-url", default=None)
    parser.add_argument("--device", default=None, help="cuda / cpu for embedder and reranker.")
    parser.add_argument("--ollama-url", default=None)
    parser.add_argument("--label", default=None, help="Free-text label stored with the run (e.g. 'bge-floor0.3').")
    args = parser.parse_args()

    from labor_rag.runtime.config import RAGRuntimeConfig
    from labor_rag.runtime.model_manager import ModelManager

    config = apply_overrides(RAGRuntimeConfig.from_yaml(args.config), args)
    cases = select_cases(args.golden, split=args.split, with_probes=args.with_abstain_probes, categories=args.categories)[: args.limit]
    golden = {row["id"]: row for row in cases}
    generate = not args.no_generate
    meta = tracing.run_metadata(
        "rag", config_path=args.config, dataset=args.golden, split=args.split, label=args.label,
        case_count=len(cases), generate=generate,
        retrieval_method=config.retrieval.method, embedding_model=config.retrieval.embedding_model,
        reranker_type=config.reranker.type if config.reranker.enabled else None, reranker_model=config.reranker.model,
        context_min_score=config.context.min_score, context_relative_to_top=config.context.relative_to_top,
        context_max_chunks=config.context.max_chunks, query_understanding=config.query_understanding.enabled,
        generation_provider=config.generation.provider if generate else None, generation_model=config.generation.model if generate else None,
    )
    tracing.configure()

    manager = ModelManager(config)
    manager.startup()
    rows: list[dict[str, Any]] = []
    try:
        with tracing.run_span(meta) as run_handle:
            for index, case in enumerate(cases, start=1):
                with tracing.case_span(meta, case) as handle:
                    started = perf_counter()
                    try:
                        result = manager.pipeline.run(case["query"], generate=generate, skip_input_guardrail=not args.keep_input_guardrail)
                        row = capture(case, result, (perf_counter() - started) * 1000, generate=generate)
                    except Exception as exc:  # noqa: BLE001 - failures are reliability data
                        row = error_row(case, exc, (perf_counter() - started) * 1000, generate=generate)
                        handle.record_exception(exc)
                    rows.append(row)
                    facts = case_metrics(row, case)
                    tracing.set_outcome(handle, status=row["status"], actual_action=row["actual_action"],
                                        latency_ms=row["latency_ms"], error=row["error"],
                                        answer=row["answer"] if tracing.logfire_spans.content_allowed() else None,
                                        **{k: v for k, v in facts.items() if not isinstance(v, (list, dict))})
                print(f"[{index}/{len(cases)}] {case['id']} {row['status']} {facts.get('failure_stage', '')}")
            report = {"run": meta, **aggregate(rows, golden)}
            tracing.log_summary(meta, {k: v for k, v in report.items() if k not in {"run", "by_category"}}, run_handle)
    finally:
        manager.shutdown()
        tracing.flush()

    run_out = Path(f"evals/runs/{meta['run_id']}.jsonl")
    run_out.parent.mkdir(parents=True, exist_ok=True)
    run_out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    report_out = Path(f"evals/reports/{meta['run_id']}.json")
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print_report(report)
    print(f"Per-case: {run_out}\nReport:   {report_out}")


if __name__ == "__main__":
    main()
