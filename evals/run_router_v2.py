"""Phase 1: evaluate the input guardrail router only (no retrieval, no generation).

The router may call a paid API (``openrouter_jev``), so this phase is run on its
own and its decisions are cached: re-running with the same classifier settings
only pays for queries it has not seen.

    python -m evals.run_router_v2 --split dev --show-errors
    python -m evals.run_router_v2 --split heldout --no-cache

Metrics: leak_rate, false_block_rate, route/action accuracy + macro-F1, accuracy
per category, p95 latency and error rate of the classifier.
Outputs: evals/runs/<run_id>.jsonl (per case) and evals/reports/<run_id>.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any

from . import tracing
from .deterministic_metrics import _percentile, classification_report

# Guardrail decision.action -> evaluation action label.
ACTION_MAP = {"continue": "answer", "abstain": "abstain", "refuse": "refuse", "clarify": "clarify"}
# Guardrail decision.category -> evaluation route label.
ROUTE_MAP = {
    "out_of_scope": "out_of_scope",
    "sensitive_or_disallowed": "sensitive",
    "unknown": "unknown",
}
BLOCKING_ACTIONS = {"refuse", "abstain", "clarify"}


def load_cases(path: str | Path, split: str = "all") -> list[dict]:
    rows = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        for key in ("id", "query", "category", "expected_route", "expected_action"):
            if not raw.get(key):
                raise ValueError(f"{path}:{line_number} missing '{key}'")
        if split == "all" or raw.get("split") == split:
            rows.append(raw)
    if not rows:
        raise ValueError(f"No cases for split={split!r} in {path}")
    return rows


def actual_labels(decision) -> tuple[str, str]:
    route = ROUTE_MAP.get(decision.category, "labor_law")
    action = ACTION_MAP.get(decision.action, decision.action)
    return route, action


def summarize(rows: list[dict]) -> dict:
    """Router metrics. Rows with ``actual_action == "error"`` count as failures."""
    by_category: dict[str, dict] = {}
    groups = defaultdict(list)
    for row in rows:
        groups[row["category"]].append(row)
    for category, items in sorted(groups.items()):
        n = len(items)
        by_category[category] = {
            "count": n,
            "route_accuracy": round(sum(r["expected_route"] == r["actual_route"] for r in items) / n, 4),
            "action_accuracy": round(sum(r["expected_action"] == r["actual_action"] for r in items) / n, 4),
            "route_and_action_accuracy": round(sum(_correct(r) for r in items) / n, 4),
            "actual_action_counts": dict(Counter(r["actual_action"] for r in items)),
        }
    should_answer = [r for r in rows if r["expected_action"] == "answer"]
    should_block = [r for r in rows if r["expected_action"] in BLOCKING_ACTIONS]
    latencies = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None and not r.get("cached")]
    return {
        # Answerable questions the router did not let through (over-blocking).
        "false_block_rate": _rate(r["actual_action"] != "answer" for r in should_answer),
        # Questions that should have been refused/abstained/clarified but were let through.
        "leak_rate": _rate(r["actual_action"] == "answer" for r in should_block),
        # Blocked correctly *and* with the right action (refuse vs abstain vs clarify).
        "correct_block_action_rate": _rate(r["actual_action"] == r["expected_action"] for r in should_block),
        "route_and_action_accuracy": _rate(_correct(r) for r in rows),
        "error_rate": _rate(r["actual_action"] == "error" for r in rows),
        "p50_latency_ms": _percentile(latencies, 0.50),
        "p95_latency_ms": _percentile(latencies, 0.95),
        "measured_latency_count": len(latencies),
        "by_category": by_category,
    }


def _correct(row: dict) -> bool:
    return row["expected_route"] == row["actual_route"] and row["expected_action"] == row["actual_action"]


def _rate(flags) -> float | None:
    flags = list(flags)
    return round(sum(flags) / len(flags), 4) if flags else None


# ---------------------------------------------------------------- cache
def _cache_key(classifier_fingerprint: str, query: str) -> str:
    return hashlib.sha256(f"{classifier_fingerprint}\n{query}".encode("utf-8")).hexdigest()


def _load_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    cache = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            cache[item["key"]] = item
    return cache


def _classify(service, query: str) -> dict[str, Any]:
    started = perf_counter()
    try:
        decision = service.check_input(query)
    except Exception as exc:  # noqa: BLE001 - a failing classifier is a measured outcome
        return {"error": f"{type(exc).__name__}: {exc}", "latency_ms": round((perf_counter() - started) * 1000, 2)}
    route, action = actual_labels(decision)
    return {
        "actual_route": route, "actual_action": action,
        "guardrail_action": decision.action, "guardrail_category": decision.category,
        "confidence": decision.confidence, "matched_rules": decision.metadata.get("matched_rules", []),
        "latency_ms": round((perf_counter() - started) * 1000, 2), "error": None,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(".env"), override=False)
    except ImportError:
        pass
    parser = argparse.ArgumentParser(description="Phase 1: router evaluation on golden_v2.")
    parser.add_argument("--cases", default="evals/datasets/golden_v2.jsonl")
    parser.add_argument("--config", default="configs/labor_rag.yaml")
    parser.add_argument("--split", choices=["all", "dev", "heldout"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--cache", default="evals/cache/router_decisions.jsonl", help="Decision cache (JSONL).")
    parser.add_argument("--no-cache", action="store_true", help="Ignore cached decisions and call the classifier again.")
    parser.add_argument("--output", default=None)
    parser.add_argument("--show-errors", action="store_true", help="Print misrouted queries.")
    args = parser.parse_args()

    from labor_rag.runtime.config import RAGRuntimeConfig
    from labor_rag.runtime.factory import build_guardrail_service

    config = RAGRuntimeConfig.from_yaml(args.config)
    service = build_guardrail_service(config)
    cases = load_cases(args.cases, args.split)[: args.limit]
    clf = config.guardrails.classifier
    fingerprint = json.dumps({
        "provider": clf.provider, "model": clf.model, "threshold": clf.threshold,
        "guardrails": {k: getattr(config.guardrails, k) for k in ("scope_enabled", "sensitive_enabled", "rules_enabled", "ambiguity_enabled", "ambiguity_mode")},
    }, sort_keys=True)

    cache_path = Path(args.cache)
    # The offline rules classifier is free and its code changes often, so its
    # decisions are never cached (a stale cache would hide rule edits).
    use_cache = not args.no_cache and clf.provider != "rules"
    cache = _load_cache(cache_path) if use_cache else {}
    meta = tracing.run_metadata(
        "router", config_path=args.config, dataset=args.cases, split=args.split,
        classifier=clf.provider, classifier_model=clf.model, classifier_threshold=clf.threshold,
    )
    tracing.configure()

    rows: list[dict] = []
    new_cache: list[dict] = []
    with tracing.run_span(meta) as run_handle:
        for index, case in enumerate(cases, start=1):
            key = _cache_key(fingerprint, case["query"])
            with tracing.case_span(meta, case) as handle:
                cached = cache.get(key)
                if cached and not cached.get("error"):
                    result, from_cache = dict(cached["result"]), True
                else:
                    result, from_cache = _classify(service, case["query"]), False
                    if use_cache and not result.get("error"):
                        new_cache.append({"key": key, "query": case["query"], "result": result})
                if result.get("error"):
                    result.update({"actual_route": None, "actual_action": "error"})
                row = {
                    "id": case["id"], "query": case["query"], "category": case["category"],
                    "subcategory": case.get("subcategory"), "split": case.get("split"),
                    "expected_route": case["expected_route"], "expected_action": case["expected_action"],
                    **result, "cached": from_cache,
                }
                row["correct"] = _correct(row)
                rows.append(row)
                tracing.set_outcome(
                    handle, actual_route=row["actual_route"], actual_action=row["actual_action"],
                    correct=row["correct"], leaked=row["expected_action"] in BLOCKING_ACTIONS and row["actual_action"] == "answer",
                    false_block=row["expected_action"] == "answer" and row["actual_action"] != "answer",
                    confidence=row.get("confidence"), cached=from_cache, error=row.get("error"),
                )
            if index % 25 == 0:
                print(f"[{index}/{len(cases)}]")

        ns = [SimpleNamespace(**row) for row in rows]
        summary = summarize(rows)
        report = {
            "run": meta,
            "dataset": args.cases, "split": args.split, "case_count": len(rows),
            "cache": {"hits": sum(r["cached"] for r in rows), "new_calls": sum(not r["cached"] for r in rows)},
            "route": classification_report(ns, "expected_route", "actual_route"),
            "action": classification_report(ns, "expected_action", "actual_action"),
            "summary": summary,
            "errors": [r for r in rows if not r["correct"]],
        }
        headline = {k: v for k, v in summary.items() if k != "by_category"}
        headline.update({"route_accuracy": report["route"]["accuracy"], "route_macro_f1": report["route"]["macro_f1"],
                         "action_accuracy": report["action"]["accuracy"], "action_macro_f1": report["action"]["macro_f1"]})
        tracing.log_summary(meta, {"router": headline, "by_category": {k: {"accuracy": v["route_and_action_accuracy"]} for k, v in summary["by_category"].items()}}, run_handle)
    tracing.flush()

    if new_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("a", encoding="utf-8") as stream:
            for item in new_cache:
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")

    run_out = Path(f"evals/runs/{meta['run_id']}.jsonl")
    run_out.parent.mkdir(parents=True, exist_ok=True)
    run_out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    out = Path(args.output or f"evals/reports/{meta['run_id']}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"run_id={meta['run_id']} cases={len(rows)} split={args.split} classifier={clf.provider} "
          f"cache_hits={report['cache']['hits']} new_calls={report['cache']['new_calls']}")
    print(f"leak_rate={summary['leak_rate']}  false_block_rate={summary['false_block_rate']}  "
          f"correct_block_action_rate={summary['correct_block_action_rate']}")
    print(f"route accuracy={report['route']['accuracy']} macro_f1={report['route']['macro_f1']} | "
          f"action accuracy={report['action']['accuracy']} macro_f1={report['action']['macro_f1']}")
    print(f"error_rate={summary['error_rate']}  p95_latency_ms={summary['p95_latency_ms']}")
    print(f"{'category':26} {'n':>3} {'route':>6} {'action':>7} {'both':>6}")
    for cat, s in summary["by_category"].items():
        print(f"{cat:26} {s['count']:>3} {s['route_accuracy']:>6} {s['action_accuracy']:>7} {s['route_and_action_accuracy']:>6}")
    if args.show_errors:
        for r in report["errors"]:
            print(f"[{r['category']}] {r['query']}  exp={r['expected_route']}/{r['expected_action']} got={r['actual_route']}/{r['actual_action']}")
    print(f"Per-case: {run_out}\nReport:   {out}")


if __name__ == "__main__":
    main()
