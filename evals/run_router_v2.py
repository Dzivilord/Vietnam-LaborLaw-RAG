"""Phase 1: evaluate the input guardrail router only (no retrieval, no generation).

The router may call a paid API (``openrouter_jev``), so this phase is run on its
own and its decisions are cached: re-running with the same classifier settings
only pays for queries it has not seen.

    python -m evals.run_router_v2 --split dev --show-errors
    python -m evals.run_router_v2 --split heldout --no-cache

Router goal: *must-block* off-topic (out_of_scope) and sensitive queries, never
block answerable ones. Ambiguous and unknown-corpus queries are *soft*: blocking
them correctly is a bonus, letting them through is acceptable because retrieval
and the evidence gate handle them downstream.

Headline metrics: critical_leak_rate (must-block answered), false_block_rate,
acceptable_rate. Also reported: soft_pass_rate, strict accuracy/macro-F1 per
route and action, accuracy per category, p95 latency, error rate.
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
# Routes the router itself must stop. Everything else expected to be blocked
# (ambiguous -> clarify, unknown -> abstain) is "soft" and may pass through.
MUST_BLOCK_ROUTES = {"out_of_scope", "sensitive"}


def is_must_block(row: dict) -> bool:
    return row["expected_route"] in MUST_BLOCK_ROUTES


def is_soft(row: dict) -> bool:
    return row["expected_action"] in BLOCKING_ACTIONS and not is_must_block(row)


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
    if decision.metadata.get("clarify_after_retrieval"):
        # The router already decided "clarify"; it is only executed after retrieval.
        action = "clarify"
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
            "acceptable_rate": round(sum(_acceptable(r) for r in items) / n, 4),
            "actual_action_counts": dict(Counter(r["actual_action"] for r in items)),
        }
    # Leak/false-block are measured on cases the classifier actually decided;
    # failed calls are reported separately as error_rate.
    decided = [r for r in rows if r["actual_action"] != "error"]
    should_answer = [r for r in decided if r["expected_action"] == "answer"]
    should_block = [r for r in decided if r["expected_action"] in BLOCKING_ACTIONS]
    must_block = [r for r in decided if is_must_block(r)]
    soft = [r for r in decided if is_soft(r)]
    latencies = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None and not r.get("cached")]
    return {
        # Off-topic / sensitive queries that were answered: the failure that matters.
        "critical_leak_rate": _rate(r["actual_action"] == "answer" for r in must_block),
        # Answerable questions the router did not let through (over-blocking).
        "false_block_rate": _rate(r["actual_action"] != "answer" for r in should_answer),
        # Correct, or a soft case that was let through.
        "acceptable_rate": _rate(_acceptable(r) for r in rows),
        # Ambiguous / unknown-corpus queries passed on to retrieval (informational).
        "soft_pass_rate": _rate(r["actual_action"] == "answer" for r in soft),
        # Strict: every expected block (incl. soft) that was let through.
        "leak_rate_all": _rate(r["actual_action"] == "answer" for r in should_block),
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


def _acceptable(row: dict) -> bool:
    """Strictly correct, or a soft case that the router let through."""
    return _correct(row) or (is_soft(row) and row["actual_action"] == "answer")


def _rate(flags) -> float | None:
    flags = list(flags)
    return round(sum(flags) / len(flags), 4) if flags else None


MAX_CONSECUTIVE_ERRORS = 5
# "cannot be relayed": OpenRouter management/provisioning keys cannot call model endpoints.
FATAL_MARKERS = ("HTTP 401", "HTTP 403", "Missing OPENROUTER_API_KEY", "cannot be relayed")


def _is_fatal(error: str) -> bool:
    """Auth/config errors will fail every remaining case, so stop at once."""
    return any(marker in error for marker in FATAL_MARKERS)


# ---------------------------------------------------------------- cache
# Only the *classifier call* is cached (label, confidence, probabilities) - the
# paid part. Policy (thresholds, ambiguity/unknown modes, rules) is re-applied
# on every run, so tuning it never triggers new API calls.
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


class CachingClassifier:
    """Wraps the semantic classifier; serves repeated queries from the cache."""

    def __init__(self, inner, fingerprint: str, cache: dict[str, dict]) -> None:
        self.inner, self.fingerprint, self.cache = inner, fingerprint, cache
        self.new_entries: list[dict] = []
        self.last_hit: bool | None = None  # None: classifier not consulted (a rule decided)

    def classify(self, query: str):
        from labor_rag.guardrails.schemas import QueryClassification

        key = _cache_key(self.fingerprint, query)
        item = self.cache.get(key)
        if item is not None:
            self.last_hit = True
            data = dict(item["classification"])
            data["matched_rules"] = tuple(data.get("matched_rules", ()))
            return QueryClassification(**data)
        self.last_hit = False
        result = self.inner.classify(query)
        entry = {"key": key, "query": query, "classification": {
            "label": result.label, "confidence": result.confidence, "scores": dict(result.scores),
            "source": result.source, "matched_rules": list(result.matched_rules),
        }}
        self.cache[key] = entry
        self.new_entries.append(entry)
        return result


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
        "confidence": decision.confidence, "scores": decision.metadata.get("scores", {}),
        "matched_rules": decision.metadata.get("matched_rules", []),
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
    parser.add_argument("--cache", default="evals/cache/router_classifications.jsonl", help="Classifier-call cache (JSONL).")
    parser.add_argument("--no-cache", action="store_true", help="Ignore cached classifications and call the classifier again.")
    parser.add_argument("--output", default=None)
    parser.add_argument("--show-errors", action="store_true", help="Print misrouted queries.")
    args = parser.parse_args()

    from labor_rag.runtime.config import RAGRuntimeConfig
    from labor_rag.runtime.factory import build_guardrail_service

    config = RAGRuntimeConfig.from_yaml(args.config)
    service = build_guardrail_service(config)
    cases = load_cases(args.cases, args.split)[: args.limit]
    clf = config.guardrails.classifier
    # Cache identity = what produces the classification, not the policy around it.
    # The classifier's own fingerprint covers its prompt/criteria, so editing them
    # invalidates the cache instead of silently reusing old labels.
    fingerprint = json.dumps({
        "provider": clf.provider, "model": clf.model, "base_url": clf.base_url,
        "prompt": getattr(service.classifier, "fingerprint", None),
    }, sort_keys=True)

    cache_path = Path(args.cache)
    # The offline rules classifier is free, so it is never cached.
    use_cache = not args.no_cache and clf.provider != "rules"
    caching = None
    if use_cache:
        caching = CachingClassifier(service.classifier, fingerprint, _load_cache(cache_path))
        service.classifier = caching
    meta = tracing.run_metadata(
        "router", config_path=args.config, dataset=args.cases, split=args.split,
        classifier=clf.provider, classifier_model=clf.model, classifier_threshold=clf.threshold,
        label_thresholds=dict(clf.label_thresholds), ambiguity_mode=config.guardrails.ambiguity_mode,
        unknown_action=config.guardrails.unknown_action,
    )
    tracing.configure()

    rows: list[dict] = []
    consecutive_errors = 0
    with tracing.run_span(meta) as run_handle:
        for index, case in enumerate(cases, start=1):
            with tracing.case_span(meta, case) as handle:
                if caching is not None:
                    caching.last_hit = None
                result = _classify(service, case["query"])
                # Paid call avoided when served from cache or decided by a rule.
                from_cache = bool(caching and caching.last_hit is not False)
                if result.get("error"):
                    # "error" (not None) so accuracy counts failed calls as wrong.
                    result.update({"actual_route": "error", "actual_action": "error"})
                    consecutive_errors += 1
                    if _is_fatal(result["error"]) or consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        raise SystemExit(
                            f"Aborting after {index}/{len(cases)} cases: classifier keeps failing "
                            f"({consecutive_errors} in a row). Last error: {result['error'][:300]}\n"
                            "Nothing was written; fix the classifier (API key, URL, quota) and re-run."
                        )
                else:
                    consecutive_errors = 0
                row = {
                    "id": case["id"], "query": case["query"], "category": case["category"],
                    "subcategory": case.get("subcategory"), "split": case.get("split"),
                    "expected_route": case["expected_route"], "expected_action": case["expected_action"],
                    **result, "cached": from_cache,
                }
                row["correct"] = _correct(row)
                row["acceptable"] = _acceptable(row)
                rows.append(row)
                tracing.set_outcome(
                    handle, actual_route=row["actual_route"], actual_action=row["actual_action"],
                    correct=row["correct"], acceptable=row["acceptable"],
                    critical_leak=is_must_block(row) and row["actual_action"] == "answer",
                    leaked=row["expected_action"] in BLOCKING_ACTIONS and row["actual_action"] == "answer",
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
            "errors": [r for r in rows if not r["acceptable"]],
            "soft_passes": [r for r in rows if r["acceptable"] and not r["correct"]],
        }
        headline = {k: v for k, v in summary.items() if k != "by_category"}
        headline.update({"route_accuracy": report["route"]["accuracy"], "route_macro_f1": report["route"]["macro_f1"],
                         "action_accuracy": report["action"]["accuracy"], "action_macro_f1": report["action"]["macro_f1"]})
        tracing.log_summary(meta, {"router": headline, "by_category": {k: {"accuracy": v["route_and_action_accuracy"]} for k, v in summary["by_category"].items()}}, run_handle)
    tracing.flush()

    if caching is not None and caching.new_entries:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("a", encoding="utf-8") as stream:
            for item in caching.new_entries:
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")

    run_out = Path(f"evals/runs/{meta['run_id']}.jsonl")
    run_out.parent.mkdir(parents=True, exist_ok=True)
    run_out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    out = Path(args.output or f"evals/reports/{meta['run_id']}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"config={args.config} run_id={meta['run_id']} cases={len(rows)} split={args.split} classifier={clf.provider} "
          f"cache_hits={report['cache']['hits']} new_calls={report['cache']['new_calls']}")
    print(f"critical_leak_rate={summary['critical_leak_rate']}  false_block_rate={summary['false_block_rate']}  "
          f"acceptable_rate={summary['acceptable_rate']}")
    print(f"soft_pass_rate={summary['soft_pass_rate']} (ambiguous/unknown let through - allowed)  "
          f"leak_rate_all={summary['leak_rate_all']}  correct_block_action_rate={summary['correct_block_action_rate']}")
    print(f"route accuracy={report['route']['accuracy']} macro_f1={report['route']['macro_f1']} | "
          f"action accuracy={report['action']['accuracy']} macro_f1={report['action']['macro_f1']}")
    print(f"error_rate={summary['error_rate']}  p95_latency_ms={summary['p95_latency_ms']}")
    if summary["error_rate"]:
        print(f"WARNING: {summary['error_rate']:.0%} of classifier calls failed; metrics cover only the decided cases. "
              "See the 'error' field in the per-case file.")
    print(f"{'category':26} {'n':>3} {'route':>6} {'action':>7} {'strict':>7} {'accept':>7}")
    for cat, s in summary["by_category"].items():
        print(f"{cat:26} {s['count']:>3} {s['route_accuracy']:>6} {s['action_accuracy']:>7} {s['route_and_action_accuracy']:>7} {s['acceptable_rate']:>7}")
    if args.show_errors:
        for r in report["errors"]:
            print(f"ERROR [{r['category']}] {r['query']}  exp={r['expected_route']}/{r['expected_action']} got={r['actual_route']}/{r['actual_action']}")
        for r in report["soft_passes"]:
            print(f"soft  [{r['category']}] {r['query']}  (let through, allowed)")
    print(f"Per-case: {run_out}\nReport:   {out}")


if __name__ == "__main__":
    main()
