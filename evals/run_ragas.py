"""Score a saved phase-2 run with RAGAS LLM-judge metrics (ragas>=0.4).

Runs anywhere (local or Kaggle) on the JSONL from ``evals.run_rag``; no RAG
model is loaded.  Only answered cases with a reference answer are scored.

    pip install -r requirements-eval.txt
    $env:RAGAS_JUDGE_MODEL = "openai/gpt-4.1-mini"     # any OpenAI-compatible model
    python -m evals.run_ragas --run evals/runs/rag-<ts>.jsonl --limit 20

Metrics (0..1, higher is better):
* faithfulness       - share of answer claims supported by the retrieved context (hallucination check)
* factual_correctness- claim-level F1 between answer and the golden reference answer
* context_precision  - are the relevant chunks ranked/kept ahead of irrelevant ones (uses reference)
* context_recall     - does the context contain what the reference answer needs
* answer_relevancy   - optional (--embeddings-model): does the answer address the question

Use a judge model different from the generator.  Judge calls are cached on disk
(``--cache-dir``), so re-running costs nothing for unchanged cases.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from . import tracing

DEFAULT_METRICS = ("faithfulness", "factual_correctness", "context_precision", "context_recall")
ANSWERED = {"generated", "generated_with_warnings"}


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def build_samples(rows: list[dict], golden: dict[str, dict]) -> list[dict[str, Any]]:
    samples = []
    for row in rows:
        gold = golden.get(row["id"])
        if not gold or row.get("status") not in ANSWERED or not row.get("answer") or not row.get("contexts"):
            continue
        if not gold.get("reference_answer"):
            continue
        samples.append({
            "id": row["id"], "category": gold.get("category"), "split": gold.get("split"),
            "user_input": row["query"], "response": row["answer"],
            "retrieved_contexts": list(row["contexts"]), "reference": gold["reference_answer"],
        })
    return samples


def build_metrics(names: list[str], args) -> dict[str, tuple[Any, tuple[str, ...]]]:
    from openai import AsyncOpenAI
    from ragas.cache import DiskCacheBackend
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (
        AnswerRelevancy, ContextPrecisionWithReference, ContextRecall, FactualCorrectness, Faithfulness,
    )

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Set {args.api_key_env} (judge API key).")
    if not args.judge_model:
        raise SystemExit("Set --judge-model or RAGAS_JUDGE_MODEL.")
    client = AsyncOpenAI(api_key=api_key, base_url=args.base_url)
    kwargs: dict[str, Any] = {"cache": DiskCacheBackend(cache_dir=args.cache_dir), "temperature": 0.0}
    if args.instructor_mode:
        import instructor
        kwargs["mode"] = getattr(instructor.Mode, args.instructor_mode.upper())
    llm = llm_factory(args.judge_model, client=client, **kwargs)

    # metric name -> (metric object, sample fields its ascore() takes)
    available: dict[str, Any] = {
        "faithfulness": lambda: (Faithfulness(llm=llm), ("user_input", "response", "retrieved_contexts")),
        "factual_correctness": lambda: (FactualCorrectness(llm=llm, mode="f1"), ("response", "reference")),
        "context_precision": lambda: (ContextPrecisionWithReference(llm=llm), ("user_input", "reference", "retrieved_contexts")),
        "context_recall": lambda: (ContextRecall(llm=llm), ("user_input", "retrieved_contexts", "reference")),
    }
    if args.embeddings_model:
        from ragas.embeddings import embedding_factory

        embeddings = embedding_factory("huggingface", model=args.embeddings_model)
        available["answer_relevancy"] = lambda: (AnswerRelevancy(llm=llm, embeddings=embeddings), ("user_input", "response"))
    unknown = [n for n in names if n not in available]
    if unknown:
        raise SystemExit(f"Unknown/unavailable metrics {unknown} (answer_relevancy needs --embeddings-model).")
    return {name: available[name]() for name in names}


async def score_all(samples, metrics, concurrency: int) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(concurrency)

    async def one(sample: dict[str, Any]) -> dict[str, Any]:
        out = {"id": sample["id"], "category": sample["category"], "split": sample["split"], "scores": {}, "errors": {}}
        for name, (metric, fields) in metrics.items():
            async with semaphore:
                try:
                    result = await metric.ascore(**{f: sample[f] for f in fields})
                    value = float(result.value)
                    out["scores"][name] = None if math.isnan(value) else round(value, 4)
                except Exception as exc:  # noqa: BLE001 - one failing judge call must not stop the run
                    out["scores"][name] = None
                    out["errors"][name] = f"{type(exc).__name__}: {exc}"[:500]
        return out

    return await asyncio.gather(*(one(s) for s in samples))


def summarize(results: list[dict[str, Any]], names: list[str]) -> dict[str, Any]:
    def mean(values):
        values = [v for v in values if v is not None]
        return round(sum(values) / len(values), 4) if values else None

    summary: dict[str, Any] = {"n": len(results)}
    for name in names:
        summary[name] = mean(r["scores"].get(name) for r in results)
        summary[f"{name}_errors"] = sum(name in r["errors"] for r in results)
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        groups[r["category"]].append(r)
    summary["by_category"] = {
        cat: {"n": len(items), **{name: mean(r["scores"].get(name) for r in items) for name in names}}
        for cat, items in sorted(groups.items())
    }
    return summary


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(".env"), override=False)
    except ImportError:
        pass
    parser = argparse.ArgumentParser(description="RAGAS scoring of a saved phase-2 run.")
    parser.add_argument("--run", required=True)
    parser.add_argument("--golden", default="evals/datasets/golden_v2.jsonl")
    parser.add_argument("--metrics", nargs="*", default=list(DEFAULT_METRICS))
    parser.add_argument("--judge-model", default=os.getenv("RAGAS_JUDGE_MODEL"))
    parser.add_argument("--base-url", default=os.getenv("RAGAS_BASE_URL", "https://openrouter.ai/api/v1"))
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--embeddings-model", default=None, help="HF model for answer_relevancy, e.g. BAAI/bge-m3")
    parser.add_argument("--instructor-mode", default=None, help="e.g. md_json if the judge rejects response_format")
    parser.add_argument("--cache-dir", default="evals/cache/ragas")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Score only the first N samples (cost control).")
    args = parser.parse_args()

    golden = {row["id"]: row for row in _load_jsonl(args.golden)}
    samples = build_samples(_load_jsonl(args.run), golden)[: args.limit]
    if not samples:
        raise SystemExit("No answered cases with reference answers in this run (was it run with --no-generate?).")
    metrics = build_metrics(args.metrics, args)
    meta = tracing.run_metadata("ragas", source_run=args.run, judge_model=args.judge_model,
                                metrics=list(metrics), sample_count=len(samples))
    tracing.configure()
    with tracing.run_span(meta) as run_handle:
        results = asyncio.run(score_all(samples, metrics, args.concurrency))
        for sample, result in zip(samples, results):
            with tracing.case_span(meta, {"id": sample["id"], "category": sample["category"], "split": sample["split"],
                                          "query": sample["user_input"]}) as handle:
                tracing.set_outcome(handle, **{f"ragas.{k}": v for k, v in result["scores"].items()},
                                    ragas_errors=sorted(result["errors"]) or None)
        summary = summarize(results, list(metrics))
        tracing.log_summary(meta, {"ragas": {k: v for k, v in summary.items() if k != "by_category"}}, run_handle)
    tracing.flush()

    out = Path(f"evals/reports/{meta['run_id']}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"run": meta, "summary": summary, "per_case": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "by_category"}, ensure_ascii=False, indent=2))
    for cat, m in summary["by_category"].items():
        print(f"{cat:26} " + "  ".join(f"{k}={v}" for k, v in m.items()))
    print(f"Report: {out}")


if __name__ == "__main__":
    main()
