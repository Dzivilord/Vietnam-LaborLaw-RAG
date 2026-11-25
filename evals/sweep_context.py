"""Calibrate context-selection thresholds offline from a saved phase-2 run.

``evals.run_rag`` stores every reranked candidate with its score, so the real
``RerankedContextBuilder`` can be replayed over a grid of ``min_score`` x
``relative_to_top`` without loading any model.  Calibrate on the dev split,
then confirm the chosen pair once on heldout.

    python -m evals.sweep_context --run evals/runs/rag-<ts>.jsonl
    python -m evals.sweep_context --run ... --min-scores 0.1 0.2 0.3 --relatives none 0.5 0.6 0.7

Per pair it reports, for answerable cases: context_hit, context_recall,
false_abstain (nothing selected -> pipeline abstains), mean chunks; and for
abstain probes (run with --with-abstain-probes): correct_abstain.
The ambiguity check that may follow context selection is not simulated.

Scores are reranker-specific: sweep BGE and Qwen3 runs separately.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import quantiles
from typing import Any


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def replay(rows, golden, chunks, *, min_score: float, relative: float | None, max_chunks: int, max_tokens: int | None) -> dict[str, Any]:
    from labor_rag.reranking.context_builder import ContextBuilderConfig, RerankedContextBuilder
    from labor_rag.retrieval.base import RetrievalResult

    builder = RerankedContextBuilder(ContextBuilderConfig(
        min_score=min_score, min_relative_to_top=relative, max_chunks=max_chunks, max_tokens=max_tokens,
    ))
    hits, recalls, abstains, sizes, probe_abstain = [], [], [], [], []
    for row in rows:
        gold = golden.get(row["id"])
        if gold is None or not row.get("reranked"):
            continue
        candidates = []
        for item in row["reranked"]:
            chunk = chunks.get(item["chunk_id"])
            if chunk is None:
                continue
            metadata = {k: v for k, v in chunk.items() if k not in {"content", "embedding_text"}}
            candidates.append(RetrievalResult(item["chunk_id"], chunk["content"], float(item["score"]), metadata))
        selected = {e.chunk_id for e in builder.build(row["query"], candidates).evidence} if candidates else set()
        relevant = set(gold.get("relevant_chunk_ids") or [])
        if relevant:
            hits.append(bool(relevant & selected))
            recalls.append(len(relevant & selected) / len(relevant))
            abstains.append(not selected)
            sizes.append(len(selected))
        else:
            probe_abstain.append(not selected)

    def mean(values):
        return round(sum(values) / len(values), 4) if values else None

    return {
        "min_score": min_score, "relative_to_top": relative,
        "context_hit": mean(hits), "context_recall": mean(recalls),
        "false_abstain": mean(abstains), "mean_chunks": mean(sizes),
        "correct_abstain": mean(probe_abstain), "n_answerable": len(hits), "n_probes": len(probe_abstain),
    }


def score_distribution(rows, golden) -> dict[str, Any]:
    """Top rerank score per case, split by answerable vs probe; guides the floor."""
    groups: dict[str, list[float]] = {"answerable": [], "probe": [], "gold_chunk": []}
    for row in rows:
        gold = golden.get(row["id"])
        if gold is None or not row.get("reranked"):
            continue
        relevant = set(gold.get("relevant_chunk_ids") or [])
        groups["answerable" if relevant else "probe"].append(float(row["reranked"][0]["score"]))
        groups["gold_chunk"].extend(float(i["score"]) for i in row["reranked"] if i["chunk_id"] in relevant)
    out = {}
    for name, values in groups.items():
        if len(values) >= 4:
            q = quantiles(values, n=10)
            out[name] = {"n": len(values), "p10": round(q[0], 3), "p50": round(q[4], 3), "p90": round(q[8], 3), "min": round(min(values), 3)}
        elif values:
            out[name] = {"n": len(values), "min": round(min(values), 3)}
    return out


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Offline threshold sweep for context selection.")
    parser.add_argument("--run", required=True, help="evals/runs/rag-*.jsonl from evals.run_rag")
    parser.add_argument("--golden", default="evals/datasets/golden_v2.jsonl")
    parser.add_argument("--chunks", default="data/processed/chunks/labor_law_legal.jsonl")
    parser.add_argument("--min-scores", nargs="*", type=float, default=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    parser.add_argument("--relatives", nargs="*", default=["none", "0.4", "0.5", "0.6", "0.7", "0.8"])
    parser.add_argument("--max-chunks", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=3500)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    rows = _load_jsonl(args.run)
    golden = {row["id"]: row for row in _load_jsonl(args.golden)}
    chunks = {row["chunk_id"]: row for row in _load_jsonl(args.chunks)}
    relatives = [None if r.lower() == "none" else float(r) for r in args.relatives]

    results = [
        replay(rows, golden, chunks, min_score=m, relative=r, max_chunks=args.max_chunks, max_tokens=args.max_tokens)
        for m in args.min_scores for r in relatives
    ]
    dist = score_distribution(rows, golden)
    print("Top-1 / gold-chunk rerank score distribution:")
    for name, d in dist.items():
        print(f"  {name:11} {d}")
    print(f"\n{'min':>5} {'rel':>5} | {'ctx_hit':>7} {'ctx_rec':>7} {'f_abst':>7} {'chunks':>6} {'c_abst':>7}")
    for r in sorted(results, key=lambda x: (-(x['context_hit'] or 0), x['mean_chunks'] or 0)):
        rel = "-" if r["relative_to_top"] is None else r["relative_to_top"]
        print(f"{r['min_score']:>5} {rel!s:>5} | {r['context_hit']!s:>7} {r['context_recall']!s:>7} {r['false_abstain']!s:>7} {r['mean_chunks']!s:>6} {r['correct_abstain']!s:>7}")
    print("\nPick the pair with high ctx_hit and correct_abstain, low f_abst, and as few chunks as possible.")
    out = Path(args.output or f"evals/reports/{Path(args.run).stem}_sweep.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"run": args.run, "score_distribution": dist, "grid": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
