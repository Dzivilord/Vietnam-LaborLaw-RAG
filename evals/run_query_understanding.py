"""Offline A/B of query understanding on golden_v2 using BM25 only.

Needs no Qdrant/models.  It isolates what the rewrite/lookup layer adds to the
lexical branch; dense and rerank effects must be measured with ``run_golden``.

    python -m evals.run_query_understanding --split all
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

KS = (1, 5, 10, 20)


def _metrics(rows: list[dict]) -> dict:
    n = len(rows)
    out: dict[str, float] = {"n": n}
    for k in KS:
        out[f"hit@{k}"] = round(sum(r["hit_rank"] is not None and r["hit_rank"] <= k for r in rows) / n, 4)
        out[f"recall@{k}"] = round(sum(r["found_at"][k] / r["relevant"] for r in rows) / n, 4)
    out["mrr@20"] = round(sum((1 / r["hit_rank"]) if r["hit_rank"] and r["hit_rank"] <= 20 else 0.0 for r in rows) / n, 4)
    return out


def _score(retriever, cases: list[dict], top_k: int = 20) -> list[dict]:
    rows = []
    for case in cases:
        relevant = set(case["relevant_chunk_ids"])
        ids = [r.chunk_id for r in retriever.retrieve(case["query"], top_k=top_k)]
        ranks = [i for i, cid in enumerate(ids, start=1) if cid in relevant]
        rows.append({
            "id": case["id"], "query": case["query"], "category": case["category"],
            "subcategory": case.get("subcategory"), "split": case.get("split"),
            "relevant": len(relevant), "hit_rank": ranks[0] if ranks else None,
            "found_at": {k: sum(1 for r in ranks if r <= k) for k in KS},
            "top5": ids[:5], "expected": sorted(relevant),
        })
    return rows


def _group(rows: list[dict]) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["category"]].append(row)
    result = {cat: _metrics(items) for cat, items in sorted(groups.items())}
    result["ALL"] = _metrics(rows)
    return result


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="BM25 A/B for query understanding.")
    parser.add_argument("--golden", default="evals/datasets/golden_v2.jsonl")
    parser.add_argument("--chunks", default="data/processed/chunks/labor_law_legal.jsonl")
    parser.add_argument("--split", choices=["all", "dev", "heldout"], default="all")
    parser.add_argument("--output", default="evals/reports/query_understanding_bm25.json")
    parser.add_argument("--show-misses", action="store_true")
    args = parser.parse_args()

    from labor_rag.retrieval.bm25_retriever import BM25Retriever
    from labor_rag.retrieval.query_process import QueryAwareRetriever

    chunks = [json.loads(l) for l in Path(args.chunks).read_text(encoding="utf-8").splitlines() if l.strip()]
    cases = [json.loads(l) for l in Path(args.golden).read_text(encoding="utf-8").splitlines() if l.strip()]
    cases = [c for c in cases if c.get("relevant_chunk_ids") and (args.split == "all" or c.get("split") == args.split)]

    baseline = BM25Retriever(chunks)
    improved = QueryAwareRetriever(BM25Retriever(chunks), chunks)

    base_rows = _score(baseline, cases)
    new_rows = _score(improved, cases)
    report = {"split": args.split, "cases": len(cases), "baseline": _group(base_rows), "query_understanding": _group(new_rows)}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"cases with gold chunks: {len(cases)} (split={args.split})")
    header = f"{'category':26}{'n':>4} | {'hit@1':>11} {'hit@5':>11} {'recall@10':>13} {'mrr@20':>13}"
    print(header)
    print("-" * len(header))
    for cat in report["baseline"]:
        b, a = report["baseline"][cat], report["query_understanding"][cat]
        print(f"{cat:26}{b['n']:>4} | {b['hit@1']:>5.2f}>{a['hit@1']:<5.2f} {b['hit@5']:>5.2f}>{a['hit@5']:<5.2f} {b['recall@10']:>6.2f}>{a['recall@10']:<6.2f} {b['mrr@20']:>6.2f}>{a['mrr@20']:<6.2f}")
    if args.show_misses:
        for old, new in zip(base_rows, new_rows):
            if new["hit_rank"] is None or (old["hit_rank"] and new["hit_rank"] > old["hit_rank"]):
                print(f"MISS/REGRESS [{new['category']}] {new['query']} base={old['hit_rank']} new={new['hit_rank']} expected={new['expected']} top3={new['top5'][:3]}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
