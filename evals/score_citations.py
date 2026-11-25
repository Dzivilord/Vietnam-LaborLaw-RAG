"""Score whether generated citations point at the *right* legal provisions.

``run_golden`` captures ``cited_legal_references`` (article/clause of every
evidence item an answer cited).  Evidence IDs such as ``E1`` change on every
run, so they cannot be compared with a gold label; legal references can.

    python -m evals.score_citations --run evals/runs/golden_<ts>.jsonl \
        --golden evals/datasets/golden_v2.jsonl

Reported per run (answerable cases that produced an answer):
* citation_precision - share of cited provisions that are in the gold set
* citation_recall    - share of required gold provisions that were cited
* uncited_answer_rate - answers that cite nothing
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _ref_matches(cited: dict[str, Any], expected: dict[str, Any]) -> bool:
    if str(cited.get("article_number") or "") != str(expected.get("article_number") or ""):
        return False
    cited_clause, expected_clause = cited.get("clause_number"), expected.get("clause_number")
    if cited_clause and expected_clause:
        return str(cited_clause) == str(expected_clause)
    return True


def score_case(cited: list[dict[str, Any]], expected: list[dict[str, Any]]) -> dict[str, float | int | None]:
    required = [ref for ref in expected if ref.get("required", True)]
    correct_cited = sum(any(_ref_matches(c, e) for e in expected) for c in cited)
    covered = sum(any(_ref_matches(c, e) for c in cited) for e in required)
    return {
        "cited": len(cited),
        "citation_precision": (correct_cited / len(cited)) if cited else None,
        "citation_recall": (covered / len(required)) if required else None,
    }


def score_run(run_rows: list[dict[str, Any]], golden_rows: list[dict[str, Any]]) -> dict[str, Any]:
    golden = {row["id"]: row for row in golden_rows}
    per_case, answered = [], 0
    for run in run_rows:
        gold = golden.get(run["id"])
        if not gold or not gold.get("expected_legal_references"):
            continue
        if run.get("actual_action") != "answer":
            continue  # abstentions are scored by the safety metrics, not here
        answered += 1
        scores = score_case(run.get("cited_legal_references") or [], gold["expected_legal_references"])
        per_case.append({"id": run["id"], "category": gold.get("category"), **scores})

    def mean(key: str) -> float | None:
        values = [row[key] for row in per_case if row[key] is not None]
        return round(sum(values) / len(values), 4) if values else None

    return {
        "case_count": len(per_case),
        "citation_precision": mean("citation_precision"),
        "citation_recall": mean("citation_recall"),
        "uncited_answer_rate": round(sum(row["cited"] == 0 for row in per_case) / len(per_case), 4) if per_case else None,
        "per_case": per_case,
    }


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Score citation correctness against gold legal references.")
    parser.add_argument("--run", required=True, help="evals/runs/golden_*.jsonl produced by run_golden")
    parser.add_argument("--golden", default="evals/datasets/golden_v2.jsonl")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    report = score_run(_read_jsonl(args.run), _read_jsonl(args.golden))
    print(json.dumps({k: v for k, v in report.items() if k != "per_case"}, ensure_ascii=False, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
