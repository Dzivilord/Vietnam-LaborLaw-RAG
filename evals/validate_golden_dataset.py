"""Validate a golden dataset against the current chunk IDs in the corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .schemas import load_golden_cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate legal-RAG golden cases.")
    parser.add_argument("--golden", default="evals/datasets/golden_v2.jsonl")
    parser.add_argument("--chunks", default="data/processed/chunks/labor_law_legal.jsonl")
    args = parser.parse_args()
    cases = load_golden_cases(args.golden)
    chunk_ids = {
        json.loads(line)["chunk_id"]
        for line in Path(args.chunks).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    missing = {
        chunk_id
        for case in cases
        for chunk_id in case.relevant_chunk_ids
        if chunk_id not in chunk_ids
    }
    if missing:
        raise SystemExit(f"Golden dataset references missing chunk IDs: {sorted(missing)}")
    case_types = {case.expected_case_type for case in cases}
    print(f"Validated {len(cases)} golden cases; case types: {', '.join(sorted(case_types))}.")


if __name__ == "__main__":
    main()
