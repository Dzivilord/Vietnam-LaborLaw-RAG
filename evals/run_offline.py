"""Run the reproducible no-network evaluation suite against a JSONL dataset."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .deterministic_metrics import build_report
from .schemas import load_cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate captured Labor Law RAG results.")
    parser.add_argument("--cases", required=True, help="JSONL evaluation dataset.")
    parser.add_argument("--output", default="evals/reports/latest.json", help="Destination JSON report.")
    args = parser.parse_args()
    cases = load_cases(args.cases)
    report = {"generated_at": datetime.now(timezone.utc).isoformat(), "dataset": str(args.cases), **build_report(cases)}
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nSaved report: {destination}")


if __name__ == "__main__":
    main()
