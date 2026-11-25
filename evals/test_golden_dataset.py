from collections import Counter
from pathlib import Path

from evals.schemas import load_golden_cases

GOLDEN = Path("evals/datasets/golden_v2.jsonl")


def test_golden_v2_is_labelled_and_contains_all_case_types() -> None:
    cases = load_golden_cases(GOLDEN)
    assert len(cases) == 250
    assert {case.expected_case_type for case in cases} == {
        "answerable", "unknown", "ambiguous", "out_of_scope", "sensitive_or_disallowed",
    }
    assert all(not hasattr(case, "actual_action") for case in cases)
    answerable = [case for case in cases if case.expected_case_type == "answerable"]
    assert all(case.reference_answer and case.expected_legal_references and case.relevant_chunk_ids for case in answerable)


def test_golden_v2_ids_queries_and_splits_are_consistent() -> None:
    import json

    rows = [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len({r["id"] for r in rows}) == len({r["query"] for r in rows}) == len(rows)
    assert set(Counter(r["split"] for r in rows)) == {"dev", "heldout"}
    answers = [r for r in rows if r["expected_action"] == "answer"]
    assert len(answers) == 171
