"""Optional DeepEval LLM-as-a-judge tests.

They never run by default. Set DEEPEVAL_RUN_LLM_JUDGE=1 and configure a
separate evaluator model before running this file. Do not use the production
answering model as the judge for a legal-quality release gate.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

if os.getenv("DEEPEVAL_RUN_LLM_JUDGE") != "1":
    pytest.skip("LLM judge is opt-in; set DEEPEVAL_RUN_LLM_JUDGE=1.", allow_module_level=True)

deepeval = pytest.importorskip("deepeval")
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from evals.schemas import load_cases


DATASET = Path(os.getenv("DEEPEVAL_CASES", "evals/datasets/generation_cases.jsonl"))


@pytest.mark.parametrize("case", load_cases(DATASET), ids=lambda case: case.id)
def test_answer_grounding_and_relevance(case) -> None:
    if not case.answer or not case.retrieval_context:
        pytest.skip("Case needs answer and retrieval_context for an LLM judge.")
    test_case = LLMTestCase(input=case.query, actual_output=case.answer, retrieval_context=list(case.retrieval_context))
    assert_test(test_case, [FaithfulnessMetric(threshold=0.7), AnswerRelevancyMetric(threshold=0.7)])
