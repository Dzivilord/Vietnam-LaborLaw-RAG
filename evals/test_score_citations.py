from evals.score_citations import score_case, score_run

EXPECTED = [
    {"article_number": "35", "required": True},
    {"article_number": "107", "clause_number": "2", "required": True},
]


def test_perfect_citations():
    cited = [{"article_number": "35"}, {"article_number": "107", "clause_number": "2"}]
    scores = score_case(cited, EXPECTED)
    assert scores["citation_precision"] == 1.0 and scores["citation_recall"] == 1.0


def test_wrong_clause_is_not_a_match():
    scores = score_case([{"article_number": "107", "clause_number": "3"}], EXPECTED)
    assert scores["citation_precision"] == 0.0 and scores["citation_recall"] == 0.0


def test_article_level_citation_matches_clause_level_gold():
    scores = score_case([{"article_number": "107"}], EXPECTED)
    assert scores["citation_precision"] == 1.0 and scores["citation_recall"] == 0.5


def test_extra_citation_lowers_precision_only():
    cited = [{"article_number": "35"}, {"article_number": "999"}]
    scores = score_case(cited, [EXPECTED[0]])
    assert scores["citation_precision"] == 0.5 and scores["citation_recall"] == 1.0


def test_score_run_skips_abstentions_and_unlabelled_cases():
    golden = [
        {"id": "a", "category": "answerable_direct", "expected_legal_references": [EXPECTED[0]]},
        {"id": "b", "category": "ambiguous"},
    ]
    runs = [
        {"id": "a", "actual_action": "answer", "cited_legal_references": []},
        {"id": "b", "actual_action": "clarify", "cited_legal_references": []},
    ]
    report = score_run(runs, golden)
    assert report["case_count"] == 1 and report["uncited_answer_rate"] == 1.0
