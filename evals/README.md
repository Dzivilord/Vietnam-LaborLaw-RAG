# Offline evaluation suite

This directory is the quality gate for the Labor Law RAG system. It starts
fully local: the deterministic runner does not call Ollama, OpenRouter,
Qdrant, Logfire, or any evaluator API. It scores a captured JSONL dataset.

`datasets/golden_v2.jsonl` is the golden set (see the golden_v2 section below).
It contains no `actual_*`, answer, latency, evidence ID, or retrieval-context
field. Answerable cases contain stable legal references plus chunk IDs from the
current corpus; the other case types deliberately contain no source chunk.
Validate it against the corpus with:

```powershell
python -m evals.validate_golden_dataset
```

To execute the real configured pipeline against the golden dataset (Qdrant and
the configured local models must already be running), use:

```powershell
python -m evals.run_golden --limit 2 --no-generate  # retrieval/rerank smoke test
python -m evals.run_golden --split heldout          # held-out cases, including generation
```

The command writes captured output separately under `evals/runs/` and a merged
metric report under `evals/reports/`; neither changes the golden dataset.

## Quality groups

| Group | Initial metrics | Required fields |
|---|---|---|
| Routing | accuracy, macro-F1, per-route recall, confusion matrix | `expected_route`, `actual_route` |
| Retrieval | Recall@K, Precision@K, MRR@K, nDCG@K, Hit Rate@K | `relevant_chunk_ids`, `retrieved_chunk_ids` |
| Reranking | same rank metrics, compared with retrieval | `relevant_chunk_ids`, `reranked_chunk_ids` |
| Generation | citation validity and expected-citation recall | `citation_ids`, `evidence_ids`, optional `expected_citation_ids` |
| Safety / abstention | action macro-F1, unsafe-answer rate | `expected_action`, `actual_action` |
| Performance / reliability | p50/p95/max latency, error rate | `latency_ms`, `error` |

Actions must use one consistent label set: `answer`, `abstain`, `clarify`,
or `refuse`. Suggested routes are `labor_law`, `out_of_scope`, `sensitive`,
and `unknown` (adapt these once, then keep them stable across benchmark runs).

## Dataset contract

Use one JSON object per line. See `datasets/template_cases.jsonl`; each row has
the gold labels and the captured output of one pipeline run. Unknown fields are
preserved in `metadata`, so the schema can grow without breaking old datasets.

Do not overwrite a benchmark after recording a baseline. Create versioned files
such as `datasets/golden_v3.jsonl` and store reports under `evals/reports/`
(reports are git-ignored).

## Run the fully local suite

```powershell
python -m pip install -r requirements.txt
python -m evals.run_offline --cases evals/datasets/template_cases.jsonl
pytest evals/test_deterministic_metrics.py -q
```

The generated report is saved at `evals/reports/latest.json` and contains all
six groups. A metric with `null` means the dataset has no eligible labelled
case; it is not a passing score.

## Optional DeepEval LLM judges

DeepEval is installed for the generation layer, but is deliberately opt-in.
After you provide a curated dataset with `answer` and `retrieval_context`, copy
those cases into `datasets/generation_cases.jsonl`, configure a separate
evaluator model, then run:

```powershell
$env:DEEPEVAL_RUN_LLM_JUDGE = "1"
pytest evals/test_deepeval_judges.py -q
```

This invokes `FaithfulnessMetric` and `AnswerRelevancyMetric`. It is not
offline and should not initially use the same Qwen model that serves answers.
For legal correctness, add a human-reviewed reference answer/rubric before
turning any LLM-judge result into a release gate.

## golden_v2 (250 router-labelled cases)

`datasets/golden_v2.jsonl` is labelled for guardrail-router evaluation. Every row
has `category` (10 groups), `expected_route`, `expected_action`, `split`
(`dev`/`heldout`) and `label_status: draft_unreviewed` (labels were drafted by an
AI and need human review, especially `ambiguous`, `unknown_corpus`, `sensitive`).
Tune on `dev`; report on `heldout` only.

| category | route / action | n |
|---|---|---|
| answerable_direct, answerable_multi, article_lookup, colloquial, edge_input | labor_law / answer | 60+30+20+40+13 |
| sensitive_hard_negative | labor_law / answer (must NOT be refused) | 8 |
| ambiguous | labor_law / clarify | 20 |
| unknown_corpus (+ non-existent article) | unknown / abstain | 20+1 |
| out_of_scope (+ pure prompt injection) | out_of_scope / abstain | 25+1 |
| sensitive | sensitive / refuse | 12 |

```powershell
python -m evals.validate_golden_dataset --golden evals/datasets/golden_v2.jsonl
python -m evals.run_router_v2 --split heldout --show-errors      # phase 1: router quality per category
python -m evals.run_query_understanding --split heldout           # offline BM25 A/B of query rewriting
python -m evals.score_citations --run evals/runs/golden_<ts>.jsonl --golden evals/datasets/golden_v2.jsonl
```

## Two-phase evaluation workflow

Routing and RAG quality are evaluated separately: the router may call a paid
API, and RAG runs need a GPU (see `documents/kaggle_eval.md`). Tune on `dev`,
report on `heldout` once.

| Phase | Command | Runs on | Cases |
|---|---|---|---|
| 1. Router | `python -m evals.run_router_v2 --split dev` | local | all 250 |
| 2. RAG | `python -m evals.run_rag --split dev --with-abstain-probes` | Kaggle GPU / local | 171 answerable (+21 unknown-corpus probes) |
| 2b. Thresholds | `python -m evals.sweep_context --run evals/runs/rag-<id>.jsonl` | anywhere, no model | from a saved run |
| 3. LLM judge | `python -m evals.run_ragas --run evals/runs/rag-<id>.jsonl` | anywhere, judge API | answered cases |

Phase 1 caches classifier decisions in `evals/cache/router_decisions.jsonl`
(except the free `rules` classifier), so re-runs only pay for new queries.
Phase 2 skips the input guardrail and stores every candidate score, context,
answer and citation per case, so 2b and 3 never re-run models. Every run gets a
`run_id` (`<phase>-<UTC time>-<config hash>`); its per-case rows go to
`evals/runs/<run_id>.jsonl` and its report to `evals/reports/<run_id>.json`.

### Metrics

| Group | Metric | Meaning |
|---|---|---|
| Router | `leak_rate` | should be refused/abstained/clarified but was answered |
| Router | `false_block_rate` | answerable but blocked |
| Router | `correct_block_action_rate` | blocked with the right action (refuse vs abstain vs clarify) |
| Router | accuracy per category, route/action macro-F1, p95 latency, error rate | |
| Retrieval | `hit@1/5/10/20`, `recall@5/10/20`, `mrr@20` | gold chunk among fused candidates |
| Rerank | `hit@1`, `hit@5`, `recall@5`, `mrr@10`, `hit@1_gain_vs_retrieval` | ordering after rerank |
| Context | `context_hit`, `context_recall`, `mean_context_chunks` | what the LLM actually sees |
| Context | `threshold_loss_rate` | gold survived rerank but thresholds dropped it (calibrate with 2b) |
| Answer | `answer_rate`, `false_abstain_rate` | answerable cases answered vs abstained |
| Citation | `citation_precision`, `citation_recall`, `uncited_answer_rate` | cited Điều/Khoản vs gold |
| Abstain | `correct_abstain_rate` | unknown-corpus probes that were not answered |
| Performance | `p50/p95_latency_ms`, `p95_<stage>_ms`, `error_rate`, `mean_completion_tokens` | |
| Diagnosis | `failure_stage` | first stage where an answerable case broke: retrieval → rerank → context → abstained → citation |
| RAGAS | `faithfulness`, `factual_correctness`, `context_precision`, `context_recall` (+ `answer_relevancy`) | LLM-judge scores, 0..1 |

### Observability (Logfire)

Every phase emits one trace per run when `LOGFIRE_TOKEN` is set (service
`labor-rag-eval`, environment `local`/`kaggle`; disable with `EVAL_LOGFIRE=false`):

```
eval.run (phase, run_id, git_sha, config_hash, models, thresholds)
├── eval.case (case_id, category, split, expected vs actual, failure_stage, scores)
│   ├── rag.retrieve / rag.rerank / rag.context / rag.generate   (phase 2)
└── eval.summary  log: eval.metric.<group>.<name>
```

Useful queries in Logfire's SQL explorer (adjust the run id):

```sql
-- Compare runs: headline metrics per run
SELECT attributes->>'eval.run_id' AS run, attributes->>'eval.phase' AS phase,
       attributes->>'eval.metric.context.context_hit' AS context_hit,
       attributes->>'eval.metric.citation.citation_precision' AS citation_precision,
       attributes->>'eval.metric.performance.p95_latency_ms' AS p95_ms
FROM records WHERE span_name = 'eval.summary' ORDER BY start_timestamp DESC LIMIT 20;

-- Where do answerable cases break? (one run)
SELECT attributes->>'eval.category' AS category, attributes->>'eval.failure_stage' AS stage, count(*) AS n
FROM records WHERE span_name = 'eval.case' AND attributes->>'eval.run_id' = 'rag-...'
GROUP BY 1, 2 ORDER BY 1, 3 DESC;

-- Router leaks to review
SELECT attributes->>'eval.category' AS category, attributes->>'eval.query' AS query
FROM records WHERE span_name = 'eval.case' AND attributes->>'eval.run_id' = 'router-...'
  AND attributes->>'eval.leaked' = 'true';

-- Slowest stage spans
SELECT span_name, approx_percentile_cont(duration, 0.95) AS p95_s
FROM records WHERE span_name LIKE 'rag.%' GROUP BY span_name;
```

Pin these as dashboard panels. Golden questions are synthetic, so eval traces
include query/answer text; the backend keeps `RAG_TRACE_INCLUDE_CONTENT=false`.
