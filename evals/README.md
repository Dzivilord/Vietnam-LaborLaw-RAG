# `evals/` — đánh giá chất lượng

Bộ công cụ đo chất lượng hệ thống, tách thành **hai pha** vì hai lý do: router
có thể gọi API trả phí, còn pha RAG cần GPU. Mọi lần chạy đều ghi kết quả từng
câu ra file để chấm lại sau mà không chạy lại model.

```text
Pha 1  run_router_v2   ── chỉ guardrail đầu vào, 250 câu, cache lời gọi classifier
Pha 2  run_rag         ── truy xuất → rerank → context → sinh, 172 câu trả lời được
   2b  sweep_context   ── hiệu chỉnh ngưỡng context từ file kết quả (không cần model)
Pha 3  run_ragas       ── chấm LLM-judge (RAGAS) trên file kết quả pha 2
```

Nguyên tắc: **chỉnh trên `dev`, báo cáo trên `heldout`** (chạy heldout một lần
sau khi đã chốt thay đổi).

## Bộ dữ liệu vàng `datasets/golden_v2.jsonl`

250 câu hỏi về Bộ luật Lao động 2019, mỗi câu có nhóm, route/hành động mong đợi
và split. Câu trả lời được có thêm đáp án tham chiếu, `relevant_chunk_ids` (khớp
`chunk_id` của corpus) và `expected_legal_references` (Điều/Khoản).

| Nhóm (`category`) | n | Route / hành động mong đợi | Nội dung |
|---|---|---|---|
| `answerable_direct` | 60 | `labor_law` / `answer` | Hỏi thẳng, một Điều |
| `answerable_multi` | 30 | `labor_law` / `answer` | Cần ghép nhiều Điều |
| `article_lookup` | 20 | `labor_law` / `answer` | "Điều 35 quy định gì?", "Khoản 2 Điều 107" |
| `colloquial` | 41 | `labor_law` / `answer` | Văn nói: "bị đuổi việc", "tăng ca", "đang bầu" |
| `edge_input` | 13 + 2 | `labor_law` / `answer` (+1 out_of_scope, +1 unknown) | Không dấu, tiếng Anh, viết tắt, teencode, prompt injection, "Điều 300" |
| `sensitive_hard_negative` | 8 | `labor_law` / `answer` | Nghe nhạy cảm nhưng là hỏi tuân thủ — **không được chặn** |
| `ambiguous` | 19 | `labor_law` / `clarify` | Thiếu tình tiết: "Công ty đuổi tôi có đúng không?" |
| `unknown_corpus` | 20 | `unknown` / `abstain` | Cần văn bản khác: BHXH, nghị định, mức phạt, lương tối thiểu vùng |
| `out_of_scope` | 25 | `out_of_scope` / `abstain` | Không liên quan luật lao động |
| `sensitive` | 12 | `sensitive` / `refuse` | Né luật, giả mạo, phân biệt đối xử |

Split xen kẽ trong từng nhóm: **dev 126 / heldout 124**, tỉ lệ nhóm gần như nhau.

> Nhãn do AI soạn (`label_status: draft_unreviewed`); mới có 1 câu được rà soát
> (`reviewed`). Ranh giới `ambiguous` / `unknown_corpus` / `sensitive` mang tính
> chủ quan — nên rà soát thủ công trước khi dùng số liệu cho mục đích chính thức.
> Bổ sung câu hỏi do người thật viết là cách tốt nhất để có số đo khách quan.

Kiểm tra bộ dữ liệu khớp corpus: `python -m evals.validate_golden_dataset`.

## Pha 1 — Router

```powershell
python -m evals.run_router_v2 --split dev --show-errors
python -m evals.run_router_v2 --split heldout
```

Mục tiêu router: **chặn chắc chắn** câu lạc chủ đề (`out_of_scope`) và câu nhạy
cảm, **không chặn nhầm** câu hợp lệ. Câu mơ hồ / ngoài corpus là "mềm": chặn đúng
thì tốt, lọt cũng chấp nhận được (truy xuất và guardrail bằng chứng xử lý sau).

| Chỉ số | Ý nghĩa |
|---|---|
| `critical_leak_rate` | Câu phải chặn (out_of_scope, sensitive) nhưng bị trả lời — **chỉ số chính** |
| `false_block_rate` | Câu trả lời được nhưng bị chặn |
| `acceptable_rate` | Đúng, hoặc câu "mềm" được cho đi tiếp |
| `soft_pass_rate` | Tỉ lệ câu mơ hồ/ngoài corpus được cho đi tiếp (tham khảo) |
| `leak_rate_all`, `correct_block_action_rate` | Phiên bản chặt: mọi câu cần chặn; chặn đúng hành động |
| accuracy/macro-F1 theo route và action, accuracy theo nhóm, p95 latency, error rate | |

Cache (`evals/cache/router_classifications.jsonl`) chỉ lưu **kết quả thô** của
classifier (nhãn, độ tin cậy, xác suất), khoá theo model + nội dung prompt. Đổi
ngưỡng, `ambiguity_mode`, `unknown_action` hay rule ⇒ chạy lại miễn phí; đổi
tiêu chí phân loại ⇒ tự gọi lại API. Classifier `rules` không cache. Script dừng
ngay khi gặp lỗi xác thực (401/403, *management key*) thay vì ghi báo cáo sai.

**Kết quả hiện tại** (`openrouter_jev`, `typesafe/jev-1.13`, ngưỡng 0.70):

| | dev (126) | heldout (124) |
|---|---|---|
| `critical_leak_rate` | 0.0 | 0.0 |
| `false_block_rate` | 0.0 | 0.0 |
| `acceptable_rate` | 1.0 | 1.0 |
| p95 latency | — | 770 ms |

Baseline classifier `rules` (chỉ regex): lọt ~95% câu cần chặn.
Lưu ý: việc giữ tiêu chí phân loại gốc được quyết định sau khi đã xem lỗi của một
lần chạy heldout với tiêu chí mở rộng (tiêu chí đó chặn nhầm 2 câu hợp lệ).

## Pha 2 — RAG

```powershell
python -m evals.run_rag --split dev --with-abstain-probes
python -m evals.run_rag --split dev --config configs/labor_rag_qwen3.yaml --label qwen3
python -m evals.run_rag --split dev --no-generate        # chỉ truy xuất/rerank/context
```

Chạy 172 câu trả lời được (`--with-abstain-probes` thêm 21 câu phải từ chối),
**bỏ qua guardrail đầu vào** (đã đo ở pha 1, không tốn API) và tự dùng classifier
`rules` để không cần khoá. Chạy trên Kaggle GPU: xem
[`documents/kaggle_eval.md`](../documents/kaggle_eval.md)
(`--qdrant-path`, `--device cuda`).

| Nhóm | Chỉ số |
|---|---|
| Truy xuất | `hit@1/5/10/20`, `recall@5/10/20`, `mrr@20` |
| Rerank | `hit@1`, `hit@5`, `recall@5`, `mrr@10`, `hit@1_gain_vs_retrieval` |
| Ngữ cảnh | `context_hit` (chunk đúng có trong ngữ cảnh đưa vào LLM), `context_recall`, `mean_context_chunks`, `threshold_loss_rate` (chunk đúng qua rerank nhưng bị ngưỡng loại) |
| Trả lời | `answer_rate`, `false_abstain_rate`, đếm theo `status` |
| Trích dẫn | `citation_precision`, `citation_recall` (so Điều/Khoản với nhãn), `uncited_answer_rate` |
| Từ chối | `correct_abstain_rate` trên câu ngoài corpus |
| Hiệu năng | `p50/p95_latency_ms`, `p95_<bước>_ms`, `error_rate`, `mean_completion_tokens` |
| Chẩn đoán | `failure_stage`: bước đầu tiên làm hỏng một câu — retrieval → rerank → context → abstained → citation |

Mọi chỉ số đều có bảng theo nhóm. **Chưa có kết quả end-to-end** — cần Qdrant +
Ollama (hoặc Kaggle).

### 2b — Hiệu chỉnh ngưỡng ngữ cảnh

```powershell
python -m evals.sweep_context --run evals/runs/rag-<id>.jsonl
```

Dùng điểm rerank đã lưu, chạy lại `RerankedContextBuilder` thật trên lưới
`min_score × relative_to_top`; in phân bố điểm top-1 (câu trả lời được vs câu phải
từ chối) và `context_hit`, `false_abstain`, `correct_abstain`, số chunk cho từng
cặp. Điểm phụ thuộc reranker — sweep riêng cho BGE và Qwen3.

### Thử nghiệm offline: query understanding

```powershell
python -m evals.run_query_understanding --split heldout
```

A/B chỉ trên BM25 (không cần model): trên heldout hit@1 0.63 → 0.76, hit@5
0.85 → 0.95, MRR@20 0.73 → 0.84; nhóm `article_lookup` hit@1 0.30 → 1.00. Bảng
từ đồng nghĩa được viết khi đã xem một phần câu hỏi, nên mức tăng ở nhóm văn nói
có thể lạc quan.

## Pha 3 — RAGAS (LLM-judge)

```powershell
pip install -r requirements-eval.txt               # ragas==0.4.4
$env:RAGAS_JUDGE_MODEL = "openai/gpt-4.1-mini"      # qua OpenRouter, dùng OPENROUTER_API_KEY
python -m evals.run_ragas --run evals/runs/rag-<id>.jsonl --limit 30
```

Chấm các câu đã được trả lời: `faithfulness` (câu trả lời có bám ngữ cảnh),
`factual_correctness` (so với đáp án tham chiếu), `context_precision`,
`context_recall`; `answer_relevancy` khi truyền `--embeddings-model`. Nên dùng
model judge khác model sinh. Kết quả judge được cache trên đĩa.

## Quan sát bằng Logfire

Đặt `LOGFIRE_TOKEN` (hoặc `logfire auth`) là mỗi lần chạy thành một trace
(service `labor-rag-eval`, môi trường `local`/`kaggle`; tắt bằng
`EVAL_LOGFIRE=false`):

```text
eval.run   (phase, run_id, git_sha, config_hash, model, ngưỡng)
├── eval.case   (case_id, category, split, expected vs actual, failure_stage, điểm số)
│   └── rag.retrieve / rag.rerank / rag.context / rag.generate   (pha 2)
└── eval.summary   log: eval.metric.<nhóm>.<tên>
```

Truy vấn mẫu trong SQL explorer của Logfire:

```sql
-- So sánh các lần chạy
SELECT attributes->>'eval.run_id' AS run, attributes->>'eval.phase' AS phase,
       attributes->>'eval.metric.context.context_hit' AS context_hit,
       attributes->>'eval.metric.citation.citation_precision' AS citation_precision
FROM records WHERE span_name = 'eval.summary' ORDER BY start_timestamp DESC LIMIT 20;

-- Câu hỏng ở bước nào (một lần chạy)
SELECT attributes->>'eval.category' AS category, attributes->>'eval.failure_stage' AS stage, count(*) AS n
FROM records WHERE span_name = 'eval.case' AND attributes->>'eval.run_id' = 'rag-...'
GROUP BY 1, 2 ORDER BY 1, 3 DESC;

-- Câu cần chặn bị lọt ở router
SELECT attributes->>'eval.category' AS category, attributes->>'eval.query' AS query
FROM records WHERE span_name = 'eval.case' AND attributes->>'eval.run_id' = 'router-...'
  AND attributes->>'eval.critical_leak' = 'true';
```

Câu hỏi trong bộ vàng là dữ liệu tổng hợp nên trace eval có kèm nội dung; backend
thật vẫn giữ `RAG_TRACE_INCLUDE_CONTENT=false`.

## Cấu trúc thư mục

| File | Vai trò |
|---|---|
| `datasets/golden_v2.jsonl` | Bộ vàng 250 câu |
| `datasets/template_cases.jsonl`, `generation_cases.jsonl` | Mẫu định dạng cho `run_offline` và DeepEval |
| `run_router_v2.py` | Pha 1 |
| `run_rag.py`, `rag_metrics.py` | Pha 2 và các chỉ số của nó |
| `sweep_context.py` | Hiệu chỉnh ngưỡng ngữ cảnh |
| `run_ragas.py` | Pha 3 |
| `run_query_understanding.py` | A/B BM25 cho query understanding |
| `score_citations.py` | Chấm trích dẫn theo Điều/Khoản |
| `run_golden.py` | Chạy end-to-end cả router + RAG trên bộ vàng (một pha) |
| `run_offline.py`, `deterministic_metrics.py`, `schemas.py` | Chấm offline file đã có sẵn kết quả; định nghĩa schema |
| `validate_golden_dataset.py` | Đối chiếu nhãn với `chunk_id` của corpus |
| `tracing.py` | Thiết lập Logfire cho eval |
| `test_*.py` | Test cho metric và bộ dữ liệu (`python -m pytest evals`) |

Kết quả ghi vào `evals/runs/<run_id>.jsonl` (từng câu) và
`evals/reports/<run_id>.json` (tổng hợp); `run_id` = `<pha>-<giờ UTC>-<hash config>`.
Ba thư mục `runs/`, `reports/`, `cache/` không commit.
