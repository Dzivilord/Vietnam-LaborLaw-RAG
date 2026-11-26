# `labor_rag` — thư viện lõi RAG

Gói Python chứa toàn bộ pipeline: tiền xử lý văn bản luật, truy xuất, rerank, chọn
ngữ cảnh, guardrail và sinh câu trả lời. Cài dạng editable từ thư mục gốc:

```powershell
python -m pip install -e .        # tên gói: labor-law-rag
```

Backend FastAPI, script CLI và bộ eval đều dùng gói này qua **một điểm vào duy
nhất**: `RAGRuntime` / `ModelManager` trong `runtime/`.

## Luồng xử lý một câu hỏi

```text
RAGRuntime.chat(query)                                   runtime/rag_runtime.py
└─ LaborLawRAGPipeline.run                               pipelines/application_pipeline.py
   ├─ GuardrailService.check_input                       guardrails/
   │    rules (regex) → classifier (rules | jev | openrouter_jev)
   │    ⇒ refuse (nhạy cảm) | abstain (ngoài phạm vi / ngoài corpus) | clarify | continue
   ├─ RetrievalContextPipeline.run                       pipelines/rag_pipeline.py
   │    ├─ retriever.retrieve(top_k=20)                  retrieval/
   │    │    QueryAwareRetriever (tuỳ chọn) → HybridRetriever(Dense + BM25, RRF k=60)
   │    ├─ reranker.rerank(top_k=10)                     reranking/bge_reranker.py | qwen3.py
   │    ├─ RerankedContextBuilder.build                  reranking/context_builder.py
   │    │    ngưỡng tuyệt đối + tương đối, khử trùng lặp Điều/Khoản,
   │    │    tối đa 8 chunk / 3500 token, sắp theo thứ tự luật, gắn nhãn E1, E2…
   │    └─ (tuỳ chọn) QueryAmbiguityClassifier           query_understanding/
   ├─ GuardrailService.check_evidence  ⇒ không có chunk nào → insufficient_context
   ├─ (câu mơ hồ) trả clarification_required + gợi ý chủ đề, không gọi LLM
   ├─ RAGGenerator.generate                              generation/
   │    prompt bắt buộc trích dẫn [E…], provider Ollama | OpenRouter | Gemini
   └─ GuardrailService.check_output  ⇒ trích dẫn sai/thiếu → abstained
```

Kết quả là `RAGRunResult` (`models/rag.py`): `status`, `answer`, `evidence`
(danh sách `EvidenceItem` kèm Điều/Khoản/Chương), `suggestions`, `diagnostics`.

## Các module

| Module | Vai trò | File chính |
|---|---|---|
| `runtime/` | Đọc config YAML, dựng mọi thành phần, vòng đời model, điểm vào | `config.py`, `factory.py`, `model_manager.py`, `rag_runtime.py`, `cli.py` |
| `pipelines/` | Ghép các bước thành pipeline | `rag_pipeline.py` (truy xuất → context), `application_pipeline.py` (+ guardrail, sinh) |
| `guardrails/` | Chính sách chặn/cho qua ở 3 điểm | `service.py` (chính sách), `classifier.py` (rules/JEV/OpenRouter), `rules.py` (regex) |
| `retrieval/` | Truy xuất | `dense_retriever.py`, `bm25_retriever.py`, `hybrid_retriever.py`, `query_process.py` |
| `reranking/` | Rerank và chọn ngữ cảnh | `bge_reranker.py`, `qwen3.py`, `context_builder.py` |
| `embeddings/` | Mô hình embedding | `dense/dek21.py` (mặc định, tách từ bằng pyvi), `dense/bge.py`, `sparse/bge_sparse.py` |
| `generation/` | Prompt, generator, provider LLM | `prompts.py`, `generator.py`, `providers/ollama.py`, `openrouter.py`, `gemini.py` |
| `query_understanding/` | Phân loại câu hỏi mơ hồ sau truy xuất (NLI) | `ambiguity_classifier.py` |
| `storage/` | Adapter Qdrant (server hoặc chế độ nhúng `path=`) | `qdrant.py` |
| `indexing/` | Embed chunk và ghi vào Qdrant | `indexer.py` |
| `ingestion/` | Đọc PDF/HTML/web, làm sạch văn bản | `loaders/`, `cleaners/` |
| `processing/` | Phân tích cấu trúc Chương/Mục/Điều/Khoản và chia chunk | `parsers/parser.py`, `chunkers/legal_structure_chunker.py` |
| `models/` | Kiểu dữ liệu dùng chung | `rag.py` (`EvidenceItem`, `RAGRunResult`) |
| `observability/` | Span Logfire tuỳ chọn, trace request | `logfire_spans.py`, `request_traces.py` |

Một số file rỗng là chỗ dành sẵn, chưa dùng: `observability/{logging,metrics,tracing}.py`,
`retrieval/{pipeline,sparse_retriever,context_builder}.py`,
`pipelines/indexing_pipeline.py`, `processing/parsers/references_parsers.py`.
Context builder đang dùng là `reranking/context_builder.py`.

## Chi tiết đáng chú ý

**Truy xuất lai.** Dense (DEk21, 768 chiều, Qdrant cosine) và BM25 (trong bộ nhớ,
tách token theo ký tự chữ/số, `k1=1.5`, `b=0.75`) mỗi bên lấy 20 ứng viên rồi hợp
nhất bằng *weighted Reciprocal Rank Fusion* (`rrf_k=60`).

**Query understanding** (`retrieval/query_process.py`, tắt mặc định, chạy offline):
- tra cứu trực tiếp "Điều N", "Khoản M Điều N" ⇒ ghim chunk tương ứng; Điều không
  tồn tại ⇒ trả rỗng để pipeline từ chối thay vì trả điều "na ná";
- mở rộng viết tắt (NLĐ, HĐLĐ, BHXH…) và từ đời thường → thuật ngữ luật
  (đuổi việc → đơn phương chấm dứt, tăng ca → làm thêm giờ…);
- khôi phục dấu cho câu gõ không dấu bằng thống kê n-gram của corpus.
Chỉ câu dùng cho truy xuất bị viết lại; reranker và LLM vẫn nhận câu gốc.

**Chọn ngữ cảnh.** Điểm rerank chỉ có ý nghĩa *so sánh trong một câu hỏi*; câu hỏi
văn nói làm tụt điểm đồng loạt. Vì vậy `min_score` (mặc định 0.30) chỉ là sàn
"có gì liên quan không", còn `relative_to_top` (0.60 × điểm cao nhất) mới quyết
định giữ chunk nào. Hiệu chỉnh bằng `evals.sweep_context`.

**Guardrail.** Ba điểm kiểm tra:
1. *Đầu vào*: regex độ chính xác cao (né luật, giả mạo, đòi system prompt, hỏi
   Điều > 220) chạy trước, sau đó tới classifier. Nhãn `out_of_scope` → abstain,
   `sensitive_or_disallowed` → refuse, `unknown` (cần văn bản ngoài Bộ luật) →
   abstain, `ambiguous` → hỏi lại sau truy xuất (`clarify_after_retrieval`).
   Ngưỡng tin cậy chung và theo từng nhãn cấu hình được.
2. *Bằng chứng*: không còn chunk nào sau khi chọn ngữ cảnh → `insufficient_context`.
3. *Đầu ra*: câu trả lời trích dẫn ID không tồn tại hoặc không trích dẫn →
   `abstained`.

**Sinh câu trả lời.** Prompt (`generation/prompts.py`) chỉ cho phép dùng thông tin
trong phần BẰNG CHỨNG, bắt buộc gắn `[E1]`…, nói rõ khi không đủ căn cứ.

## Mở rộng

Thêm một biến thể thành phần (retriever, reranker, provider…):

1. Viết class theo interface có sẵn (`retrieval/base.py`, `reranking/base.py`,
   `generation/base.py`).
2. Thêm giá trị vào `Literal` tương ứng trong `runtime/config.py`.
3. Dựng nó trong hàm `build_*` của `runtime/factory.py` (kèm kiểm tra trong
   `validate_runtime_config`).
4. Chọn qua `configs/labor_rag.yaml`; viết unit test với stub trong `tests/`.

Span Logfire dùng tên ổn định `rag.request`, `rag.guardrail.input`, `rag.retrieve`,
`rag.rerank`, `rag.context`, `rag.generate` (thuộc tính `rag.*`); chỉ bật khi
`LOGFIRE_ENABLED=true`, không gửi nội dung câu hỏi trừ khi
`RAG_TRACE_INCLUDE_CONTENT=true`.
