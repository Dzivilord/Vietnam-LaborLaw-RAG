# `configs/` — cấu hình pipeline

| File | Dùng cho |
|---|---|
| `labor_rag.yaml` | Cấu hình chính của backend, CLI và eval |
| `labor_rag_qwen3.yaml` | Bản sao của `labor_rag.yaml`, chỉ khác reranker (Qwen3-Reranker) — để so sánh |
| `retrieval_queries.json` | Câu hỏi mẫu cho `scripts/retrieve_labor_law.py --query-id …` |

YAML được đọc bởi `RAGRuntimeConfig.from_yaml`
([`src/labor_rag/runtime/config.py`](../src/labor_rag/runtime/config.py)) và kiểm
tra bởi `validate_runtime_config` (`runtime/factory.py`) lúc khởi động. File được
viết **thưa có chủ đích**: khoá nào không ghi sẽ lấy giá trị mặc định trong
dataclass. Đường dẫn tương đối tính từ thư mục gốc repo.

Khi đổi một khoá trong `labor_rag.yaml`, nhớ sửa tương ứng trong
`labor_rag_qwen3.yaml` nếu muốn hai bản vẫn so sánh được.

## Các nhóm khoá

### `retrieval`

| Khoá | Mặc định | Ý nghĩa |
|---|---|---|
| `type` | `hybrid` | `dense`, `bm25` hoặc `hybrid` (ánh xạ sang `method`) |
| `data_path` | — | Corpus JSONL cho BM25 (bắt buộc với `bm25`/`hybrid`) |
| `collection` | theo embedder | Tên collection Qdrant, phải khớp số chiều (`labor_law_dek21_768`) |
| `top_k` / `candidate_k` | 20 / 20 | Số ứng viên trả ra / lấy từ mỗi nhánh trước khi hợp nhất |
| `qdrant_url` | `http://localhost:6333` | Qdrant server |
| `qdrant_path` | — | Thư mục Qdrant nhúng (không cần server, ví dụ trên Kaggle); có thì bỏ qua `qdrant_url` |
| `rrf_k`, `dense_weight`, `bm25_weight` | 60, 1.0, 1.0 | Tham số Reciprocal Rank Fusion |
| `bm25_k1`, `bm25_b` | 1.5, 0.75 | Tham số BM25 |
| `device` | tự chọn | `cpu`, `cuda` cho embedder |

### `embedding`

`model`: giá trị `bge` chọn BGE-M3; **mọi giá trị khác** (kể cả tên đầy đủ
`huyydangg/DEk21_hcmute_embedding_v2`) đều chọn DEk21 768 chiều.

### `query_understanding`

| Khoá | Mặc định | Ý nghĩa |
|---|---|---|
| `enabled` | `false` | Bật lớp hiểu câu hỏi trước truy xuất |
| `article_lookup` | `true` | Ghim chunk khi câu hỏi nêu "Điều N"/"Khoản M Điều N" |
| `expand_terms`, `expand_abbreviations`, `restore_diacritics` | `true` | Mở rộng từ đời thường, viết tắt; khôi phục dấu |

### `reranker`

| Khoá | Mặc định | Ý nghĩa |
|---|---|---|
| `enabled` | `true` | Tắt thì chọn context thẳng từ kết quả truy xuất |
| `type` | `bge` | `bge` hoặc `qwen3` |
| `model` | theo `type` | `BAAI/bge-reranker-v2-m3` / `Qwen/Qwen3-Reranker-0.6B` |
| `top_k` | 10 | Số chunk giữ sau rerank (≤ `retrieval.top_k`) |
| `no_rerank_context_k` | 5 | Số chunk lấy khi tắt rerank |
| `qwen3_instruction` | có sẵn | Chỉ dẫn cho Qwen3 (mặc định: bỏ qua lời chào/văn nói) |

### `context`

| Khoá | Mặc định trong YAML | Ý nghĩa |
|---|---|---|
| `min_score` | 0.30 | Sàn tuyệt đối: dưới mức này coi như không liên quan |
| `relative_to_top` | 0.60 | Giữ chunk có điểm ≥ tỉ lệ × điểm cao nhất |
| `max_chunks` / `max_tokens` | 8 / 3500 | Ngân sách ngữ cảnh đưa vào LLM |
| `order_by` | `legal` | Sắp theo thứ tự Điều/Khoản (`relevance` = theo điểm) |

Các ngưỡng **chưa được hiệu chỉnh** và phụ thuộc reranker (điểm BGE ≠ Qwen3).
Hiệu chỉnh bằng `python -m evals.sweep_context --run <file run>`.

### `ambiguity` (phân loại mơ hồ sau truy xuất, tắt mặc định)

`enabled`, `min_relevant_chunks` (4), `model` (NLI đa ngôn ngữ),
`min_decision_score`, `min_decision_margin`.

### `generation`

| Khoá | Mặc định | Ý nghĩa |
|---|---|---|
| `provider` | `ollama` | `ollama` hoặc `openrouter` |
| `model` | `qwen3.5:4b` | Tên model của provider |
| `temperature`, `max_output_tokens` | 0.2, 1200 | |
| `ollama_url` | `OLLAMA_BASE_URL` hoặc `http://localhost:11434` | |
| `timeout` | 180 | Giây |

### `guardrails`

| Khoá | Giá trị hiện tại | Ý nghĩa |
|---|---|---|
| `scope_enabled`, `sensitive_enabled`, `rules_enabled` | `true` | Bật chặn ngoài phạm vi / nhạy cảm / regex |
| `ambiguity_mode` | `clarify_after_retrieval` | `clarify` (dừng ngay), `clarify_after_retrieval` (trả gợi ý chủ đề sau khi truy xuất), `broad_context` (bỏ qua) |
| `unknown_action` | `abstain` | Câu cần văn bản ngoài Bộ luật: `abstain` hoặc `continue` |
| `evidence_quality_enabled`, `output_enabled` | `true` | Guardrail bằng chứng và đầu ra |
| `require_citations`, `reject_invalid_citations` | `true` | Câu trả lời thiếu/sai trích dẫn thì không trả |
| `classifier.provider` | `openrouter_jev` | `rules` (offline), `jev` (checkpoint HF cục bộ, cần `model`), `openrouter_jev` (cần `OPENROUTER_API_KEY`) |
| `classifier.threshold` | 0.70 | Độ tin cậy tối thiểu để chặn |
| `classifier.label_thresholds` | `{}` | Ngưỡng riêng theo nhãn, ví dụ `{unknown: 0.35}` |
| `classifier.model` | `typesafe/jev-1.13` | Model Decisions API |

Mục tiêu hiện tại của router: **chặn chắc chắn** câu lạc chủ đề và câu nhạy cảm,
**không chặn nhầm** câu hợp lệ; câu mơ hồ/ngoài corpus được phép đi tiếp.
