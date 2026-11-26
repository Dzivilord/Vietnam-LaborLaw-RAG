# Backend — FastAPI

Lớp API mỏng bọc quanh thư viện `labor_rag`. Toàn bộ logic RAG nằm ở
[`src/labor_rag`](../../src/labor_rag/README.md); backend chỉ khởi tạo runtime,
nhận request, trả JSON và ghi trace.

## Chạy

Từ **thư mục gốc repo** (không phải `app/backend`), sau khi đã chạy Qdrant và
index corpus:

```powershell
python -m uvicorn app.backend.api.main:app --reload
```

- Swagger UI: <http://127.0.0.1:8000/docs>
- Kiểm tra sống: `GET /health` → `{"status": "ok"}`

Khi khởi động (`lifespan` trong `api/main.py`), backend đọc `.env` ở thư mục gốc,
dựng `ModelManager` (tải embedder, reranker, dựng chỉ mục BM25 trong bộ nhớ) và
`RAGRuntime` theo `configs/labor_rag.yaml`. Nó **không** tự index Qdrant.

## Cấu trúc

```text
api/
├── main.py          # FastAPI app, lifespan, CORS, Logfire (tuỳ chọn), /health
├── schemas.py       # Pydantic: ChatRequest, ChatResponse, EvidenceCitation, ...
└── routes/
    ├── chat.py      # POST /chat — endpoint cho giao diện chính
    └── admin.py     # /admin/* — trace và chạy chẩn đoán cho trang /admin
requirements.txt     # trỏ về requirements.txt gốc (một bộ dependency duy nhất)
```

## Endpoint

| Method | Đường dẫn | Mục đích |
|---|---|---|
| `GET` | `/health` | Kiểm tra backend sống |
| `POST` | `/chat` | Hỏi đáp; trả câu trả lời, trạng thái, căn cứ, gợi ý |
| `GET` | `/admin/traces?limit=50` | Trace gần nhất trong bộ nhớ (tối đa 200) |
| `POST` | `/admin/runs` | Chạy một request chẩn đoán, trả `request_id` |
| `GET` | `/admin/runs/{id}/events` | Luồng SSE các bước pipeline của request đó |

### `POST /chat`

```json
{ "query": "Thời gian thử việc tối đa là bao lâu?",
  "options": { "enable_rerank": true, "generation": { "temperature": 0.2 } } }
```

`options` là tuỳ chọn và chỉ áp dụng cho request đó (không lưu vào config).
`query` dài tối đa 4000 ký tự.

Phản hồi (rút gọn):

```json
{
  "request_id": "…",
  "status": "generated",
  "answer": "Thời gian thử việc … không quá 60 ngày [E1] …",
  "citations": ["E1", "E2"],
  "evidence": [{ "evidence_id": "E1", "chunk_id": "bo-luat-lao-dong-2019:article-25",
                 "article_number": "25", "article_title": "Thời gian thử việc",
                 "content": "Điều 25. …", "chapter_title": "…" }],
  "suggestions": [],
  "debug": { "status": "generated", "total_latency_ms": 4230.5, "diagnostics": { … } }
}
```

`status` cho biết kết quả của pipeline — giao diện dựa vào đây để hiển thị:

| `status` | Ý nghĩa |
|---|---|
| `generated` | Có câu trả lời, trích dẫn hợp lệ |
| `generated_with_warnings` | Có câu trả lời nhưng trích dẫn có cảnh báo |
| `clarification_required` | Câu hỏi quá chung; `suggestions` chứa chủ đề (Điều + tiêu đề) |
| `insufficient_context` | Không tìm thấy căn cứ đủ liên quan |
| `abstained` | Câu ngoài corpus (`unknown`) hoặc trích dẫn sai/thiếu ⇒ không trả lời |
| `out_of_scope` | Không liên quan luật lao động |
| `refused` | Yêu cầu nhạy cảm (né luật, giả mạo…) |
| `error` | Lỗi nội bộ; chi tiết trong `debug.error` |

`debug.diagnostics` chứa số liệu chẩn đoán (thời gian từng bước, số chunk, quyết
định guardrail, token) — dùng cho `/admin`, không hiển thị ở giao diện chính.

## Cấu hình và bảo mật

- CORS chỉ cho phép `http://localhost:3000` và `http://127.0.0.1:3000`; đổi cổng
  frontend thì sửa `allow_origins` trong `api/main.py`.
- Logfire bật bằng `LOGFIRE_ENABLED=true` (instrument FastAPI + span `rag.*`).
- Trace lưu trong bộ nhớ tiến trình (`RequestTraceRecorder`, 200 bản ghi gần nhất);
  mirror Firestore khi đặt `FIRESTORE_TRACE_COLLECTION`. Mặc định chỉ gửi hash câu hỏi.
- **`/admin/*` không có xác thực.** Không public ra Internet nếu chưa đặt sau
  proxy/xác thực.
