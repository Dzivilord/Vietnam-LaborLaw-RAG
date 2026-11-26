# Cấu trúc project

Bản đồ toàn bộ repo và vai trò từng phần. Mỗi thư mục quan trọng có README riêng
(liên kết bên dưới); tài liệu này cho cái nhìn tổng thể và các quy ước.

```text
RAG/
├── app/
│   ├── backend/                         → app/backend/README.md
│   │   └── api/
│   │       ├── main.py                  # FastAPI app, lifespan (nạp model), CORS, /health
│   │       ├── schemas.py               # ChatRequest / ChatResponse / EvidenceCitation
│   │       └── routes/
│   │           ├── chat.py              # POST /chat — giao diện chính
│   │           └── admin.py             # /admin/traces, /admin/runs (+ SSE)
│   └── frontend/                        → app/frontend/README.md
│       └── src/
│           ├── app/                     # page.tsx (hỏi đáp), admin/ (quan sát), CSS
│           ├── components/              # Welcome, ChatTurn, AnswerMarkdown, EvidencePanel, Composer
│           ├── lib/status.ts            # trạng thái pipeline → thông báo hiển thị
│           └── services/api.ts          # client gọi backend
│
├── src/labor_rag/                       → src/labor_rag/README.md
│   ├── runtime/                         # config YAML, factory, ModelManager, RAGRuntime (điểm vào)
│   ├── pipelines/                       # rag_pipeline (truy xuất→context), application_pipeline (+guardrail, sinh)
│   ├── guardrails/                      # rules, classifier (rules/JEV/OpenRouter), chính sách
│   ├── retrieval/                       # dense, BM25, hybrid RRF, query understanding
│   ├── reranking/                       # BGE, Qwen3, context builder
│   ├── embeddings/                      # DEk21 (mặc định), BGE-M3, sparse
│   ├── generation/                      # prompt, generator, provider Ollama/OpenRouter/Gemini
│   ├── query_understanding/             # phân loại câu mơ hồ sau truy xuất
│   ├── storage/                         # Qdrant (server hoặc nhúng)
│   ├── indexing/, ingestion/, processing/   # đưa dữ liệu vào: đọc, làm sạch, parse, chunk, index
│   ├── models/                          # EvidenceItem, RAGRunResult
│   └── observability/                   # span Logfire, trace request
│
├── configs/                             → configs/README.md
│   ├── labor_rag.yaml                   # cấu hình chính
│   ├── labor_rag_qwen3.yaml             # biến thể reranker Qwen3
│   └── retrieval_queries.json           # câu mẫu cho script truy xuất
│
├── data/                                → data/README.md
│   ├── raw/                             # PDF Bộ luật Lao động 2019
│   ├── extracted/                       # text, cây cấu trúc, dẫn chiếu
│   └── processed/chunks/labor_law_legal.jsonl   # corpus 278 chunk (bắt buộc)
│
├── evals/                               → evals/README.md
│   ├── datasets/golden_v2.jsonl         # 250 câu gán nhãn (dev/heldout)
│   ├── run_router_v2.py                 # pha 1: router
│   ├── run_rag.py, rag_metrics.py       # pha 2: RAG
│   ├── sweep_context.py                 # hiệu chỉnh ngưỡng
│   ├── run_ragas.py                     # pha 3: RAGAS
│   └── tracing.py                       # Logfire cho eval
│
├── scripts/                             → scripts/README.md   (setup, dev, tiền xử lý, index, chạy thử)
├── tests/                               → tests/README.md     (unit test với stub)
├── documents/
│   ├── project-structure.md             # tài liệu này
│   └── kaggle_eval.md                   # chạy pha 2 trên Kaggle GPU
│
├── docker-compose.yml                   # Qdrant v1.19.0, lưu ở ./data/qdrant
├── pyproject.toml                       # gói labor-law-rag (src/)
├── requirements.txt                     # dependency Python chạy hệ thống
├── requirements-eval.txt                # + ragas cho pha 3
├── .env.example                         # mẫu biến môi trường (không chứa bí mật)
├── CLAUDE.md                            # hướng dẫn cho trợ lý lập trình AI
└── README.md                            # tổng quan, cài đặt, kết quả
```

Thư mục `infra/`, `monitoring/`, `notebooks/` hiện **trống** (dành cho triển khai,
giám sát, thử nghiệm sau này). `01_data_overview.ipynb` là notebook khám phá dữ
liệu. `reference.txt` là tài liệu tham khảo cá nhân, không thuộc hệ thống.

## Luồng một câu hỏi

```text
Trình duyệt ─► Next.js (app/frontend) ─► POST /chat (app/backend)
  ─► RAGRuntime.chat ─► LaborLawRAGPipeline
       guardrail đầu vào ─► truy xuất lai (Qdrant + BM25) ─► rerank
       ─► chọn ngữ cảnh ─► guardrail bằng chứng ─► LLM (Ollama) ─► guardrail đầu ra
  ◄─ { status, answer có [E1]…, evidence (nguyên văn Điều/Khoản), suggestions }
Giao diện: hiển thị câu trả lời, nhãn trích dẫn bấm được, khung căn cứ pháp lý.
```

## Luồng dữ liệu

```text
data/raw/labor_law.pdf
  ─► load_pdf ─► clean_text ─► parse_labor_law (cây Chương/Điều/Khoản)
  ─► chunk_labor_law (Điều; Điều dài tách theo Khoản)
  ─► data/processed/chunks/labor_law_legal.jsonl
  ─► index_labor_law_chunks (DEk21 → Qdrant)   +   BM25 dựng lúc backend khởi động
```

## Quy ước

- Logic nghiệp vụ nằm trong `src/labor_rag/`; route API chỉ kiểm tra request, gọi
  runtime và trả kết quả.
- Thành phần mới thêm qua `runtime/factory.py`, chọn bằng một `Literal` trong
  `runtime/config.py`, bật trong YAML.
- Chạy Python từ thư mục gốc repo.
- Không commit `.env`, `data/qdrant/`, `evals/{runs,reports,cache}/`,
  `node_modules/`, `.next/`, cache model.
- Không chia chunk lại corpus: nhãn của bộ eval tham chiếu `chunk_id` hiện tại.
- Thay đổi hành vi (router, ngưỡng, prompt, model) phải đo lại bằng `evals/`
  trên `dev`, báo cáo trên `heldout`.
