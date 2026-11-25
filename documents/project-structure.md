# Cấu trúc project Labor Law RAG

Tài liệu này mô tả cấu trúc hiện tại của project và vai trò của từng khu vực. Project là hệ thống hỏi đáp pháp luật lao động Việt Nam, gồm backend FastAPI, frontend Next.js và thư viện Python cho pipeline RAG.

```text
RAG/
├── app/
│   ├── backend/
│   │   ├── api/
│   │   │   ├── main.py                 # Khởi tạo FastAPI, lifespan và CORS
│   │   │   ├── schemas.py              # Pydantic request/response schemas
│   │   │   └── routes/
│   │   │       └── chat.py             # POST /chat: gọi RAG runtime và trả evidence
│   │   └── requirements.txt            # Dependencies dành cho backend
│   └── frontend/
│       ├── src/
│       │   ├── app/                    # Next.js App Router, page và CSS tổng thể
│       │   ├── components/             # ChatPanel, ExplainPanel, DocumentViewer...
│       │   ├── services/api.ts          # Client gọi FastAPI
│       │   └── types/rag.ts             # TypeScript types cho RAG response
│       ├── package.json                 # Scripts và dependencies frontend
│       └── next.config.js               # Cấu hình Next.js
│
├── src/labor_rag/                      # Python package: lõi nghiệp vụ RAG
│   ├── embeddings/                      # Dense/sparse embedding implementations
│   ├── generation/                      # Prompt, generator và LLM providers
│   │   └── providers/                   # Ollama, Gemini
│   ├── indexing/                        # Đưa chunks và vectors vào vector database
│   ├── ingestion/                       # Load PDF/HTML/web và làm sạch văn bản đầu vào
│   ├── models/                          # Domain models: document, RAG result
│   ├── observability/                   # Nơi dành cho logging, metrics, tracing
│   ├── pipelines/                       # Ghép retrieval, rerank, context, generation
│   ├── processing/                      # Parser và chunker theo cấu trúc văn bản pháp luật
│   ├── query_understanding/             # Phân loại câu hỏi mơ hồ và gợi ý làm rõ
│   ├── reranking/                       # Reranker và context builder sau rerank
│   ├── retrieval/                       # Dense, BM25, hybrid retrieval và query processing
│   ├── runtime/                         # Config, factory, lifecycle model, health, API boundary
│   └── storage/                         # Adapter Qdrant
│
├── data/
│   ├── raw/                             # PDF/text pháp luật gốc
│   ├── extracted/                       # Văn bản sau trích xuất và metadata trung gian
│   └── processed/
│       └── chunks/                      # Corpus chunks JSONL dùng cho retrieval/indexing
│
├── configs/
│   ├── labor_rag.yaml                   # Model, retrieval type, Qdrant collection và corpus
│   └── retrieval_queries.json           # Câu query phục vụ kiểm tra/thử nghiệm retrieval
│
├── scripts/                             # CLI cho ingest, parse, clean, chunk, index và dev setup
│   ├── run_rag_pipeline.py
│   ├── index_labor_law_chunks.py
│   ├── chunk_labor_law.py
│   ├── parse_labor_law.py
│   ├── setup.ps1
│   └── dev.ps1
│
├── tests/                               # Unit/contract tests cho pipeline và API contract
├── evals/                               # Dành cho benchmark, golden set và đánh giá chất lượng RAG
├── infra/                               # Dành cho IaC, manifests deploy, CI/CD production
├── monitoring/                          # Dành cho dashboards, alerts và cấu hình observability
├── notebooks/                           # EDA và thử nghiệm retrieval/embedding
├── documents/                           # Tài liệu kỹ thuật, vận hành và sản phẩm
│   └── project-structure.md             # Tài liệu này
│
├── docker-compose.yml                   # Local Qdrant service
├── pyproject.toml                       # Package metadata và Python dependencies chính
├── requirements.txt                     # Dependencies Python runtime/indexing
├── README.md                            # Hướng dẫn cài đặt, chạy và troubleshoot
└── .env                                 # Cấu hình local/secrets, không được commit
```

## Luồng chạy chính

```text
Browser
  -> Next.js frontend (app/frontend)
  -> FastAPI POST /chat (app/backend/api/routes/chat.py)
  -> RAGRuntime (src/labor_rag/runtime)
  -> LaborLawRAGPipeline (src/labor_rag/pipelines)
  -> Hybrid retrieval: Qdrant dense vectors + BM25
  -> Reranker + context builder
  -> LLM provider (mặc định Ollama)
  -> Answer + evidence citations
  -> Frontend hiển thị câu trả lời và nguồn trích dẫn
```

## Luồng dữ liệu/indexing

```text
Văn bản gốc (data/raw)
  -> extract/parse/clean scripts
  -> legal structure chunker
  -> data/processed/chunks/*.jsonl
  -> embedding model
  -> Qdrant collection
  -> runtime retrieval
```

## Quy ước phát triển

- Business logic mới thuộc `src/labor_rag/`; API route chỉ nên làm nhiệm vụ validate request, gọi runtime và map response.
- Không commit `.env`, Qdrant storage (`data/qdrant/`), model cache, `node_modules/` hoặc build artifacts.
- Thay đổi corpus phải đi kèm metadata nguồn, version/effective date và quy trình re-index có thể tái lập.
- `evals/`, `infra/` và `monitoring/` là các điểm mở rộng ưu tiên để đưa project từ local MVP lên môi trường production.
