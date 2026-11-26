# Trợ lý Luật Lao động (Labor Law RAG)

Hệ thống hỏi đáp **Bộ luật Lao động 2019 (Luật số 45/2019/QH14)** theo kiến trúc
RAG (Retrieval-Augmented Generation). Người dùng hỏi bằng tiếng Việt thường
ngày; hệ thống tìm đúng Điều/Khoản liên quan, sinh câu trả lời **có trích dẫn**,
và cho phép mở nguyên văn điều luật để kiểm chứng. Khi không đủ căn cứ, hệ thống
nói rõ thay vì suy đoán.

Toàn bộ pipeline chạy được **cục bộ**: embedding và reranker từ Hugging Face,
vector DB Qdrant, mô hình sinh `qwen3.5:4b` qua Ollama.

```text
Người dùng ──► Next.js UI ──► FastAPI /chat ──► RAG runtime
                                              │
   1. Guardrail đầu vào ── chặn câu lạc chủ đề / nhạy cảm
   2. Truy xuất lai ────── Qdrant (dense, DEk21) + BM25, hợp nhất bằng RRF
   3. Rerank ───────────── BGE-reranker-v2-m3 (hoặc Qwen3-Reranker)
   4. Chọn ngữ cảnh ────── ngưỡng điểm, ngân sách token, thứ tự Điều/Khoản
   5. Guardrail bằng chứng ─ không có căn cứ ⇒ không trả lời
   6. Sinh câu trả lời ─── Ollama qwen3.5:4b, bắt buộc trích dẫn [E1]…
   7. Guardrail đầu ra ─── trích dẫn sai/thiếu ⇒ không trả lời
```

## Tính năng chính

- **Trả lời có căn cứ**: mỗi nhận định gắn Điều/Khoản; bấm để mở nguyên văn.
- **Biết từ chối**: câu ngoài phạm vi, câu nhạy cảm (né luật, giả mạo hồ sơ…),
  câu không tìm thấy căn cứ đều có trạng thái và thông điệp riêng.
- **Hỏi lại khi câu hỏi quá chung** (ví dụ "tiền lương"): trả về danh sách chủ
  đề lấy từ các điều luật tìm được để người dùng chọn.
- **Hiểu cách hỏi đời thường**: tra cứu trực tiếp "Điều 35", "Khoản 2 Điều 107";
  mở rộng từ lóng/viết tắt (NLĐ, đuổi việc, tăng ca…), khôi phục dấu tiếng Việt.
- **Bộ đánh giá hai pha** (router và RAG) với 250 câu hỏi gán nhãn, theo dõi bằng
  Logfire.
- **Trang quan sát nội bộ** `/admin`: chạy thử, xem thời gian từng bước, trace.

## Kết quả đánh giá hiện có

| Hạng mục | Kết quả | Ghi chú |
|---|---|---|
| Router (`openrouter_jev`), heldout 124 câu | lọt câu cần chặn **0%**, chặn nhầm **0%** | câu mơ hồ/ngoài corpus được phép đi tiếp |
| Truy xuất BM25 + query understanding, heldout | hit@5 **0.85 → 0.95**, MRR@20 0.73 → 0.84 | đo offline, chưa gồm dense/rerank |
| End-to-end (rerank, sinh, trích dẫn, RAGAS) | **chưa đo** | công cụ đã sẵn sàng, xem [`evals/`](evals/README.md) |

Nhãn bộ dữ liệu do AI soạn và mới được rà soát một phần; xem chi tiết và cảnh báo
trong [`evals/README.md`](evals/README.md).

## Cấu trúc thư mục

| Thư mục | Nội dung | Tài liệu |
|---|---|---|
| `app/frontend/` | Giao diện Next.js (trang hỏi đáp + `/admin`) | [README](app/frontend/README.md) |
| `app/backend/` | API FastAPI: `/chat`, `/health`, `/admin/*` | [README](app/backend/README.md) |
| `src/labor_rag/` | Thư viện Python: toàn bộ pipeline RAG | [README](src/labor_rag/README.md) |
| `configs/` | Cấu hình YAML của pipeline | [README](configs/README.md) |
| `data/` | Văn bản luật gốc, bản trích xuất, corpus chunk | [README](data/README.md) |
| `scripts/` | Script tiền xử lý, index, chạy thử, khởi động dev | [README](scripts/README.md) |
| `evals/` | Bộ dữ liệu vàng, đánh giá router/RAG/RAGAS | [README](evals/README.md) |
| `tests/` | Unit test (không cần dịch vụ ngoài) | [README](tests/README.md) |
| `documents/` | Tài liệu bổ sung: cấu trúc chi tiết, chạy eval trên Kaggle | [project-structure](documents/project-structure.md) |

## Cài đặt nhanh (Windows / PowerShell)

Yêu cầu: Python 3.10+ (đã kiểm thử Conda 3.11), Node.js 20.9+, Docker Desktop,
Ollama. GPU không bắt buộc (CPU chạy được, chậm hơn).

```powershell
cd D:\Projects\RAG
conda create -n labor python=3.11 -y
conda activate labor
.\scripts\setup.ps1                      # cài Python package + npm ci cho frontend

docker compose up -d qdrant              # Qdrant v1.19.0, dữ liệu ở ./data/qdrant
ollama pull qwen3.5:4b                   # mô hình sinh câu trả lời

# Index corpus vào Qdrant — chỉ cần một lần trên máy mới
python scripts/index_labor_law_chunks.py --data data/processed/chunks/labor_law_legal.jsonl `
    --model dek21 --collection labor_law_dek21_768 --qdrant-url http://localhost:6333

.\scripts\dev.ps1                        # mở backend :8000 và frontend :3000
```

Mở <http://localhost:3000> và thử: *"Thời gian thử việc tối đa là bao lâu?"*.
Lần chạy đầu sẽ tải model Hugging Face (DEk21, BGE reranker) nên mất vài phút.

Chạy thủ công thay cho `dev.ps1` (mỗi lệnh một terminal, từ thư mục gốc):

```powershell
python -m uvicorn app.backend.api.main:app --reload     # backend, Swagger ở /docs
cd app/frontend; npm run dev                            # frontend
```

**Luôn chạy lệnh Python từ thư mục gốc repo** — import `app.backend…` và các
đường dẫn tương đối trong `configs/labor_rag.yaml` phụ thuộc vào điều đó.

## Biến môi trường

Sao chép `.env.example` thành `.env` (backend tự đọc khi khởi động; biến đã có
trong shell được ưu tiên hơn).

Cấu hình hiện tại dùng guardrail `openrouter_jev`, nên **cần `OPENROUTER_API_KEY`**
(chỉ cho bước phân loại câu hỏi; truy xuất và sinh vẫn chạy cục bộ). Muốn chạy
hoàn toàn offline, đặt `guardrails.classifier.provider: rules` trong
`configs/labor_rag.yaml` — đổi lại router sẽ chỉ chặn được các mẫu regex rõ ràng.

| Biến | Khi nào cần |
|---|---|
| `OPENROUTER_API_KEY` | Guardrail `openrouter_jev`, sinh qua OpenRouter, hoặc RAGAS. Dùng key thường, không dùng *management key*. |
| `OLLAMA_BASE_URL`, `OLLAMA_MODEL` | Đổi địa chỉ/mô hình Ollama |
| `LOGFIRE_ENABLED`, `LOGFIRE_TOKEN` | Gửi trace lên Logfire (tắt mặc định) |
| `FIRESTORE_TRACE_COLLECTION` | Sao lưu trace lên Firestore (chỉ hash câu hỏi, trừ khi `RAG_TRACE_INCLUDE_CONTENT=true`) |
| `NEXT_PUBLIC_RAG_API_URL` | Frontend gọi backend ở địa chỉ khác `http://localhost:8000` |

## Kiểm thử

```powershell
python -m pytest tests evals          # unit test + test metric/dataset, không cần dịch vụ
cd app/frontend; npm run build        # build + kiểm tra TypeScript cho frontend
```

## Lỗi thường gặp

| Triệu chứng | Cách xử lý |
|---|---|
| `No module named app` / `labor_rag` | Chạy từ thư mục gốc; `python -m pip install -e .` |
| Lỗi kết nối Qdrant / thiếu collection | `docker compose ps`; chạy lệnh index một lần |
| Giao diện báo "Không kết nối được máy chủ" | Backend chưa chạy ở cổng 8000, hoặc `/health` chưa trả `{"status":"ok"}` |
| Lỗi CORS | Frontend phải ở cổng 3000, hoặc thêm origin vào `app/backend/api/main.py` |
| OpenRouter `Request path cannot be relayed` | Đang dùng *management key*; tạo API key thường |
| OpenRouter `401 API key expired` | Tạo key mới, cập nhật `.env`, xoá biến cũ trong shell |
| Cổng 8000/3000 bị chiếm | `Get-NetTCPConnection -LocalPort 8000` rồi dừng tiến trình cũ |

## Giới hạn

- Chỉ có **một văn bản**: Bộ luật Lao động 2019. Câu hỏi cần Luật BHXH, Luật Việc
  làm, nghị định, thông tư, mức lương tối thiểu vùng… sẽ không có câu trả lời.
- Thông tin mang tính tham khảo, **không thay thế tư vấn pháp lý**.
- `/admin` không có xác thực — không mở ra Internet.
- Phiên bản Python chỉ đặt mức tối thiểu; frontend và image Qdrant được ghim phiên bản.
