# `scripts/` — script dòng lệnh

Chạy mọi script **từ thư mục gốc repo**, trong môi trường `labor`
(`conda activate labor`). Thêm `--help` để xem đầy đủ tham số.

## Khởi động và cài đặt (PowerShell)

| Script | Làm gì |
|---|---|
| `setup.ps1` | Kiểm tra Python ≥ 3.10, Node, npm, Docker; cài `requirements.txt`, `pip install -e .`, `npm ci` cho frontend (`-SkipFrontend` để bỏ qua) |
| `dev.ps1` | Bật Qdrant (`docker compose`) nếu cổng 6333 trống, mở cửa sổ backend (:8000) và frontend (:3000). Không tự index. `-SkipQdrant` để bỏ bước Qdrant |

## Chuỗi xử lý dữ liệu (chỉ cần khi tạo lại corpus)

Corpus đã có sẵn trong `data/processed/chunks/`; máy mới **chỉ cần bước 5**.

| Bước | Script | Đầu vào → đầu ra |
|---|---|---|
| 1 | `load_pdf.py --input data/raw/labor_law.pdf` | PDF → `data/extracted/labor_law.txt` |
| 2 | `clean_text.py` | text → `labor_law_cleaned.txt` (bỏ header/footer, chuẩn hoá khoảng trắng) |
| 3 | `parse_labor_law.py` | text sạch → `labor_law.json` (cây Chương/Mục/Điều/Khoản/Điểm) |
| 4 | `chunk_labor_law.py --strategy legal --max-tokens 450` | cây → `labor_law_legal.jsonl` (Điều; Điều dài tách theo Khoản) |
| 5 | `index_labor_law_chunks.py` | JSONL → vector + payload trong Qdrant |

```powershell
python scripts/index_labor_law_chunks.py --data data/processed/chunks/labor_law_legal.jsonl `
    --model dek21 --collection labor_law_dek21_768 --qdrant-url http://localhost:6333
# Qdrant nhúng, không cần server (ví dụ Kaggle):
python scripts/index_labor_law_chunks.py --model dek21 --qdrant-path ./data/qdrant_local --device cuda
```

Không dùng `--recreate` trừ khi cố ý xây lại collection. Tạo lại corpus (bước
1–4) sẽ làm đổi `chunk_id` → phải kiểm tra lại bộ eval.

## Chạy thử pipeline từ dòng lệnh

| Script | Làm gì |
|---|---|
| `retrieve_labor_law.py` | Chỉ truy xuất (dense / bm25 / hybrid); `--list-queries` liệt kê câu mẫu trong `configs/retrieval_queries.json`; `--save-candidates` lưu kết quả |
| `rerank_labor_law.py` | Rerank một file ứng viên đã lưu bằng BGE |
| `build_labor_law_context.py` | Chọn ngữ cảnh từ file đã rerank, in/lưu context |
| `run_rag_pipeline.py` | Truy xuất → rerank → chọn ngữ cảnh, tương tác, không sinh câu trả lời |
| `run_generation.py --query "…"` | Toàn bộ pipeline trong bộ nhớ, kèm sinh câu trả lời |

Ba script đầu dùng để soi từng bước riêng lẻ; hai script sau đi qua
`labor_rag.runtime` giống backend. Kết quả trung gian ghi vào `data/logs/`,
`data/retrieval_results/`, `data/reranking_results/` (không commit).

Muốn đánh giá có hệ thống thì dùng [`evals/`](../evals/README.md), không dùng các script này.
