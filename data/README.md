# `data/` — dữ liệu

Nguồn duy nhất: **Bộ luật Lao động 2019**, Luật số 45/2019/QH14, ban hành
20/11/2019, hiệu lực 01/01/2021. 17 chương, 220 điều.

```text
data/
├── raw/                         # tài liệu gốc (có trong Git)
│   ├── labor_law.pdf            # văn bản luật
│   └── labor_law_citations.txt  # các cụm dẫn chiếu nội bộ ("khoản 2 Điều 18…") trích ra để kiểm tra
├── extracted/                   # kết quả trung gian (có trong Git)
│   ├── labor_law.txt            # text trích từ PDF
│   ├── labor_law_cleaned.txt    # text sau khi làm sạch
│   ├── labor_law.json           # cây Chương → Mục → Điều → Khoản → Điểm sau khi parse
│   ├── labor_law_references.json# dẫn chiếu giữa các điều
│   └── labor_law_eda.xlsx       # thống kê khám phá dữ liệu
├── processed/chunks/
│   ├── labor_law_legal.jsonl    # ★ CORPUS DÙNG CHO HỆ THỐNG (278 chunk)
│   └── labor_law_legal.pretty.json  # cùng nội dung, định dạng dễ đọc
├── qdrant/                      # dữ liệu Qdrant (sinh ra, không commit)
└── logs/, retrieval_results/, reranking_results/   # đầu ra script (không commit)
```

## Corpus `labor_law_legal.jsonl`

**Bắt buộc** cho cả BM25 (đọc lúc backend khởi động) và indexing Qdrant.
**Không xoá, không chia chunk lại** — bộ dữ liệu đánh giá (`evals/`) tham chiếu
đúng các `chunk_id` hiện tại; chia lại sẽ làm hỏng nhãn.

Chiến lược chia (`processing/chunkers/legal_structure_chunker.py`): mỗi **Điều**
là một chunk; Điều quá dài (> 450 token) được tách theo **Khoản**.

| | Số chunk |
|---|---|
| Cấp Điều (`chunk_level: article`) | 208 |
| Cấp Khoản (`chunk_level: clause`, từ các Điều dài) | 70 |
| Tổng | 278 (median 154 token, lớn nhất 1203) |

Một Điều chỉ xuất hiện ở **một** cấp: hoặc nguyên Điều, hoặc tách thành các Khoản.

### Trường dữ liệu chính

| Trường | Ví dụ | Ghi chú |
|---|---|---|
| `chunk_id` | `bo-luat-lao-dong-2019:article-25`, `…:article-107-clause-2` | Khoá ổn định, dùng trong nhãn eval |
| `parent_id` | `bo-luat-lao-dong-2019:article-107` | Với chunk cấp Khoản |
| `chunk_level` | `article` / `clause` | |
| `document_id`, `document_title`, `document_number` | `bo-luat-lao-dong-2019`, `Bộ luật Lao động 2019`, `45/2019/QH14` | |
| `issued_date`, `effective_date` | `2019-11-20`, `2021-01-01` | |
| `chapter_number`, `chapter_title`, `section_*`, `article_number`, `article_title`, `clause_number` | `III`, `HỢP ĐỒNG LAO ĐỘNG`, …, `25`, `Thời gian thử việc` | Phân cấp luật, hiển thị ở khung căn cứ |
| `content` | `Điều 25. Thời gian thử việc …` | Nguyên văn hiển thị cho người dùng và đưa vào LLM |
| `embedding_text` | `Chương III. … \n Điều 25. … \n …` | Văn bản đem đi embed: có thêm tiêu đề Chương/Điều để tăng ngữ cảnh |
| `token_count`, `split_reason`, `is_oversized` | | Thông tin chia chunk |

## Tạo lại từ PDF (chỉ khi thật sự cần)

Chuỗi script trong [`scripts/`](../scripts/README.md):
`load_pdf.py` → `clean_text.py` → `parse_labor_law.py` → `chunk_labor_law.py`
→ `index_labor_law_chunks.py`. Máy mới chỉ cần bước cuối (index).
Nếu tạo lại corpus, phải chạy lại `python -m evals.validate_golden_dataset` để
kiểm tra nhãn eval còn khớp `chunk_id`.
