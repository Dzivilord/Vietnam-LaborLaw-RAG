# `tests/` — unit test

Kiểm thử nhanh, **không cần dịch vụ ngoài**: không Qdrant, không Ollama, không
tải model. Retriever, reranker, classifier và LLM được thay bằng stub.

```powershell
python -m pytest tests                 # unit test pipeline
python -m pytest tests evals           # + test metric và bộ dữ liệu eval
python -m pytest tests/test_guardrails.py::test_pipeline_returns_out_of_scope_before_retrieval
```

| File | Kiểm tra |
|---|---|
| `test_guardrails.py` | Chính sách guardrail: chặn ngoài phạm vi trước truy xuất, từ chối câu nhạy cảm, `unknown → abstain` (và công tắc `continue`), ngưỡng theo nhãn, câu mơ hồ trả gợi ý chủ đề sau truy xuất, rule "Điều > 220" và đòi system prompt, bỏ qua guardrail đầu vào khi eval, adapter OpenRouter JEV |
| `test_query_process.py` | Hiểu câu hỏi: bỏ dấu, nhận diện "Điều/Khoản", khôi phục dấu, mở rộng viết tắt/từ đời thường, ghim chunk theo Điều, Điều không tồn tại trả rỗng |
| `test_rag_contracts.py` | Hợp đồng dữ liệu giữa context builder, generator và kết quả pipeline |

Test của bộ đánh giá nằm trong [`evals/`](../evals/README.md) (`evals/test_*.py`).
Frontend không có test riêng; `npm run build` đóng vai trò kiểm tra kiểu.
