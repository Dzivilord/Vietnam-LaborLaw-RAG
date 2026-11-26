# Chạy pha 2 (RAG eval) trên Kaggle GPU

Pha 1 (router) chạy ở máy local. Pha 2 (retrieval → rerank → context → generation)
chạy trên Kaggle. Sweep ngưỡng và RAGAS chạy được ở bất kỳ đâu trên file kết quả.

## 0. Chuẩn bị (một lần)

1. Commit và push code lên GitHub (notebook clone từ đó). Kiểm tra trên GitHub có
   `evals/`, `evals/datasets/golden_v2.jsonl`, `configs/labor_rag_qwen3.yaml`,
   `data/processed/chunks/labor_law_legal.jsonl`.
2. Tài khoản Kaggle đã xác minh số điện thoại (bắt buộc để bật GPU và Internet).
3. Tạo notebook mới → panel bên phải:
   - **Accelerator**: GPU T4 x2 (hoặc P100)
   - **Internet**: On
   - **Add-ons → Secrets** (tuỳ chọn):
     - `GITHUB_TOKEN`: chỉ cần nếu repo private (token quyền `repo` read)
     - `LOGFIRE_TOKEN`: write token của project Logfire, để xem trace
     - `OPENROUTER_API_KEY`: chỉ cần cho RAGAS (key thường, không phải management key)

   Bật từng secret cho notebook (tick "Attached").

Pha 2 không gọi router nên **không cần** `OPENROUTER_API_KEY`.

## 1. Lấy code và cài thư viện

```python
import os, subprocess
from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()
for name in ("LOGFIRE_TOKEN", "OPENROUTER_API_KEY", "GITHUB_TOKEN"):
    try:
        os.environ[name] = secrets.get_secret(name)
    except Exception:
        pass  # secret not attached: fine

repo = "github.com/Dzivilord/Vietnam-LaborLaw-RAG.git"
token = os.environ.get("GITHUB_TOKEN")
url = f"https://{token}@{repo}" if token else f"https://{repo}"
subprocess.run(["git", "clone", "--depth", "1", url, "/kaggle/working/RAG"], check=True)
os.chdir("/kaggle/working/RAG")   # mọi lệnh sau chạy từ repo root
```

```bash
!pip install -q -r requirements.txt && pip install -q -e .
```

Cảnh báo xung đột phiên bản của pip thường vô hại. Nếu `import torch` lỗi sau
khi cài, chọn **Run → Restart & clear outputs**, rồi chạy lại cell 1 (bỏ qua clone
nếu thư mục đã có).

## 2. Model sinh câu trả lời (Ollama trên GPU)

Bỏ qua phần này nếu chạy `--no-generate`.

```bash
!apt-get -qq install -y zstd > /dev/null
!curl -fsSL https://ollama.com/install.sh | sh
```

```python
import subprocess, time
subprocess.Popen(["ollama", "serve"], stdout=open("/kaggle/working/ollama.log", "w"), stderr=subprocess.STDOUT)
time.sleep(5)
```

```bash
!ollama pull qwen3.5:4b
!ollama run qwen3.5:4b "xin chào" --verbose 2>&1 | tail -3   # kiểm tra chạy trên GPU
```

## 3. Index corpus vào Qdrant nhúng (không cần server, khoảng 1 phút)

```bash
!python scripts/index_labor_law_chunks.py --data data/processed/chunks/labor_law_legal.jsonl \
    --model dek21 --collection labor_law_dek21_768 --qdrant-path /kaggle/working/qdrant --device cuda
```

## 4. Chạy pha 2

Smoke test 3 câu trước:

```bash
!python -m evals.run_rag --split dev --limit 3 --qdrant-path /kaggle/working/qdrant --device cuda
```

Chạy thật trên dev, BGE và Qwen3:

```bash
!python -m evals.run_rag --split dev --with-abstain-probes \
    --qdrant-path /kaggle/working/qdrant --device cuda --label bge
!python -m evals.run_rag --split dev --with-abstain-probes --config configs/labor_rag_qwen3.yaml \
    --qdrant-path /kaggle/working/qdrant --device cuda --label qwen3
```

Mỗi lần chạy in ra `Per-case: evals/runs/rag-<id>.jsonl`. Ghi lại hai id này.

## 5. Hiệu chỉnh ngưỡng (vài giây, không cần model)

```bash
!python -m evals.sweep_context --run evals/runs/rag-<id-bge>.jsonl
!python -m evals.sweep_context --run evals/runs/rag-<id-qwen3>.jsonl
```

Chọn cặp `min_score` / `relative_to_top` có `ctx_hit` cao, `f_abst` thấp, ít chunk.
Sửa vào config (trên Kaggle hoặc sửa local rồi push), sau đó chạy heldout một lần:

```bash
!python -m evals.run_rag --split heldout --with-abstain-probes \
    --qdrant-path /kaggle/working/qdrant --device cuda --label bge-final
```

## 6. RAGAS (tuỳ chọn, tốn token judge)

```bash
!pip install -q ragas==0.4.4
!RAGAS_JUDGE_MODEL=openai/gpt-4.1-mini python -m evals.run_ragas --run evals/runs/rag-<id>.jsonl --limit 30
```

## 7. Lưu kết quả

Kaggle xoá session khi tắt. Nén rồi tải từ panel **Output**:

```bash
!zip -qr /kaggle/working/eval_results.zip evals/runs evals/reports
```

Giải nén vào repo local (`evals/runs`, `evals/reports`) để chạy lại sweep/RAGAS ở máy.

## Mẹo

* Chạy dài: dùng **Save Version → Save & Run All (Commit)**. Notebook chạy nền,
  tối đa 12 giờ, file trong `/kaggle/working` thành Output. Khi đó thay các
  `<id>` bằng cách đọc id tự động, ví dụ `sorted(glob("evals/runs/rag-*.jsonl"))[-1]`.
* Quota GPU: khoảng 30 giờ/tuần. Tắt session khi xong (**Stop session**).
* Lần đầu tải DEk21, BGE/Qwen3 reranker và qwen3.5:4b, mất vài phút.
* `Qwen/Qwen3-Reranker-4B` chạy được trên T4 (fp16) nếu bản 0.6B chưa đủ tốt:
  sửa `reranker.model` trong `configs/labor_rag_qwen3.yaml`.
* Qdrant nhúng chỉ cho một process: đừng index và eval song song trên cùng thư mục.
* Nếu Ollama báo lỗi kết nối: xem `/kaggle/working/ollama.log`, chạy lại cell `ollama serve`.
