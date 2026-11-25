# Running phase-2 RAG evals on Kaggle (GPU)

Phase 1 (router) runs locally: it needs no GPU, only the classifier API.
Phase 2 (retrieval → rerank → context → generation) runs here on a Kaggle GPU.
RAGAS scoring can run on Kaggle or locally on the downloaded run file.

Notebook settings: **Accelerator = GPU (T4 x2 or P100)**, **Internet = on**.
Secrets (Add-ons → Secrets): `GITHUB_TOKEN` (private repo only),
`LOGFIRE_TOKEN` (write token of the Logfire project), `OPENROUTER_API_KEY`
(only for RAGAS or OpenRouter generation).

> The notebook clones the GitHub repo, so commit and push the current code
> first. Alternatively upload the repo as a Kaggle Dataset and copy it to
> `/kaggle/working/RAG`.

## 1. Code and dependencies

```python
import os, subprocess
from kaggle_secrets import UserSecretsClient
secrets = UserSecretsClient()
for name in ("LOGFIRE_TOKEN", "OPENROUTER_API_KEY", "GITHUB_TOKEN"):
    try:
        os.environ[name] = secrets.get_secret(name)
    except Exception:
        pass

token = os.environ.get("GITHUB_TOKEN")
url = f"https://{token}@github.com/Dzivilord/Vietnam-LaborLaw-RAG.git" if token else "https://github.com/Dzivilord/Vietnam-LaborLaw-RAG.git"
subprocess.run(["git", "clone", "--depth", "1", url, "/kaggle/working/RAG"], check=True)
os.chdir("/kaggle/working/RAG")          # every command below runs from the repo root
```

```bash
!pip install -q -r requirements.txt && pip install -q -e .
!pip install -q -r requirements-eval.txt   # only if RAGAS runs here
```

## 2. Generation model (Ollama on the GPU)

```bash
!curl -fsSL https://ollama.com/install.sh | sh
!nohup ollama serve > /kaggle/working/ollama.log 2>&1 &
!sleep 5 && ollama pull qwen3.5:4b
```

Skip this section for `--no-generate` runs, or switch `generation.provider`
to `openrouter` in a config copy.

## 3. Index the corpus into embedded Qdrant (no server, ~1 minute)

```bash
!python scripts/index_labor_law_chunks.py --data data/processed/chunks/labor_law_legal.jsonl \
    --model dek21 --collection labor_law_dek21_768 --qdrant-path /kaggle/working/qdrant --device cuda
```

## 4. Phase 2 runs

```bash
# Logfire: traces go to the project of LOGFIRE_TOKEN, environment="kaggle"
!python -m evals.run_rag --split dev --with-abstain-probes \
    --qdrant-path /kaggle/working/qdrant --device cuda --label bge-default

!python -m evals.run_rag --split dev --with-abstain-probes --config configs/labor_rag_qwen3.yaml \
    --qdrant-path /kaggle/working/qdrant --device cuda --label qwen3-default
```

Calibrate thresholds on each dev run (no model needed, seconds):

```bash
!python -m evals.sweep_context --run evals/runs/<rag-run-id>.jsonl
```

Put the chosen `min_score` / `relative_to_top` in the config, then report
once on heldout:

```bash
!python -m evals.run_rag --split heldout --with-abstain-probes --qdrant-path /kaggle/working/qdrant --device cuda --label bge-calibrated
```

## 5. RAGAS (optional, costs judge tokens)

```bash
!RAGAS_JUDGE_MODEL=openai/gpt-4.1-mini python -m evals.run_ragas --run evals/runs/<rag-run-id>.jsonl --limit 30
```

## 6. Keep the results

Kaggle wipes the session. Download `evals/runs/` and `evals/reports/`
(they are git-ignored), e.g.:

```bash
!cd /kaggle/working/RAG && zip -qr /kaggle/working/eval_results.zip evals/runs evals/reports
```

Notes
* The first run downloads DEk21, BGE/Qwen3 reranker weights from Hugging Face.
* `Qwen/Qwen3-Reranker-4B` fits a T4 in fp16 if 0.6B is not robust enough.
* Embedded Qdrant is single-process: don't index and evaluate in two notebooks at once against the same folder.
