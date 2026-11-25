# Labor Law RAG

Setup and operation guide for the Labor Law RAG product.

This repository contains the source code, processed chunks, configuration,
FastAPI backend, and Next.js frontend. It does not require committed Python or
Node environments, Qdrant runtime storage, or model caches.

## Architecture

```text
Browser
	|
Next.js / React (app/frontend)
	|
FastAPI (app/backend/api)
	|
RAG Runtime (src/labor_rag/runtime)
	|
RAG Pipeline
	|-- hybrid retrieval (Qdrant dense + in-memory BM25)
	|-- BGE reranking
	|-- optional ambiguity check (disabled in current config)
	|-- hierarchy-aware context builder
	`-- generation via local Ollama

Qdrant <--- dense vectors and original chunk payloads
Hugging Face <--- embedding and reranker model downloads
Ollama <--- local qwen3.5:4b generation model
```

The backend entrypoint is `app.backend.api.main:app`. Run Python commands from
the repository root, `D:\Projects\RAG`, so paths and imports resolve correctly.

## Quick Start

```powershell
cd D:\Projects\RAG
conda create -n labor python=3.11 -y
conda activate labor
.\scripts\setup.ps1
docker compose up -d qdrant
ollama pull qwen3.5:4b
python scripts/index_labor_law_chunks.py --data data/processed/chunks/labor_law_legal.jsonl --model dek21 --collection labor_law_dek21_768 --qdrant-url http://localhost:6333
.\scripts\dev.ps1
```

Open `http://localhost:3000` and ask:

```text
Thời gian thử việc tối đa là bao lâu?
```

The first Python startup downloads Hugging Face models. The first Ollama
response loads the local generation model into memory.

## Prerequisites

- Python 3.10+. The validated setup uses Conda Python 3.11.
- Conda, or a Python 3.10+ virtual environment.
- Node.js 20.9+ and npm. The lockfile was tested with Node 24.14.1/npm 11.11.0.
- Docker Desktop on Windows/macOS, or Docker Engine plus Compose on Linux.
- Ollama running locally.
- Internet access for PyPI, npm, Hugging Face, and Ollama on first setup.

GPU is not required. CPU mode works but is slower. Exact RAM/VRAM needs depend
on Torch and model runtime; reserve several GB of RAM and several GB of disk
for the Ollama model and Hugging Face cache.

The supplied automation scripts are PowerShell and target Windows. On Linux or
macOS, run the equivalent commands manually from the repository root.

## Clone or Copy

For a private repository, use the normal authenticated remote. No public URL is
assumed:

```powershell
git clone <your-authenticated-repository-url> RAG
cd RAG
```

## Python Environment

```powershell
conda create -n labor python=3.11 -y
conda activate labor
python -m pip install -r requirements.txt
python -m pip install -e .
```

`requirements.txt` contains runtime/indexing dependencies. `pyproject.toml`
installs `src/labor_rag` as the `labor-law-rag` package. The backend
requirements file includes the root requirements file, so there is one
dependency set rather than two competing sets.

The configured DEk21 embedder requires `pyvi` for Vietnamese word segmentation.
The optional `.env` loader used by CLI scripts requires `python-dotenv`.

Verify an existing environment:

```powershell
python --version
python -c "import fastapi, pyvi, qdrant_client, sentence_transformers; print('Python imports OK')"
```

## Frontend

```powershell
cd app/frontend
npm ci
cd ../..
```

`package.json` declares direct dependencies/scripts and `package-lock.json`
records their resolved tree. `npm ci` recreates `node_modules` from the lockfile;
use `npm install` only when intentionally changing dependencies. Never commit
`node_modules/` or `.next/`.

## Qdrant and Indexing

Start Qdrant from the repository root:

```powershell
docker compose up -d qdrant
```

Compose starts service `labor-rag-qdrant` using Qdrant `v1.19.0` with REST at
`http://localhost:6333`, gRPC at `localhost:6334`, and storage mounted at
`./data/qdrant:/qdrant/storage`.

The active runtime uses collection `labor_law_dek21_768`, 768-dimensional
vectors, and `data/processed/chunks/labor_law_legal.jsonl`. That JSONL is
included and currently has 278 chunks.

Do **not** chunk the data again. A fresh machine still needs one indexing run to
create/populate its Qdrant collection:

```powershell
python scripts/index_labor_law_chunks.py `
	--data data/processed/chunks/labor_law_legal.jsonl `
	--model dek21 `
	--collection labor_law_dek21_768 `
	--qdrant-url http://localhost:6333
```

This downloads `huyydangg/DEk21_hcmute_embedding_v2` through Sentence
Transformers on first use and writes vectors plus full chunk payloads. Do not
use `--recreate` unless intentionally rebuilding. `scripts/dev.ps1` never
indexes automatically.

## Models

| Component       | Model                                 | Acquisition                     |
| --------------- | ------------------------------------- | ------------------------------- |
| Dense embedding | `huyydangg/DEk21_hcmute_embedding_v2` | Automatic Hugging Face download |
| Reranker        | `BAAI/bge-reranker-v2-m3`             | Automatic Hugging Face download |
| Generation      | `qwen3.5:4b`                          | Local Ollama                    |

Start Ollama locally and pull the default generation model:

```powershell
ollama pull qwen3.5:4b
```

## Configuration

`configs/labor_rag.yaml` currently specifies hybrid retrieval, BM25 data at
`data/processed/chunks/labor_law_legal.jsonl`, collection
`labor_law_dek21_768`, DEk21 768-dimensional embeddings, BGE reranking,
local Ollama generation with `qwen3.5:4b`, and ambiguity checking disabled.

Qdrant is currently configured as the code default `http://localhost:6333`, not
as an environment variable. Ollama supports these process variables:

```text
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3.5:4b
```

Create local configuration without committing secrets:

```powershell
Copy-Item .env.example .env
```

The backend loads the repository-root `.env` automatically at startup
(`load_dotenv(..., override=False)` in `app/backend/api/main.py`), so no extra
flag is needed. Variables already set in the process environment (for example
in the PowerShell session) take precedence over values in `.env`. `scripts/dev.ps1`
additionally passes `--env-file .env` to Uvicorn when that file exists; this is
redundant but harmless. The current YAML explicitly sets
the generation model, so `OPENROUTER_MODEL` is only relevant after changing
that config or using the provider outside the current YAML path.

## Start the Product

Manual startup from `D:\Projects\RAG`, in separate PowerShell terminals:

```powershell
# Terminal 1
conda activate labor
docker compose up -d qdrant
```

```powershell
# Terminal 2
conda activate labor
python -m uvicorn app.backend.api.main:app --reload
```

If `.env` exists at the repository root, the backend loads it automatically;
`--env-file .env` is not required.

```powershell
# Terminal 3
cd D:\Projects\RAG\app\frontend
npm run dev
```

Backend startup builds the in-memory BM25 index and loads embedding/reranker
components. It does not index Qdrant automatically.

Windows launcher:

```powershell
conda activate labor
.\scripts\dev.ps1
```

It starts Qdrant if needed and opens separate PowerShell windows for backend
and frontend. It checks ports 6333, 8000, and 3000 and does not start duplicate
servers. It does not index data. Close the opened server windows to stop the
app, then run `docker compose down` to stop Qdrant.

## Health Checks

```powershell
Invoke-WebRequest http://localhost:6333/ -UseBasicParsing
Invoke-WebRequest http://localhost:6333/collections/labor_law_dek21_768 -UseBasicParsing
Invoke-WebRequest http://127.0.0.1:8000/health -UseBasicParsing
```

Backend health must return:

```json
{ "status": "ok" }
```

Open Swagger at `http://127.0.0.1:8000/docs` and frontend at
`http://localhost:3000`. Submit the Quick Start query. The answer should
contain `[E1]`-style citations and the Evidence panel should contain matching
chunks.

## Admin Observability Console

Open `http://localhost:3000/admin` to run diagnostic queries and inspect the
retrieval, reranking, context-building, and generation timings, output, cited
evidence, and the most recent request traces. The API exposes the current
process's trace buffer at `GET /admin/traces` for this console.

The console can override, for one diagnostic request only, whether reranking is
enabled, the local Ollama model, temperature, and token budget. It never
persists these controls to YAML. Configure the default model in
`configs/labor_rag.yaml`.

The console uses Server-Sent Events (SSE) to show every pipeline transition as
it happens: queued, retrieval, reranking, context selection, generation, and
completion/error. This works completely locally. For an additional hosted
trace dashboard, run `logfire projects new` or `logfire projects use` locally
after installing dependencies; Logfire then uses its local project credential.
`LOGFIRE_TOKEN` is only needed for another machine or deployment. FastAPI
request spans are then exported to Logfire. Do not enable it for legal queries
until Logfire data-scrubbing and retention are approved for your environment.

Logfire is disabled by default. To explicitly enable it, set
`LOGFIRE_ENABLED=true` and restart the backend.

For local development, the backend keeps the latest 200 traces in memory.

## Offline quality evaluation

The evaluation baseline lives in [`evals/`](evals/README.md). It scores routing,
retrieval, reranking, citation validity, abstention, latency, and errors from a
versioned JSONL dataset without calling an external model or service:

```powershell
python -m evals.run_offline --cases evals/datasets/template_cases.jsonl
```

See [`evals/README.md`](evals/README.md) for the dataset fields and the optional
DeepEval faithfulness/relevance judge. The LLM judge is disabled by default.

## Quality guardrails

The pipeline has local, configurable guardrails before retrieval, after context
selection, and after generation. Their primary role is answer quality and
grounding: they can return `out_of_scope`, `refused`, `clarification_required`,
or `insufficient_context` before an unsupported answer is produced.

Configure every stage in `configs/labor_rag.yaml` under `guardrails`:

```yaml
guardrails:
  scope_enabled: true
  sensitive_enabled: true
  ambiguity_enabled: true
  ambiguity_mode: broad_context # or clarify
  evidence_quality_enabled: true
  output_enabled: true
  classifier:
    provider: rules # or jev
    threshold: 0.80
```

`rules` is the offline high-precision baseline. To use a local JEV checkpoint,
set `provider: jev` and `classifier.model` to its local path; the checkpoint
must emit scores for the labels `in_scope`, `out_of_scope`,
`sensitive_or_disallowed`, `ambiguous`, or `unknown`.

To use TypeSafe Jev through OpenRouter's typed Decisions API instead, set
`OPENROUTER_API_KEY` in `.env`, then change `provider` to `openrouter_jev`.
The default model is `typesafe/jev-1.13`; `classifier.model`, `base_url`, and
`timeout` remain configurable under `guardrails.classifier`. No web lookup or
human-review action is triggered automatically: insufficient evidence is only
marked `handoff_eligible` in diagnostics for a future integration.
To mirror trace metadata to Firestore, install the updated dependencies and
configure a Firebase service-account credential outside the repository:

```text
FIRESTORE_TRACE_COLLECTION=rag_request_traces
GOOGLE_APPLICATION_CREDENTIALS=/secure/path/service-account.json
RAG_TRACE_INCLUDE_CONTENT=false
```

Firestore logging is off unless `FIRESTORE_TRACE_COLLECTION` is set. With the
default `RAG_TRACE_INCLUDE_CONTENT=false`, only a query hash is mirrored; the
query content remains in the local in-memory admin trace. Do not expose
`/admin` or `/admin/traces` publicly: protect it with your identity-aware proxy
or application authentication before production deployment.

## Required Runtime Assets

### Required and included

- `src/labor_rag/`: runtime, retrieval, reranking, context, generation, and storage.
- `app/backend/api/`: FastAPI entrypoint and routes.
- `app/frontend/src/`: Next.js UI source.
- `configs/labor_rag.yaml` and `configs/retrieval_queries.json`.
- `data/processed/chunks/labor_law_legal.jsonl`: required chunk corpus.
- `scripts/`, `tests/`, `pyproject.toml`, requirements files, and package manifests.

### Required but downloaded during setup

- Python packages from PyPI.
- Node packages from npm.
- DEk21 and BGE reranker models from Hugging Face.
- OpenRouter API access and an `OPENROUTER_API_KEY` for generation.

### Generated locally

- `__pycache__/`, `*.pyc`, and `*.egg-info/`.
- `app/frontend/node_modules/` and `.next/`.
- Qdrant storage under `data/qdrant/`.
- Logs and retrieval/reranking outputs under `data/logs/`,
  `data/retrieval_results/`, and `data/reranking_results/`.
- `.env` and local `.conda/` environments.

### Not required for default startup

- Re-running PDF parsing, cleaning, or chunking; processed chunks are included.
- Notebook files and empty experimental directories.
- Local Ollama/Gemini runtime packages and credentials.

Raw/extracted documents are useful for corpus maintenance, but are not needed
to start the product after the included chunks have been indexed.

## Common Problems

### `uvicorn` is not recognized

```powershell
conda activate labor
python -m uvicorn app.backend.api.main:app --reload
```

### `No module named app`

Run from `D:\Projects\RAG`, not `app/backend`:

```powershell
cd D:\Projects\RAG
python -m uvicorn app.backend.api.main:app --reload
```

### `No module named labor_rag`

```powershell
python -m pip install -e .
```

### BM25 `data_path` missing

```powershell
Test-Path data/processed/chunks/labor_law_legal.jsonl
Get-Content configs/labor_rag.yaml
```

Do not delete the chunk JSONL; hybrid retrieval requires it.

### Qdrant connection error

```powershell
docker compose ps
Invoke-WebRequest http://localhost:6333/ -UseBasicParsing
```

If the collection is absent, run the indexing command once.

### OpenRouter connection error

```powershell
if (-not $env:OPENROUTER_API_KEY) { Write-Error "OPENROUTER_API_KEY is not configured" }
```

### Model not found

Verify the configured OpenRouter model name and API-key permissions. Hugging
Face embedding/reranking models download automatically; check network access
and disk space if that download fails.

### Frontend CORS error

Use frontend port 3000 and backend port 8000. The backend allows
`localhost:3000` and `127.0.0.1:3000`. If the frontend port changes, update the
allow-list in `app/backend/api/main.py`.

### npm dependency error

Use Node.js 20.9+ and run from `app/frontend`:

```powershell
Remove-Item -Recurse -Force node_modules -ErrorAction SilentlyContinue
npm ci
npm run build
```

### Port 8000 or 3000 is occupied

```powershell
Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
Get-NetTCPConnection -LocalPort 3000 -ErrorAction SilentlyContinue
```

Stop the existing process or deliberately change the port and CORS setting.

## Reproducibility Status

The supported fresh-machine flow is: install dependencies, start pinned
Qdrant, index the included chunks, configure OpenRouter, start backend/frontend, and
query through the browser. No indexing or model cache is assumed to be present.

Python transitive versions, Hugging Face revisions, and OpenRouter model
OS/GPU-specific Torch wheels are not fully locked by this repository. The
frontend lockfile and Qdrant image are pinned; Python requirements use minimum
versions. A byte-for-byte production environment would additionally need a
Conda lockfile generated for the target OS/GPU.
