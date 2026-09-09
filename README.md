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
	`-- generation via Ollama

Qdrant <--- dense vectors and original chunk payloads
Hugging Face <--- embedding and reranker model downloads
Ollama <--- qwen3.5:4b generation model
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
response loads the local model into memory.

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
| Generation      | `qwen3.5:4b`                          | Ollama pull                     |

```powershell
ollama serve
ollama pull qwen3.5:4b
ollama list
```

If Ollama is already running as a desktop/service application, do not start a
second `ollama serve`. The provider calls
`http://localhost:11434/api/chat`.

The active configuration does not use Gemini. The optional Gemini provider
would additionally require `google-genai`, `GEMINI_API_KEY`, and a changed
provider setting; do not install it for the default path.

## Configuration

`configs/labor_rag.yaml` currently specifies hybrid retrieval, BM25 data at
`data/processed/chunks/labor_law_legal.jsonl`, collection
`labor_law_dek21_768`, DEk21 768-dimensional embeddings, BGE reranking, Ollama
generation with `qwen3.5:4b`, and ambiguity checking disabled.

Qdrant is currently configured as the code default `http://localhost:6333`, not
as an environment variable. Ollama supports these optional process variables:

```text
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3.5:4b
```

Create local configuration without committing secrets:

```powershell
Copy-Item .env.example .env
```

The backend does not implicitly load `.env`. `scripts/dev.ps1` passes
`--env-file .env` when that file exists; for manual startup use the same option
or set variables in the PowerShell session. The current YAML explicitly sets
the generation model to `qwen3.5:4b`, so `OLLAMA_MODEL` is only relevant after
changing that config or using the provider outside the current YAML path.

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

If you created `.env` and want Uvicorn to load it, append
`--env-file .env`.

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
- Ollama `qwen3.5:4b`.

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
- Gemini packages/credentials when using Ollama.

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

### Ollama connection error

```powershell
ollama list
Invoke-WebRequest http://localhost:11434/api/tags -UseBasicParsing
```

### Model not found

Run `ollama pull qwen3.5:4b`. Hugging Face models download automatically; check
network access and disk space if that download fails.

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
Qdrant, index the included chunks, pull Ollama, start backend/frontend, and
query through the browser. No indexing or model cache is assumed to be present.

Python transitive versions, Hugging Face revisions, Ollama revisions, and
OS/GPU-specific Torch wheels are not fully locked by this repository. The
frontend lockfile and Qdrant image are pinned; Python requirements use minimum
versions. A byte-for-byte production environment would additionally need a
Conda lockfile generated for the target OS/GPU.
