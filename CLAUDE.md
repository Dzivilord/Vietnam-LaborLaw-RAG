# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Vietnamese labor-law RAG system: Next.js frontend (`app/frontend`) → FastAPI backend (`app/backend/api`) → `labor_rag` package (`src/labor_rag`, installed editable as `labor-law-rag`). Retrieval uses Qdrant (dense) + in-memory BM25 over a processed chunk JSONL; reranking with BGE; generation via local Ollama (default) or OpenRouter. Windows/PowerShell is the primary dev environment; conda env is `labor`.

**Always run Python commands from the repo root** (`D:\Projects\RAG`) — imports like `app.backend.api.main` and relative paths in `configs/labor_rag.yaml` depend on it.

## Commands

```powershell
# Setup
conda activate labor
python -m pip install -r requirements.txt; python -m pip install -e .
cd app/frontend; npm ci; cd ../..

# Infra (Qdrant v1.19.0, storage in ./data/qdrant)
docker compose up -d qdrant
ollama pull qwen3.5:4b

# One-time index of the included chunks into Qdrant (do NOT re-chunk; avoid --recreate unless intentional)
python scripts/index_labor_law_chunks.py --data data/processed/chunks/labor_law_legal.jsonl --model dek21 --collection labor_law_dek21_768 --qdrant-url http://localhost:6333

# Run
python -m uvicorn app.backend.api.main:app --reload     # backend :8000 (Swagger at /docs, /health)
cd app/frontend; npm run dev                            # frontend :3000 (/admin = observability console)
.\scripts\dev.ps1                                       # starts Qdrant + backend + frontend windows; never indexes

# Frontend build (also the closest thing to a type check; no lint script exists)
cd app/frontend; npm run build

# Tests (pytest; unit tests use stub retrievers/rerankers/providers, no services needed)
python -m pytest tests
python -m pytest tests/test_guardrails.py::test_pipeline_returns_out_of_scope_before_retrieval

# Evals (see evals/README.md)
python -m pytest evals                                   # deterministic metric + dataset tests
python -m evals.validate_golden_dataset                  # golden set vs current corpus chunk IDs
python -m evals.run_offline --cases evals/datasets/template_cases.jsonl   # fully offline scoring
python -m evals.run_golden --limit 2 --no-generate       # real pipeline smoke test (needs Qdrant + models)
python -m evals.run_router_v2 --split dev                 # phase 1: router only (cached classifier calls)
python -m evals.run_rag --split dev --no-generate         # phase 2: RAG on answerable cases (input guardrail skipped)
python -m evals.sweep_context --run evals/runs/rag-<id>.jsonl   # offline context-threshold calibration
python -m evals.run_ragas --run evals/runs/rag-<id>.jsonl       # RAGAS judge (requirements-eval.txt)
```

## Architecture

**Config → factory → pipeline.** `configs/labor_rag.yaml` is loaded by `RAGRuntimeConfig.from_yaml` ([src/labor_rag/runtime/config.py](src/labor_rag/runtime/config.py)), which remaps YAML keys onto dataclasses (e.g. `retrieval.type` → `method`, top-level `embedding.model` → `retrieval.embedding_model`; any `embedding.model` value other than `bge` silently selects the `dek21` embedder). Unset fields fall back to dataclass defaults — the YAML is intentionally sparse. [runtime/factory.py](src/labor_rag/runtime/factory.py) validates the config and builds every component (embedder, Qdrant store, BM25/dense/hybrid retriever, reranker, context builder, ambiguity classifier, guardrails, LLM provider). Add new component variants there, behind a config `Literal`.

**Request flow.** FastAPI `lifespan` in [app/backend/api/main.py](app/backend/api/main.py) loads `.env`, builds a `ModelManager` (loads models, builds BM25 index at startup) and a `RAGRuntime` ([runtime/rag_runtime.py](src/labor_rag/runtime/rag_runtime.py)) — the stable entry point used by both API routes and CLI. The runtime calls `LaborLawRAGPipeline` ([pipelines/application_pipeline.py](src/labor_rag/pipelines/application_pipeline.py)), which wraps:
1. input guardrails (scope / sensitive / ambiguity) — can return `out_of_scope`, `refused`, `clarification_required` before retrieval runs;
2. `RetrievalContextPipeline` ([pipelines/rag_pipeline.py](src/labor_rag/pipelines/rag_pipeline.py)): hybrid retrieval (dense + BM25 fused via weighted RRF) → rerank → `RerankedContextBuilder` (score thresholds, token/chunk budget, legal-hierarchy ordering);
3. evidence-quality guardrail → `insufficient_context` (marked `handoff_eligible`);
4. `RAGGenerator` with `[E1]`-style evidence citations → output guardrail (citation validity) → may return `abstained`.

The context builder actually used is `labor_rag/reranking/context_builder.py`; `retrieval/context_builder.py` is a separate older module.

**Routes.** `POST /chat` (chat.py) for the main UI. `admin.py` exposes `GET /admin/traces` (in-memory ring buffer of the last 200 traces, `RequestTraceRecorder`), `POST /admin/runs` + `GET /admin/runs/{id}/events` (SSE stream of pipeline stage events for the `/admin` console; per-request overrides of rerank/model/temperature/tokens are never persisted). Frontend API calls and response types live in [app/frontend/src/services/api.ts](app/frontend/src/services/api.ts); `src/lib/status.ts` maps pipeline `status` values to UI messages. The main page (`src/app/page.tsx`) and `/admin` have separate stylesheets (`globals.css`, `admin/admin.css`).

**Observability.** Logfire is opt-in (`LOGFIRE_ENABLED=true`); spans via `observability/logfire_spans.py`. Firestore trace mirroring is enabled only when `FIRESTORE_TRACE_COLLECTION` is set; with `RAG_TRACE_INCLUDE_CONTENT=false` (default) only a query hash leaves the machine.

**Data / offline pipeline.** `data/processed/chunks/labor_law_legal.jsonl` is the required corpus for both BM25 and indexing — don't delete or regenerate it. `scripts/` contains the one-off ingestion chain (load_pdf → clean_text → parse → chunk → index) plus standalone retrieve/rerank/generate scripts; only indexing is needed on a fresh machine. Qdrant collection name must match embedder dimension (`labor_law_dek21_768`).

## Gotchas

- The backend auto-loads the root `.env` via `load_dotenv(override=False)` in `main.py`; process env vars take precedence over `.env`.
- Changing the frontend port requires updating the CORS allow-list in `main.py`.
- Guardrail classifier providers: `rules` (offline default), `jev` (local HF checkpoint, needs `classifier.model`), `openrouter_jev` (needs `OPENROUTER_API_KEY`).
- `reranker.top_k` and `no_rerank_context_k` must not exceed `retrieval.top_k` — `validate_runtime_config` enforces this at startup.
- `/admin` and `/admin/traces` have no auth; don't expose publicly.
