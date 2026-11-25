"""Shared CLI argument definitions mapped to runtime configuration.

Keeping these definitions in one place ensures run_rag_pipeline.py and
run_generation.py expose the same retrieval/reranker/context/ambiguity knobs.
The future FastAPI layer will use the config dataclasses directly, not argparse.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .config import (
    AmbiguityRuntimeConfig,
    ContextRuntimeConfig,
    GenerationRuntimeConfig,
    RAGRuntimeConfig,
    RerankerRuntimeConfig,
    RetrievalRuntimeConfig,
)


DEFAULT_BGE_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_QWEN3_RERANKER_MODEL = "Qwen/Qwen3-Reranker-0.6B"
DEFAULT_AMBIGUITY_MODEL = "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"


def add_core_pipeline_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_data_path: Path,
    include_legacy_model_alias: bool = True,
) -> None:
    group = parser.add_argument_group("retrieval")
    group.add_argument("--retrieval", choices=("dense", "bm25", "hybrid"), default="hybrid")
    group.add_argument("--retrieval-top-k", type=int, default=20)
    group.add_argument("--candidate-k", type=int, default=20)
    group.add_argument("--data", type=Path, default=default_data_path)
    group.add_argument("--bm25-k1", type=float, default=1.5)
    group.add_argument("--bm25-b", type=float, default=0.75)
    embedding_flags = ["--embedding-model"]
    if include_legacy_model_alias:
        embedding_flags.append("--model")
    group.add_argument(
        *embedding_flags,
        dest="model",
        choices=("bge", "dek21"),
        default="bge",
        help=(
            "Dense embedding model key"
            + (" (legacy alias: --model)." if include_legacy_model_alias else ".")
        ),
    )
    group.add_argument("--collection", default=None)
    group.add_argument("--qdrant-url", default="http://localhost:6333")
    group.add_argument("--qdrant-api-key", default=None)
    group.add_argument("--device", default=None)
    group.add_argument("--dek21-dimension", type=int, choices=(768, 512, 256), default=768)
    group.add_argument("--rrf-k", type=int, default=60)
    group.add_argument("--dense-weight", type=float, default=1.0)
    group.add_argument("--bm25-weight", type=float, default=1.0)

    group = parser.add_argument_group("reranker")
    group.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=True)
    group.add_argument("--no-rerank-context-k", type=int, default=5)
    group.add_argument("--rerank-top-k", type=int, default=10)
    group.add_argument("--reranker", choices=("bge", "qwen3"), default="bge")
    group.add_argument("--reranker-model", default=None)
    group.add_argument("--reranker-device", default=None)
    group.add_argument("--reranker-batch-size", type=int, default=8)
    group.add_argument("--reranker-max-length", type=int, default=None)
    group.add_argument("--qwen3-instruction", default=None)

    group = parser.add_argument_group("context builder")
    group.add_argument("--min-score", type=float, default=0.60)
    group.add_argument("--relative-to-top", type=float, default=None)
    group.add_argument("--score-gap", type=float, default=None)
    group.add_argument("--max-chunks", type=int, default=8)
    group.add_argument("--max-tokens", type=int, default=3500, help="0 disables the token budget.")
    group.add_argument("--order-by", choices=("legal", "relevance"), default="legal")

    group = parser.add_argument_group("ambiguity")
    group.add_argument("--check-ambiguity", action="store_true")
    group.add_argument("--ambiguity-min-chunks", type=int, default=4)
    group.add_argument("--ambiguity-model", default=DEFAULT_AMBIGUITY_MODEL)
    group.add_argument("--ambiguity-device", default=None)
    group.add_argument("--ambiguity-min-decision-score", type=float, default=0.45)
    group.add_argument("--ambiguity-min-margin", type=float, default=0.08)


def add_generation_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("generation")
    group.add_argument("--provider", choices=("ollama", "openrouter"), default="ollama")
    group.add_argument(
        "--generation-model", "--llm-model", "--model",
        dest="generation_model",
        default=None,
        help=(
            "Generation model override. OpenRouter default is OPENROUTER_MODEL "
            "or the configured provider default."
        ),
    )
    group.add_argument("--temperature", type=float, default=0.2)
    group.add_argument("--max-output-tokens", type=int, default=1200)
    group.add_argument("--allow-missing-citations", action="store_true")
    group.add_argument("--ollama-url", default=None)
    group.add_argument("--ollama-think", action="store_true")
    group.add_argument("--timeout", type=float, default=180.0)


def _resolve_path(path: Path, project_root: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else project_root / path


def runtime_config_from_args(
    args: argparse.Namespace,
    *,
    project_root: Path,
    enable_generation: bool,
) -> RAGRuntimeConfig:
    generation_model = getattr(args, "generation_model", None)
    provider = getattr(args, "provider", "ollama")
    if enable_generation and provider == "ollama" and generation_model is None:
        generation_model = os.getenv("OLLAMA_MODEL", "qwen3.5:4b")

    return RAGRuntimeConfig(
        retrieval=RetrievalRuntimeConfig(
            method=args.retrieval,
            top_k=args.retrieval_top_k,
            candidate_k=args.candidate_k,
            data_path=_resolve_path(args.data, project_root),
            bm25_k1=args.bm25_k1,
            bm25_b=args.bm25_b,
            embedding_model=args.model,
            collection=args.collection,
            qdrant_url=args.qdrant_url,
            qdrant_api_key=args.qdrant_api_key,
            device=args.device,
            dek21_dimension=args.dek21_dimension,
            rrf_k=args.rrf_k,
            dense_weight=args.dense_weight,
            bm25_weight=args.bm25_weight,
        ),
        reranker=RerankerRuntimeConfig(
            enabled=args.rerank,
            type=args.reranker,
            model=args.reranker_model,
            device=args.reranker_device,
            top_k=args.rerank_top_k,
            no_rerank_context_k=args.no_rerank_context_k,
            batch_size=args.reranker_batch_size,
            max_length=args.reranker_max_length,
            qwen3_instruction=args.qwen3_instruction,
        ),
        context=ContextRuntimeConfig(
            min_score=args.min_score,
            relative_to_top=args.relative_to_top,
            score_gap=args.score_gap,
            max_chunks=args.max_chunks,
            max_tokens=None if args.max_tokens == 0 else args.max_tokens,
            order_by=args.order_by,
        ),
        ambiguity=AmbiguityRuntimeConfig(
            enabled=args.check_ambiguity,
            min_relevant_chunks=args.ambiguity_min_chunks,
            model=args.ambiguity_model,
            device=args.ambiguity_device,
            min_decision_score=args.ambiguity_min_decision_score,
            min_decision_margin=args.ambiguity_min_margin,
        ),
        generation=GenerationRuntimeConfig(
            enabled=enable_generation,
            provider=provider,
            model=generation_model,
            temperature=getattr(args, "temperature", 0.2),
            max_output_tokens=getattr(args, "max_output_tokens", 1200),
            require_citations=not getattr(args, "allow_missing_citations", False),
            ollama_url=getattr(args, "ollama_url", None),
            ollama_think=getattr(args, "ollama_think", False),
            timeout=getattr(args, "timeout", 180.0),
        ),
    )
