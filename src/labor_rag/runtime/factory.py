"""Factories that compose configurable RAG components from runtime settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import RAGRuntimeConfig


DEFAULT_BGE_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_QWEN3_RERANKER_MODEL = "Qwen/Qwen3-Reranker-0.6B"


def default_collection_name(config: RAGRuntimeConfig) -> str:
    retrieval = config.retrieval
    if retrieval.embedding_model == "dek21":
        return f"labor_law_dek21_{retrieval.dek21_dimension}"
    return "labor_law_bge_m3_dense"


def resolved_reranker_model(config: RAGRuntimeConfig) -> str:
    reranker = config.reranker
    if reranker.model:
        return reranker.model
    return (
        DEFAULT_QWEN3_RERANKER_MODEL
        if reranker.type == "qwen3"
        else DEFAULT_BGE_RERANKER_MODEL
    )


def resolved_reranker_max_length(config: RAGRuntimeConfig) -> int:
    reranker = config.reranker
    if reranker.max_length is not None:
        return reranker.max_length
    return 2048 if reranker.type == "qwen3" else 512


def validate_runtime_config(config: RAGRuntimeConfig) -> None:
    r = config.retrieval
    rr = config.reranker
    c = config.context
    a = config.ambiguity
    g = config.generation

    if r.top_k <= 0 or r.candidate_k <= 0:
        raise ValueError("retrieval top_k/candidate_k must be greater than 0.")
    if r.bm25_k1 <= 0 or not 0.0 <= r.bm25_b <= 1.0:
        raise ValueError("Invalid BM25 k1/b configuration.")
    if r.rrf_k < 0 or r.dense_weight < 0 or r.bm25_weight < 0:
        raise ValueError("Invalid RRF configuration.")
    if r.method == "hybrid" and r.dense_weight == 0 and r.bm25_weight == 0:
        raise ValueError("Hybrid retrieval requires a positive branch weight.")
    if rr.top_k <= 0 or rr.no_rerank_context_k <= 0 or rr.batch_size <= 0:
        raise ValueError("Invalid reranker top-k/batch configuration.")
    if rr.top_k > r.top_k:
        raise ValueError("reranker.top_k cannot exceed retrieval.top_k.")
    if rr.no_rerank_context_k > r.top_k:
        raise ValueError("no_rerank_context_k cannot exceed retrieval.top_k.")
    if rr.max_length is not None and rr.max_length <= 0:
        raise ValueError("reranker.max_length must be greater than 0.")
    if not 0.0 <= c.min_score <= 1.0:
        raise ValueError("context.min_score must be between 0 and 1.")
    if c.relative_to_top is not None and not 0.0 <= c.relative_to_top <= 1.0:
        raise ValueError("context.relative_to_top must be between 0 and 1.")
    if c.score_gap is not None and c.score_gap < 0:
        raise ValueError("context.score_gap must be non-negative.")
    if c.max_chunks <= 0 or (c.max_tokens is not None and c.max_tokens <= 0):
        raise ValueError("Invalid context budget.")
    if a.min_relevant_chunks <= 0:
        raise ValueError("ambiguity.min_relevant_chunks must be greater than 0.")
    if not 0.0 <= a.min_decision_score <= 1.0:
        raise ValueError("ambiguity.min_decision_score must be between 0 and 1.")
    if not 0.0 <= a.min_decision_margin <= 1.0:
        raise ValueError("ambiguity.min_decision_margin must be between 0 and 1.")
    if not 0.0 <= g.temperature <= 2.0 or g.max_output_tokens <= 0:
        raise ValueError("Invalid generation temperature/token configuration.")
    if g.timeout <= 0:
        raise ValueError("generation.timeout must be greater than 0.")


def _build_embedder(config: RAGRuntimeConfig):
    retrieval = config.retrieval
    if retrieval.embedding_model == "dek21":
        from labor_rag.embeddings.dense.dek21 import DEk21V2Embedder

        return DEk21V2Embedder(
            dimension=retrieval.dek21_dimension,
            device=retrieval.device,
            normalize_embeddings=True,
        )

    from labor_rag.embeddings.dense.bge import BGEM3DenseEmbedder

    return BGEM3DenseEmbedder(
        device=retrieval.device,
        normalize_embeddings=True,
    )


def _build_dense_retriever(config: RAGRuntimeConfig):
    from labor_rag.retrieval.dense_retriever import DenseRetriever
    from labor_rag.storage.qdrant import QdrantStore

    retrieval = config.retrieval
    collection = retrieval.collection or default_collection_name(config)
    embedder = _build_embedder(config)
    store = QdrantStore(
        collection_name=collection,
        url=retrieval.qdrant_url,
        api_key=retrieval.qdrant_api_key,
    )
    return DenseRetriever(embedder=embedder, store=store), embedder, collection


def _build_bm25_retriever(config: RAGRuntimeConfig):
    from labor_rag.retrieval.bm25_retriever import BM25Retriever

    retrieval = config.retrieval
    if retrieval.data_path is None:
        raise ValueError("BM25/hybrid retrieval requires retrieval.data_path.")
    return BM25Retriever.from_jsonl(
        Path(retrieval.data_path),
        k1=retrieval.bm25_k1,
        b=retrieval.bm25_b,
    )


def build_retriever(config: RAGRuntimeConfig):
    retrieval = config.retrieval
    if retrieval.method == "bm25":
        return _build_bm25_retriever(config), None, None

    dense, embedder, collection = _build_dense_retriever(config)
    if retrieval.method == "dense":
        return dense, embedder, collection

    from labor_rag.retrieval.hybrid_retriever import HybridRetriever

    bm25 = _build_bm25_retriever(config)
    hybrid = HybridRetriever(
        dense_retriever=dense,
        bm25_retriever=bm25,
        candidate_k=retrieval.candidate_k,
        rrf_k=retrieval.rrf_k,
        dense_weight=retrieval.dense_weight,
        bm25_weight=retrieval.bm25_weight,
    )
    return hybrid, embedder, collection


def build_reranker(config: RAGRuntimeConfig):
    reranker = config.reranker
    if not reranker.enabled:
        return None

    model = resolved_reranker_model(config)
    max_length = resolved_reranker_max_length(config)

    if reranker.type == "qwen3":
        from labor_rag.reranking.qwen3 import Qwen3Reranker

        kwargs: dict[str, Any] = {
            "model_name": model,
            "device": reranker.device,
            "batch_size": reranker.batch_size,
            "max_length": max_length,
            "use_fp16": True,
        }
        if reranker.qwen3_instruction:
            kwargs["instruction"] = reranker.qwen3_instruction
        return Qwen3Reranker(**kwargs)

    from labor_rag.reranking.bge_reranker import BGEReranker

    return BGEReranker(
        model_name=model,
        device=reranker.device,
        batch_size=reranker.batch_size,
        max_length=max_length,
        normalize_scores=True,
        use_fp16=True,
    )


def build_context_builder(config: RAGRuntimeConfig):
    from labor_rag.reranking.context_builder import (
        ContextBuilderConfig,
        RerankedContextBuilder,
    )

    context = config.context
    return RerankedContextBuilder(
        ContextBuilderConfig(
            min_score=context.min_score,
            min_relative_to_top=context.relative_to_top,
            score_gap=context.score_gap,
            max_chunks=context.max_chunks,
            max_tokens=context.max_tokens,
            order_by=context.order_by,
        )
    )


def build_ambiguity_classifier(config: RAGRuntimeConfig):
    ambiguity = config.ambiguity
    if not ambiguity.enabled:
        return None

    from labor_rag.query_understanding.ambiguity_classifier import (
        QueryAmbiguityClassifier,
        QueryAmbiguityConfig,
    )

    return QueryAmbiguityClassifier(
        QueryAmbiguityConfig(
            model_name=ambiguity.model,
            device=ambiguity.device,
            min_decision_score=ambiguity.min_decision_score,
            min_decision_margin=ambiguity.min_decision_margin,
            max_titles=config.context.max_chunks,
            include_legal_reference=True,
        )
    )


def build_retrieval_context_pipeline(config: RAGRuntimeConfig):
    from labor_rag.pipelines.rag_pipeline import (
        RetrievalContextPipeline,
        RetrievalContextPipelineConfig,
    )

    validate_runtime_config(config)
    retriever, embedder, collection = build_retriever(config)
    pipeline = RetrievalContextPipeline(
        retriever=retriever,
        reranker=build_reranker(config),
        context_builder=build_context_builder(config),
        ambiguity_classifier=build_ambiguity_classifier(config),
        config=RetrievalContextPipelineConfig(
            retrieval_top_k=config.retrieval.top_k,
            rerank_top_k=config.reranker.top_k,
            enable_rerank=config.reranker.enabled,
            no_rerank_context_k=config.reranker.no_rerank_context_k,
            enable_ambiguity_check=config.ambiguity.enabled,
            ambiguity_check_min_chunks=config.ambiguity.min_relevant_chunks,
        ),
    )
    return pipeline, embedder, collection


def build_generator(config: RAGRuntimeConfig):
    generation = config.generation
    if not generation.enabled:
        return None

    from labor_rag.generation import GenerationConfig, RAGGenerator
    from labor_rag.generation.providers import GeminiProvider, OllamaProvider

    if generation.provider == "gemini":
        provider = GeminiProvider(model=generation.model)
    else:
        provider = OllamaProvider(
            model=generation.model,
            base_url=generation.ollama_url,
            timeout=generation.timeout,
            think=generation.ollama_think,
        )

    return RAGGenerator(
        provider=provider,
        config=GenerationConfig(
            temperature=generation.temperature,
            max_output_tokens=generation.max_output_tokens,
            require_citations=generation.require_citations,
        ),
    )


def build_application_pipeline(config: RAGRuntimeConfig):
    from labor_rag.pipelines.application_pipeline import LaborLawRAGPipeline

    retrieval_pipeline, embedder, collection = build_retrieval_context_pipeline(config)
    return (
        LaborLawRAGPipeline(
            retrieval_pipeline=retrieval_pipeline,
            generator=build_generator(config),
        ),
        embedder,
        collection,
    )
