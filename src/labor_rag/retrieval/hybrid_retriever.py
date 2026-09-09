"""Hybrid dense + BM25 retrieval using Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

from dataclasses import dataclass

from .base import BaseRetriever, RetrievalResult


@dataclass(slots=True)
class _FusionItem:
    result: RetrievalResult
    rrf_score: float = 0.0
    dense_rank: int | None = None
    dense_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None


class HybridRetriever(BaseRetriever):
    """Fuse dense and BM25 rankings with weighted Reciprocal Rank Fusion."""

    def __init__(
        self,
        *,
        dense_retriever: BaseRetriever,
        bm25_retriever: BaseRetriever,
        candidate_k: int = 20,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ) -> None:
        if candidate_k <= 0:
            raise ValueError("candidate_k must be greater than 0.")
        if rrf_k < 0:
            raise ValueError("rrf_k must be non-negative.")
        if dense_weight < 0 or bm25_weight < 0:
            raise ValueError("fusion weights must be non-negative.")
        if dense_weight == 0 and bm25_weight == 0:
            raise ValueError("at least one fusion weight must be positive.")

        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self.dense_weight = float(dense_weight)
        self.bm25_weight = float(bm25_weight)

    def retrieve(self, query: str, *, top_k: int = 5) -> list[RetrievalResult]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        candidate_k = max(self.candidate_k, top_k)
        dense_results = self.dense_retriever.retrieve(query, top_k=candidate_k)
        bm25_results = self.bm25_retriever.retrieve(query, top_k=candidate_k)

        fused: dict[str, _FusionItem] = {}

        for rank, result in enumerate(dense_results, start=1):
            item = fused.setdefault(
                result.chunk_id,
                _FusionItem(result=result),
            )
            item.rrf_score += self.dense_weight / (self.rrf_k + rank)
            item.dense_rank = rank
            item.dense_score = result.score

        for rank, result in enumerate(bm25_results, start=1):
            item = fused.setdefault(
                result.chunk_id,
                _FusionItem(result=result),
            )
            item.rrf_score += self.bm25_weight / (self.rrf_k + rank)
            item.bm25_rank = rank
            item.bm25_score = result.score

        ordered = sorted(
            fused.values(),
            key=lambda item: item.rrf_score,
            reverse=True,
        )[:top_k]

        results: list[RetrievalResult] = []
        for item in ordered:
            metadata = dict(item.result.metadata)
            metadata.update(
                {
                    "retrieval_method": "hybrid_rrf",
                    "dense_rank": item.dense_rank,
                    "dense_score": item.dense_score,
                    "bm25_rank": item.bm25_rank,
                    "bm25_score": item.bm25_score,
                    "rrf_score": item.rrf_score,
                }
            )

            results.append(
                RetrievalResult(
                    chunk_id=item.result.chunk_id,
                    content=item.result.content,
                    score=item.rrf_score,
                    metadata=metadata,
                )
            )

        return results
