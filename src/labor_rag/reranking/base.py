"""Shared reranking contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from labor_rag.retrieval.base import RetrievalResult


class BaseReranker(ABC):
    """Common interface implemented by all reranking strategies.

    A reranker does not search the corpus again. It receives a small candidate
    set produced by retrieval, computes a stronger query-document relevance
    score, and returns the candidates in a new order.
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        *,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Rerank retrieval candidates for one query.

        Args:
            query: Original user query.
            candidates: Results returned by a retriever.
            top_k: Number of reranked results to return. ``None`` keeps all.

        Returns:
            New ``RetrievalResult`` objects sorted by reranker relevance score.
            ``result.score`` is the reranker score; the original retrieval
            score should be preserved in ``result.metadata`` by implementations.
        """
