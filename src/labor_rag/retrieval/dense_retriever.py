"""Dense vector retrieval against Qdrant."""

from __future__ import annotations

from labor_rag.embeddings.dense.base import BaseDenseEmbedder
from labor_rag.storage.qdrant import QdrantStore

from .base import BaseRetriever, RetrievalResult


class DenseRetriever(BaseRetriever):
    """Retrieve chunks with a dense query embedding."""

    def __init__(
        self,
        *,
        embedder: BaseDenseEmbedder,
        store: QdrantStore,
    ) -> None:
        self.embedder = embedder
        self.store = store

    def retrieve(self, query: str, *, top_k: int = 5) -> list[RetrievalResult]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        query_vector = self.embedder.embed_query(query.strip())
        points = self.store.search(query_vector, top_k=top_k)

        results: list[RetrievalResult] = []

        for point in points:
            payload = dict(point.get("payload") or {})
            chunk_id = str(payload.get("chunk_id") or point.get("id") or "")
            content = str(
                payload.get("content")
                or payload.get("embedding_text")
                or ""
            )

            metadata = dict(payload)
            metadata["retrieval_method"] = "dense"

            results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    content=content,
                    score=float(point.get("score", 0.0)),
                    metadata=metadata,
                )
            )

        return results
