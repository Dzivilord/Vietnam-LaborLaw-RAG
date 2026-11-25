"""Small Qdrant wrapper used by the RAG project."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from qdrant_client import QdrantClient, models


class QdrantStore:
    """Minimal storage wrapper for one dense vector space per collection."""

    def __init__(
        self,
        collection_name: str,
        *,
        url: str = "http://localhost:6333",
        api_key: str | None = None,
        path: str | None = None,
    ) -> None:
        """``path`` selects Qdrant's embedded local mode (no server, e.g. on
        Kaggle); otherwise ``url`` points at a running Qdrant server."""
        if not collection_name.strip():
            raise ValueError("collection_name must not be empty.")

        self.collection_name = collection_name
        if path:
            self.client = QdrantClient(path=str(path))
        else:
            self.client = QdrantClient(
                url=url,
                api_key=api_key,
            )

    def collection_exists(self) -> bool:
        """Return True when the configured collection already exists."""
        return self.client.collection_exists(self.collection_name)

    def create_collection(
        self,
        *,
        vector_size: int,
        recreate: bool = False,
    ) -> None:
        """Create a cosine-similarity collection for one dense vector space."""
        if vector_size <= 0:
            raise ValueError("vector_size must be greater than 0.")

        exists = self.collection_exists()

        if recreate and exists:
            self.client.delete_collection(self.collection_name)
            exists = False

        if exists:
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )

    def upsert(
        self,
        *,
        ids: Sequence[str],
        vectors: Sequence[Sequence[float]],
        payloads: Sequence[dict[str, Any]],
    ) -> None:
        """Insert/update dense vectors and their payloads."""
        if not (len(ids) == len(vectors) == len(payloads)):
            raise ValueError(
                "ids, vectors and payloads must have the same length."
            )

        points = []

        for chunk_id, vector, payload in zip(ids, vectors, payloads):
            vector_list = (
                vector.tolist()
                if hasattr(vector, "tolist")
                else [float(value) for value in vector]
            )

            qdrant_id = self._to_qdrant_id(chunk_id)

            point_payload = dict(payload)
            point_payload["chunk_id"] = chunk_id

            points.append(
                models.PointStruct(
                    id=qdrant_id,
                    vector=vector_list,
                    payload=point_payload,
                )
            )

        if not points:
            return

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

    def search(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Return the nearest stored chunks for one dense query vector."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        vector = (
            query_vector.tolist()
            if hasattr(query_vector, "tolist")
            else [float(value) for value in query_vector]
        )

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

        return [
            {
                "id": str(point.id),
                "score": float(point.score),
                "payload": dict(point.payload or {}),
            }
            for point in response.points
        ]

    def delete(self, ids: Sequence[str]) -> None:
        """Delete points using the application's chunk IDs."""
        if not ids:
            return

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(
                points=[self._to_qdrant_id(chunk_id) for chunk_id in ids]
            ),
            wait=True,
        )

    @staticmethod
    def _to_qdrant_id(chunk_id: str) -> str:
        """Convert an arbitrary chunk ID into a stable Qdrant UUID."""
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError("chunk_id must be a non-empty string.")

        try:
            return str(UUID(chunk_id))
        except ValueError:
            return str(uuid5(NAMESPACE_URL, chunk_id))
