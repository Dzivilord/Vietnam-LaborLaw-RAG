"""BGE-M3 dense embedding adapter for multilingual semantic retrieval."""

from __future__ import annotations

from .sentence_transformer import SentenceTransformerDenseEmbedder


class BGEM3DenseEmbedder(SentenceTransformerDenseEmbedder):
    """Dense-only adapter for ``BAAI/bge-m3``.

    BGE-M3 can support dense, sparse/lexical and multi-vector retrieval, but
    this adapter intentionally exposes only its dense sentence embedding mode
    so it fits the ``embeddings/dense`` contract used by the current project.

    The model produces 1024-dimensional dense vectors. Query and document
    texts are encoded without an additional retrieval instruction prefix.
    """

    MODEL_NAME = "BAAI/bge-m3"
    OUTPUT_DIMENSION = 1024

    def __init__(
        self,
        *,
        device: str | None = None,
        batch_size: int = 16,
        normalize_embeddings: bool = True,
    ) -> None:
        super().__init__(
            self.MODEL_NAME,
            device=device,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
        )

    @property
    def dimension(self) -> int:
        """Return BGE-M3's fixed dense embedding dimension."""
        return self.OUTPUT_DIMENSION
