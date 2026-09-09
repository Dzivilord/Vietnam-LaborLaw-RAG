"""Base contract for dense embedding models."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence

import numpy as np

from ..base import BaseEmbedder, EmbeddingKind


class BaseDenseEmbedder(BaseEmbedder):
    """Interface used by dense indexing and dense retrieval.

    Implementations must return float32 NumPy arrays:
    - ``embed_documents`` -> shape ``(n_documents, dimension)``
    - ``embed_query`` -> shape ``(dimension,)``
    """

    @property
    def kind(self) -> EmbeddingKind:
        return EmbeddingKind.DENSE

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimension of vectors exposed to downstream components."""

    @abstractmethod
    def embed_documents(
        self,
        texts: Sequence[str],
        *,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Embed document/chunk texts into a dense matrix."""

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """Embed one user query into a dense vector."""
