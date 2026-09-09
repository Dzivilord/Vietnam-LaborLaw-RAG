"""Base contract for learned sparse embedding models.

BM25 does NOT belong here: BM25 is a lexical retrieval/ranking method rather
than a neural sparse embedding model. This package is reserved for future
learned sparse models such as SPLADE-like implementations.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence

from ..base import BaseEmbedder, EmbeddingKind

# Sparse neural models commonly expose token/vocabulary IDs with learned
# non-zero weights. Keeping the representation explicit avoids pretending a
# BM25 inverted index is an embedding vector.
SparseVector = dict[int, float]


class BaseSparseEmbedder(BaseEmbedder):
    """Interface for future learned sparse embedding implementations."""

    @property
    def kind(self) -> EmbeddingKind:
        return EmbeddingKind.SPARSE

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[SparseVector]:
        """Embed document texts into sparse token-weight mappings."""

    @abstractmethod
    def embed_query(self, text: str) -> SparseVector:
        """Embed one query into a sparse token-weight mapping."""
