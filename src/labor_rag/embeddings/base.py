"""Shared interfaces for embedding implementations.

This module intentionally contains only concepts shared by all embedding
representations. Dense and sparse embeddings have different output shapes,
so their representation-specific contracts live in ``dense/base.py`` and
``sparse/base.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum


class EmbeddingKind(str, Enum):
    """Representation produced by an embedder."""

    DENSE = "dense"
    SPARSE = "sparse"


class BaseEmbedder(ABC):
    """Minimal common contract for every embedding adapter."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Stable model identifier used for logging and index metadata."""

    @property
    @abstractmethod
    def kind(self) -> EmbeddingKind:
        """Embedding representation produced by this adapter."""
