"""Shared retrieval contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RetrievalResult:
    """One chunk returned by a retriever."""

    chunk_id: str
    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseRetriever(ABC):
    """Common interface implemented by all retrieval strategies."""

    @abstractmethod
    def retrieve(self, query: str, *, top_k: int = 5) -> list[RetrievalResult]:
        """Return the top-k chunks for one query."""
