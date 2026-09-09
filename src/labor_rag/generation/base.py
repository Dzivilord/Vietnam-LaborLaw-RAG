"""Shared contracts for LLM generation providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ProviderResponse:
    """Normalized response returned by every LLM provider adapter."""

    text: str
    provider: str
    model: str
    usage: dict[str, Any] = field(default_factory=dict)
    raw_metadata: dict[str, Any] = field(default_factory=dict)


class BaseLLMProvider(ABC):
    """Minimal provider interface used by :class:`RAGGenerator`.

    Provider adapters only know how to send a system/user prompt to a model.
    Retrieval, reranking, context selection, citation policy, and answer
    validation stay outside the provider implementation.
    """

    provider_name: str
    model_name: str

    @abstractmethod
    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int,
    ) -> ProviderResponse:
        """Generate one answer from already-built RAG context."""
