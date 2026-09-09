"""Generation layer for the labor-law RAG."""

from .generator import RAGGenerator
from .schemas import GenerationConfig, GenerationRequest, GenerationResult

__all__ = [
    "GenerationConfig",
    "GenerationRequest",
    "GenerationResult",
    "RAGGenerator",
]
