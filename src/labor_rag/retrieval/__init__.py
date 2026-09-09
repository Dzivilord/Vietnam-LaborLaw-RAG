"""Retrieval contracts and lazily-loaded implementations.

Keeping heavy backend imports lazy prevents unrelated code (schemas, BM25-only
runs, tests, future FastAPI startup) from requiring Qdrant just to import the
retrieval package.
"""

from .base import BaseRetriever, RetrievalResult

__all__ = ["BaseRetriever", "DenseRetriever", "RetrievalResult"]


def __getattr__(name: str):
    if name == "DenseRetriever":
        from .dense_retriever import DenseRetriever

        return DenseRetriever
    raise AttributeError(name)
