"""Reusable Sentence Transformers implementation for dense embedders."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .base import BaseDenseEmbedder


class SentenceTransformerDenseEmbedder(BaseDenseEmbedder):
    """Generic dense embedder backed by ``sentence-transformers``.

    Model-specific adapters can override ``_prepare_query``,
    ``_prepare_documents`` and ``_postprocess_embeddings`` without
    duplicating batching/device/model-loading logic.
    """

    def __init__(
        self,
        model_name: str,
        *,
        device: str | None = None,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

        self._model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self._model: Any | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def _get_model(self) -> Any:
        """Lazily load the model so importing the package stays lightweight."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise ImportError(
                    "sentence-transformers is required for dense embedding. "
                    "Install it with: pip install sentence-transformers"
                ) from exc

            kwargs: dict[str, Any] = {}
            if self.device is not None:
                kwargs["device"] = self.device

            self._model = SentenceTransformer(self.model_name, **kwargs)

        return self._model

    @property
    def dimension(self) -> int:
        dimension = self._get_model().get_sentence_embedding_dimension()
        if dimension is None:
            raise RuntimeError(
                f"Could not determine embedding dimension for {self.model_name!r}"
            )
        return int(dimension)

    def _prepare_query(self, text: str) -> str:
        return text

    def _prepare_documents(self, texts: Sequence[str]) -> list[str]:
        return list(texts)

    def _postprocess_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        return np.asarray(embeddings, dtype=np.float32)

    def embed_documents(
        self,
        texts: Sequence[str],
        *,
        batch_size: int | None = None,
    ) -> np.ndarray:
        prepared = self._prepare_documents(texts)

        if not prepared:
            return np.empty((0, self.dimension), dtype=np.float32)

        embeddings = self._get_model().encode(
            prepared,
            batch_size=batch_size or self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        return self._postprocess_embeddings(embeddings)

    def embed_query(self, text: str) -> np.ndarray:
        prepared = self._prepare_query(text)
        embedding = self._get_model().encode(
            prepared,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        return self._postprocess_embeddings(embedding)
