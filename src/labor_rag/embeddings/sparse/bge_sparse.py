"""BGE-M3 sparse embedding adapter for learned lexical retrieval."""

from __future__ import annotations

from collections.abc import Sequence

from .base import BaseSparseEmbedder, SparseVector


class BGEM3SparseEmbedder(BaseSparseEmbedder):
    """Sparse-only adapter for ``BAAI/bge-m3``.

    BGE-M3 supports dense, sparse lexical, and multi-vector representations.
    This adapter intentionally exposes only the learned sparse lexical weights
    so it fits the ``embeddings/sparse`` contract.

    The returned sparse vector is a mapping:

        {token_id: learned_weight}

    BM25 is intentionally not implemented here because BM25 is a lexical
    ranking/retrieval method, not a learned sparse embedding model.
    """

    MODEL_NAME = "BAAI/bge-m3"

    def __init__(
        self,
        *,
        device: str | None = None,
        batch_size: int = 16,
        max_length: int = 8192,
        use_fp16: bool = True,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0.")
        if max_length <= 0:
            raise ValueError("max_length must be greater than 0.")

        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:
            raise ImportError(
                "BGEM3SparseEmbedder requires FlagEmbedding. "
                "Install it with: pip install -U FlagEmbedding"
            ) from exc

        model_kwargs: dict[str, object] = {
            "use_fp16": use_fp16,
        }

        if device is not None:
            model_kwargs["devices"] = [device]

        self._model = BGEM3FlagModel(
            self.MODEL_NAME,
            **model_kwargs,
        )
        self._batch_size = batch_size
        self._max_length = max_length

    @property
    def model_name(self) -> str:
        return self.MODEL_NAME

    def embed_documents(self, texts: Sequence[str]) -> list[SparseVector]:
        """Embed document/chunk texts into sparse token-weight mappings."""
        normalized = [self._validate_text(text) for text in texts]

        if not normalized:
            return []

        output = self._model.encode(
            normalized,
            batch_size=self._batch_size,
            max_length=self._max_length,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        return [
            self._to_sparse_vector(weights)
            for weights in output["lexical_weights"]
        ]

    def embed_query(self, text: str) -> SparseVector:
        """Embed one query into a sparse token-weight mapping."""
        normalized = self._validate_text(text)

        output = self._model.encode(
            [normalized],
            batch_size=1,
            max_length=self._max_length,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        return self._to_sparse_vector(
            output["lexical_weights"][0]
        )

    @staticmethod
    def _to_sparse_vector(weights: dict) -> SparseVector:
        """Convert FlagEmbedding lexical weights to canonical sparse format."""
        vector: SparseVector = {}

        for token_id, weight in weights.items():
            value = float(weight)
            if value > 0.0:
                vector[int(token_id)] = value

        return vector

    @staticmethod
    def _validate_text(text: str) -> str:
        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        text = text.strip()
        if not text:
            raise ValueError("text must not be empty.")

        return text