"""DEk21_hcmute_embedding_v2 adapter for Vietnamese semantic retrieval."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .sentence_transformer import SentenceTransformerDenseEmbedder


class DEk21V2Embedder(SentenceTransformerDenseEmbedder):
    """Dense adapter for ``huyydangg/DEk21_hcmute_embedding_v2``.

    The model card recommends Vietnamese word segmentation and documents
    Matryoshka-friendly dimensions of 768, 512 and 256. The default remains
    the full 768 dimensions so quality is not reduced before benchmarking.
    """

    MODEL_NAME = "huyydangg/DEk21_hcmute_embedding_v2"
    SUPPORTED_DIMENSIONS = (768, 512, 256)

    def __init__(
        self,
        *,
        dimension: int = 768,
        word_segment: bool = True,
        device: str | None = None,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
    ) -> None:
        if dimension not in self.SUPPORTED_DIMENSIONS:
            raise ValueError(
                f"dimension must be one of {self.SUPPORTED_DIMENSIONS}, "
                f"got {dimension}"
            )

        super().__init__(
            self.MODEL_NAME,
            device=device,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
        )
        self._output_dimension = dimension
        self.word_segment = word_segment

    @property
    def dimension(self) -> int:
        return self._output_dimension

    def _segment(self, text: str) -> str:
        if not self.word_segment:
            return text

        try:
            from pyvi import ViTokenizer
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "pyvi is required when word_segment=True. "
                "Install it with: pip install pyvi"
            ) from exc

        return ViTokenizer.tokenize(text)

    def _prepare_query(self, text: str) -> str:
        return self._segment(text)

    def _prepare_documents(self, texts: Sequence[str]) -> list[str]:
        return [self._segment(text) for text in texts]

    def _postprocess_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        result = np.asarray(embeddings, dtype=np.float32)

        if result.shape[-1] < self._output_dimension:
            raise RuntimeError(
                f"Model returned dimension {result.shape[-1]}, which is smaller "
                f"than requested dimension {self._output_dimension}."
            )

        if result.shape[-1] != self._output_dimension:
            result = result[..., : self._output_dimension]

            # Truncating a normalized Matryoshka vector changes its norm.
            # Re-normalize so cosine search can safely use dot product later.
            if self.normalize_embeddings:
                norm = np.linalg.norm(result, axis=-1, keepdims=True)
                result = np.divide(
                    result,
                    norm,
                    out=np.zeros_like(result),
                    where=norm > 0,
                )

        return result.astype(np.float32, copy=False)
