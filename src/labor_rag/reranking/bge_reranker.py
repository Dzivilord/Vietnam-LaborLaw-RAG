"""BGE cross-encoder reranker for retrieval candidates."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from labor_rag.retrieval.base import RetrievalResult

from .base import BaseReranker


class BGEReranker(BaseReranker):
    """Rerank candidates with ``BAAI/bge-reranker-v2-m3``.

    BGE reranker is a cross-encoder: each ``(query, chunk)`` pair is encoded
    together and the model directly predicts a relevance logit. This is more
    expensive than dense retrieval, so it should only be applied to a small
    candidate set such as the top 20-50 retrieved chunks.

    The model is loaded lazily on the first ``rerank`` call so importing or
    constructing the class does not immediately allocate model memory.
    """

    MODEL_NAME = "BAAI/bge-reranker-v2-m3"

    def __init__(
        self,
        *,
        model_name: str = MODEL_NAME,
        device: str | None = None,
        batch_size: int = 8,
        max_length: int = 512,
        normalize_scores: bool = True,
        use_fp16: bool = True,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0.")
        if max_length <= 0:
            raise ValueError("max_length must be greater than 0.")

        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.normalize_scores = normalize_scores
        self.use_fp16 = use_fp16

        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._resolved_device: str | None = None

    def _ensure_model(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "BGEReranker requires torch and transformers. "
                "They are normally installed with sentence-transformers."
            ) from exc

        resolved_device = self.device
        if resolved_device is None:
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        model.eval()
        model.to(resolved_device)

        if self.use_fp16 and resolved_device.startswith("cuda"):
            model.half()

        self._tokenizer = tokenizer
        self._model = model
        self._resolved_device = resolved_device

    @staticmethod
    def _validate(
        query: str,
        candidates: Sequence[RetrievalResult],
        top_k: int | None,
    ) -> str:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be greater than 0 when provided.")

        for index, candidate in enumerate(candidates):
            if not isinstance(candidate, RetrievalResult):
                raise TypeError(
                    f"candidates[{index}] must be a RetrievalResult, "
                    f"got {type(candidate).__name__}."
                )

        return query.strip()

    def _compute_scores(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
    ) -> list[float]:
        self._ensure_model()

        import torch

        if self._tokenizer is None or self._model is None or self._resolved_device is None:
            raise RuntimeError("Reranker model was not initialized correctly.")

        scores: list[float] = []

        with torch.inference_mode():
            for start in range(0, len(candidates), self.batch_size):
                batch = candidates[start : start + self.batch_size]
                pairs = [
                    [query, candidate.content]
                    for candidate in batch
                ]

                inputs = self._tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                inputs = {
                    key: value.to(self._resolved_device)
                    for key, value in inputs.items()
                }

                logits = self._model(**inputs, return_dict=True).logits.view(-1).float()

                if self.normalize_scores:
                    logits = torch.sigmoid(logits)

                scores.extend(logits.cpu().tolist())

        return [float(score) for score in scores]

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        *,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        query = self._validate(query, candidates, top_k)

        if not candidates:
            return []

        reranker_scores = self._compute_scores(query, candidates)
        reranked: list[RetrievalResult] = []

        for candidate, reranker_score in zip(
            candidates,
            reranker_scores,
            strict=True,
        ):
            metadata = dict(candidate.metadata)
            metadata.update(
                {
                    "retrieval_score": candidate.score,
                    "reranker": self.model_name,
                    "reranker_score": reranker_score,
                    "reranker_score_normalized": self.normalize_scores,
                }
            )

            reranked.append(
                RetrievalResult(
                    chunk_id=candidate.chunk_id,
                    content=candidate.content,
                    score=reranker_score,
                    metadata=metadata,
                )
            )

        reranked.sort(key=lambda result: result.score, reverse=True)

        if top_k is not None:
            return reranked[:top_k]
        return reranked
