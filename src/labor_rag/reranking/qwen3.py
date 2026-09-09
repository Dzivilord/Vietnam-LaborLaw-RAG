"""Qwen3 reranker for retrieval candidates.

This module implements ``Qwen/Qwen3-Reranker-0.6B`` using the model's
instruction-aware yes/no scoring formulation.  It keeps the same
``BaseReranker`` interface as the existing BGE reranker so it can be swapped
into the current retrieval -> rerank -> context pipeline.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from labor_rag.retrieval.base import RetrievalResult

from .base import BaseReranker


class Qwen3Reranker(BaseReranker):
    """Rerank candidates with ``Qwen/Qwen3-Reranker-0.6B``.

    Qwen3-Reranker is instruction-aware.  For each ``(query, chunk)`` pair,
    the model is prompted to judge whether the document meets the query and
    must answer ``yes`` or ``no``.  The returned relevance score is the
    normalized probability assigned to ``yes`` relative to ``yes/no``.

    The default instruction is tailored to Vietnamese labor-law RAG and
    explicitly asks the model to treat concise and conversational paraphrases
    with the same intent as equivalent.  The instruction can be overridden
    without changing the model implementation.
    """

    MODEL_NAME = "Qwen/Qwen3-Reranker-0.6B"
    DEFAULT_INSTRUCTION = (
        "Given a Vietnamese labor-law query, judge whether the Document contains"
        "directly relevant information about the user's requested topic or question."

        "A Document is relevant even if it answers only one part or one aspect of a"
        "broad or comprehensive query. Do not require a single Document to answer the"
        "entire query."

        "Focus on semantic topical relevance. Do not reward documents merely because"
        "they discuss a related legal domain, such as employment contracts in general,"
        "unless they directly concern the requested topic."
    )

    _PREFIX = (
        '<|im_start|>system\n'
        'Judge whether the Document meets the requirements based on the Query '
        'and the Instruct provided. Note that the answer can only be "yes" or '
        '"no".<|im_end|>\n<|im_start|>user\n'
    )
    _SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'

    def __init__(
        self,
        *,
        model_name: str = MODEL_NAME,
        instruction: str = DEFAULT_INSTRUCTION,
        device: str | None = None,
        batch_size: int = 8,
        max_length: int = 2048,
        use_fp16: bool = True,
        trust_remote_code: bool = True,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0.")
        if max_length <= 0:
            raise ValueError("max_length must be greater than 0.")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be a non-empty string.")

        self.model_name = model_name
        self.instruction = instruction.strip()
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.use_fp16 = use_fp16
        self.trust_remote_code = trust_remote_code

        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._resolved_device: str | None = None
        self._token_yes_id: int | None = None
        self._token_no_id: int | None = None
        self._prefix_tokens: list[int] | None = None
        self._suffix_tokens: list[int] | None = None

    def _ensure_model(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Qwen3Reranker requires torch and transformers>=4.51.0."
            ) from exc

        resolved_device = self.device
        if resolved_device is None:
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
            padding_side="left",
        )

        dtype = (
            torch.float16
            if self.use_fp16 and resolved_device.startswith("cuda")
            else torch.float32
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
            torch_dtype=dtype,
        )
        model.eval()
        model.to(resolved_device)

        token_yes_id = tokenizer.convert_tokens_to_ids("yes")
        token_no_id = tokenizer.convert_tokens_to_ids("no")
        if token_yes_id is None or token_no_id is None:
            raise RuntimeError("Could not resolve Qwen3 reranker yes/no token IDs.")

        prefix_tokens = tokenizer.encode(self._PREFIX, add_special_tokens=False)
        suffix_tokens = tokenizer.encode(self._SUFFIX, add_special_tokens=False)

        reserved_tokens = len(prefix_tokens) + len(suffix_tokens)
        if self.max_length <= reserved_tokens:
            raise ValueError(
                "max_length is too small for Qwen3 reranker prompt tokens: "
                f"max_length={self.max_length}, reserved={reserved_tokens}."
            )

        self._tokenizer = tokenizer
        self._model = model
        self._resolved_device = resolved_device
        self._token_yes_id = int(token_yes_id)
        self._token_no_id = int(token_no_id)
        self._prefix_tokens = prefix_tokens
        self._suffix_tokens = suffix_tokens

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

    def _format_pair(self, query: str, document: str) -> str:
        return (
            f"<Instruct>: {self.instruction}\n"
            f"<Query>: {query}\n"
            f"<Document>: {document}"
        )

    def _prepare_inputs(self, texts: list[str]) -> dict[str, Any]:
        if (
            self._tokenizer is None
            or self._prefix_tokens is None
            or self._suffix_tokens is None
            or self._resolved_device is None
        ):
            raise RuntimeError("Qwen3 reranker tokenizer was not initialized correctly.")

        available_length = (
            self.max_length
            - len(self._prefix_tokens)
            - len(self._suffix_tokens)
        )

        encoded = self._tokenizer(
            texts,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=available_length,
        )

        encoded["input_ids"] = [
            self._prefix_tokens + token_ids + self._suffix_tokens
            for token_ids in encoded["input_ids"]
        ]

        inputs = self._tokenizer.pad(
            encoded,
            padding=True,
            return_tensors="pt",
            max_length=self.max_length,
        )

        return {
            key: value.to(self._resolved_device)
            for key, value in inputs.items()
        }

    def _compute_scores(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
    ) -> list[float]:
        self._ensure_model()

        import torch

        if (
            self._model is None
            or self._token_yes_id is None
            or self._token_no_id is None
        ):
            raise RuntimeError("Qwen3 reranker model was not initialized correctly.")

        scores: list[float] = []

        with torch.inference_mode():
            for start in range(0, len(candidates), self.batch_size):
                batch = candidates[start : start + self.batch_size]
                texts = [
                    self._format_pair(query, candidate.content)
                    for candidate in batch
                ]
                inputs = self._prepare_inputs(texts)

                next_token_logits = self._model(**inputs).logits[:, -1, :].float()
                yes_logits = next_token_logits[:, self._token_yes_id]
                no_logits = next_token_logits[:, self._token_no_id]

                yes_no_logits = torch.stack([no_logits, yes_logits], dim=1)
                batch_scores = torch.softmax(yes_no_logits, dim=1)[:, 1]
                scores.extend(batch_scores.cpu().tolist())

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
                    "reranker_score_type": "qwen3_yes_probability",
                    "reranker_instruction": self.instruction,
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
