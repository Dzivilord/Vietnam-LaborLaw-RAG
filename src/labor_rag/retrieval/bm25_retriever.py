"""Dependency-free BM25 lexical retriever for processed legal chunks."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .base import BaseRetriever, RetrievalResult


TokenizeFn = Callable[[str], list[str]]


def default_tokenize(text: str) -> list[str]:
    """Simple Unicode word tokenizer suitable for Vietnamese BM25 matching."""
    return re.findall(r"[^\W_]+", text.lower(), flags=re.UNICODE)


class BM25Retriever(BaseRetriever):
    """BM25 retriever built from the same JSONL chunks used for dense indexing.

    The index is built in memory when the retriever is created. This keeps the
    first implementation simple and avoids changing the existing Qdrant dense
    collection.
    """

    def __init__(
        self,
        chunks: Sequence[dict[str, Any]],
        *,
        text_field: str = "embedding_text",
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: TokenizeFn = default_tokenize,
    ) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be greater than 0.")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1.")
        if not chunks:
            raise ValueError("chunks must not be empty.")

        self._chunks = [dict(chunk) for chunk in chunks]
        self._text_field = text_field
        self._k1 = float(k1)
        self._b = float(b)
        self._tokenizer = tokenizer

        self._doc_tokens: list[list[str]] = []
        self._term_frequencies: list[Counter[str]] = []
        self._doc_lengths: list[int] = []
        document_frequency: Counter[str] = Counter()

        for index, chunk in enumerate(self._chunks):
            text = str(
                chunk.get(text_field)
                or chunk.get("content")
                or ""
            ).strip()
            if not text:
                raise ValueError(
                    f"Chunk at index {index} has no text in {text_field!r} "
                    "or 'content'."
                )

            tokens = tokenizer(text)
            term_frequency = Counter(tokens)

            self._doc_tokens.append(tokens)
            self._term_frequencies.append(term_frequency)
            self._doc_lengths.append(len(tokens))
            document_frequency.update(term_frequency.keys())

        self._document_count = len(self._chunks)
        self._avg_doc_length = (
            sum(self._doc_lengths) / self._document_count
            if self._document_count
            else 0.0
        )
        self._idf = {
            term: math.log(
                1.0
                + (
                    self._document_count - frequency + 0.5
                ) / (frequency + 0.5)
            )
            for term, frequency in document_frequency.items()
        }

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        **kwargs: Any,
    ) -> "BM25Retriever":
        """Build a BM25 index from the processed legal chunk JSONL file."""
        path = Path(path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Chunk file not found: {path}")

        chunks: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as file:
            for line_number, raw_line in enumerate(file, start=1):
                line = raw_line.strip()
                if not line:
                    continue

                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON at line {line_number}: {exc}"
                    ) from exc

                if not isinstance(chunk, dict):
                    raise ValueError(
                        f"Line {line_number} must contain a JSON object."
                    )
                chunks.append(chunk)

        return cls(chunks, **kwargs)

    def retrieve(self, query: str, *, top_k: int = 5) -> list[RetrievalResult]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        query_terms = self._tokenizer(query.strip())
        if not query_terms:
            return []

        query_term_counts = Counter(query_terms)
        scored: list[tuple[float, int]] = []

        for doc_index, term_frequency in enumerate(self._term_frequencies):
            doc_length = self._doc_lengths[doc_index]
            score = 0.0

            for term, query_frequency in query_term_counts.items():
                tf = term_frequency.get(term, 0)
                if tf == 0:
                    continue

                idf = self._idf.get(term, 0.0)
                length_norm = 1.0 - self._b
                if self._avg_doc_length > 0:
                    length_norm += self._b * doc_length / self._avg_doc_length

                denominator = tf + self._k1 * length_norm
                score += (
                    query_frequency
                    * idf
                    * (tf * (self._k1 + 1.0) / denominator)
                )

            if score > 0.0:
                scored.append((score, doc_index))

        scored.sort(key=lambda item: item[0], reverse=True)

        results: list[RetrievalResult] = []
        for score, doc_index in scored[:top_k]:
            chunk = self._chunks[doc_index]
            metadata = dict(chunk)
            metadata["retrieval_method"] = "bm25"

            results.append(
                RetrievalResult(
                    chunk_id=str(chunk.get("chunk_id") or doc_index),
                    content=str(
                        chunk.get("content")
                        or chunk.get(self._text_field)
                        or ""
                    ),
                    score=float(score),
                    metadata=metadata,
                )
            )

        return results
