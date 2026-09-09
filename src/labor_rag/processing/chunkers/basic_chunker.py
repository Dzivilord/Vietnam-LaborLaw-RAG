"""Baseline chunker: one legal article equals one chunk."""

from __future__ import annotations

from typing import Any

from .base import BaseChunker, TokenCounter


class BasicChunker(BaseChunker):
    """Create exactly one chunk for every parsed article."""

    def __init__(self, token_counter: TokenCounter | None = None) -> None:
        super().__init__(token_counter=token_counter)

    def chunk(self, document: dict[str, Any]) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []

        for article, hierarchy in self.iter_articles(document):
            chunks.append(
                self.build_article_chunk(
                    article=article,
                    hierarchy=hierarchy,
                )
            )

        return chunks
