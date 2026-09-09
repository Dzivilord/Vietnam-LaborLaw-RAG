"""Structure-aware chunker for Vietnamese legal documents.

Policy:
- Short article -> one article chunk.
- Long article with numbered clauses -> one chunk per clause.
- Long article without clauses -> keep the whole article as one chunk because
  the parser intentionally stops at clause level and there is no lower legal
  structure available to split on.
"""

from __future__ import annotations

from typing import Any

from .base import BaseChunker, TokenCounter


class LegalStructureChunker(BaseChunker):
    """Split only long articles, using clauses as legal boundaries.

    Args:
        max_tokens: Maximum preferred size for an article chunk. The default
            450 leaves headroom for hierarchy/context when using embedding
            models with roughly 512-token limits.
        token_counter: Optional exact token-counting function from the embedding
            model tokenizer.
    """

    def __init__(
        self,
        max_tokens: int = 450,
        token_counter: TokenCounter | None = None,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than 0.")

        super().__init__(token_counter=token_counter)
        self.max_tokens = max_tokens

    def chunk(self, document: dict[str, Any]) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []

        for article, hierarchy in self.iter_articles(document):
            article_chunk = self.build_article_chunk(
                article=article,
                hierarchy=hierarchy,
            )

            clauses = article.get("clauses", [])
            is_long_article = article_chunk["token_count"] > self.max_tokens

            if not is_long_article or not clauses:
                article_chunk["split_reason"] = (
                    "within_limit"
                    if not is_long_article
                    else "no_clause_boundary"
                )
                article_chunk["is_oversized"] = is_long_article
                chunks.append(article_chunk)
                continue

            for clause in clauses:
                clause_chunk = self.build_clause_chunk(
                    article=article,
                    clause=clause,
                    hierarchy=hierarchy,
                )
                clause_chunk["split_reason"] = "article_exceeds_max_tokens"
                clause_chunk["is_oversized"] = (
                    clause_chunk["token_count"] > self.max_tokens
                )
                chunks.append(clause_chunk)

        return chunks
