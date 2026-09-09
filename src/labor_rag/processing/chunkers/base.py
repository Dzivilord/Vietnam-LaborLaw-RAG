"""Shared primitives for legal-document chunkers.

The parser produces a hierarchy:
Document -> Chapter -> Section -> Article -> Clause.

Chunkers consume that parsed dictionary and return flat, JSON-serializable
chunk records ready for embedding/indexing.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from typing import Any


TokenCounter = Callable[[str], int]


class BaseChunker(ABC):
    """Base class for chunking parsed Vietnamese legal documents.

    Args:
        token_counter: Optional function ``text -> token_count``. In production,
            pass the tokenizer/counting function of the embedding model. If it is
            omitted, a lightweight dependency-free approximation is used.
    """

    def __init__(self, token_counter: TokenCounter | None = None) -> None:
        self._token_counter = token_counter or self._approximate_token_count

    @abstractmethod
    def chunk(self, document: dict[str, Any]) -> list[dict[str, Any]]:
        """Convert one parsed legal document into flat chunks."""

    def iter_articles(
        self,
        document: dict[str, Any],
    ) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
        """Yield ``(article, hierarchy_metadata)`` for every article.

        Handles both legal layouts used by the parser:
        - Chapter -> Article
        - Chapter -> Section -> Article
        """

        document_meta = {
            "document_id": document.get("document_id"),
            "document_title": document.get("document_title"),
            "document_number": document.get("document_number"),
            "document_type": document.get("document_type"),
            "issued_date": document.get("issued_date"),
            "effective_date": document.get("effective_date"),
            "source_file": document.get("source_file"),
        }

        for chapter in document.get("chapters", []):
            chapter_meta = {
                **document_meta,
                "chapter_id": chapter.get("chapter_id"),
                "chapter_number": chapter.get("chapter_number"),
                "chapter_title": chapter.get("chapter_title"),
            }

            for article in chapter.get("articles", []):
                yield article, {
                    **chapter_meta,
                    "section_id": None,
                    "section_number": None,
                    "section_title": None,
                }

            for section in chapter.get("sections", []):
                section_meta = {
                    **chapter_meta,
                    "section_id": section.get("section_id"),
                    "section_number": section.get("section_number"),
                    "section_title": section.get("section_title"),
                }
                for article in section.get("articles", []):
                    yield article, section_meta

    def build_article_chunk(
        self,
        article: dict[str, Any],
        hierarchy: dict[str, Any],
    ) -> dict[str, Any]:
        """Build one chunk containing the whole article."""

        article_id = str(article.get("article_id") or "")

        content = self._article_text(article)
        embedding_content = self._article_body_text(article)

        embedding_text = self._build_embedding_text(
            hierarchy=hierarchy,
            article=article,
            content=embedding_content,
        )

        return {
            "chunk_id": self._make_chunk_id(
                hierarchy.get("document_id"),
                article_id,
            ),
            "parent_id": None,
            "chunk_level": "article",
            **hierarchy,
            "article_id": article.get("article_id"),
            "article_number": article.get("article_number"),
            "article_title": article.get("article_title"),
            "clause_id": None,
            "clause_number": None,
            "content": content,
            "embedding_text": embedding_text,
            "token_count": self.count_tokens(embedding_text),
        }

    def build_clause_chunk(
        self,
        article: dict[str, Any],
        clause: dict[str, Any],
        hierarchy: dict[str, Any],
    ) -> dict[str, Any]:
        """Build one chunk for a clause while preserving article context.

        ``content`` remains the exact clause content. The article intro and legal
        hierarchy are added only to ``embedding_text`` so retrieval gets context
        without changing the quoted legal text.
        """

        article_id = str(article.get("article_id") or "")
        clause_id = str(clause.get("clause_id") or "")
        clause_content = str(clause.get("content") or "").strip()

        context_parts: list[str] = []
        intro_text = str(article.get("intro_text") or "").strip()
        if intro_text:
            context_parts.append(intro_text)
        context_parts.append(
            f"Khoản {clause.get('clause_number')}. {clause_content}".strip()
        )
        contextual_content = "\n".join(context_parts)

        embedding_text = self._build_embedding_text(
            hierarchy=hierarchy,
            article=article,
            content=contextual_content,
        )

        return {
            "chunk_id": self._make_chunk_id(
                hierarchy.get("document_id"),
                clause_id,
            ),
            "parent_id": self._make_chunk_id(
                hierarchy.get("document_id"),
                article_id,
            ),
            "chunk_level": "clause",
            **hierarchy,
            "article_id": article.get("article_id"),
            "article_number": article.get("article_number"),
            "article_title": article.get("article_title"),
            "clause_id": clause.get("clause_id"),
            "clause_number": clause.get("clause_number"),
            "content": clause_content,
            "embedding_text": embedding_text,
            "token_count": self.count_tokens(embedding_text),
        }

    def count_tokens(self, text: str) -> int:
        """Count tokens using the configured counter."""

        return int(self._token_counter(text))

    def article_token_count(
        self,
        article: dict[str, Any],
        hierarchy: dict[str, Any],
    ) -> int:
        """Return the token count of the article as one retrieval chunk."""

        return self.build_article_chunk(article, hierarchy)["token_count"]
    def _article_body_text(
        self,
        article: dict[str, Any],
    ) -> str:
        parts: list[str] = []

        if article.get("intro_text"):
            parts.append(article["intro_text"].strip())

        if article.get("article_content"):
            parts.append(article["article_content"].strip())

        for clause in article.get("clauses", []):
            clause_number = clause.get("clause_number")
            clause_content = clause.get("content", "").strip()

            if clause_content:
                if clause_number:
                    parts.append(f"{clause_number}. {clause_content}")
                else:
                    parts.append(clause_content)

        return "\n".join(parts).strip()
    def _build_embedding_text(
        self,
        hierarchy: dict[str, Any],
        article: dict[str, Any],
        content: str,
    ) -> str:
        """Attach legal hierarchy to the text used for embedding."""

        parts: list[str] = []

        # document_title = hierarchy.get("document_title")
        # document_number = hierarchy.get("document_number")
        # if document_title:
        #     label = str(document_title)
        #     if document_number:
        #         label += f" ({document_number})"
        #     parts.append(label)

        chapter_number = hierarchy.get("chapter_number")
        chapter_title = hierarchy.get("chapter_title")
        if chapter_number or chapter_title:
            chapter_label = "Chương"
            if chapter_number:
                chapter_label += f" {chapter_number}"
            if chapter_title:
                chapter_label += f". {chapter_title}"
            parts.append(chapter_label)

        section_number = hierarchy.get("section_number")
        section_title = hierarchy.get("section_title")
        if section_number or section_title:
            section_label = "Mục"
            if section_number:
                section_label += f" {section_number}"
            if section_title:
                section_label += f". {section_title}"
            parts.append(section_label)

        article_number = article.get("article_number")
        article_title = article.get("article_title")
        article_label = "Điều"
        if article_number:
            article_label += f" {article_number}"
        if article_title:
            article_label += f". {article_title}"
        parts.append(article_label)

        if content:
            parts.append(content.strip())

        return "\n".join(parts)

    @staticmethod
    def _article_text(article: dict[str, Any]) -> str:
        """Return a complete article text, preferring parser-preserved raw text."""

        raw_text = str(article.get("raw_text") or "").strip()
        if raw_text:
            return raw_text

        parts: list[str] = []
        article_number = article.get("article_number")
        article_title = article.get("article_title")
        header = f"Điều {article_number}." if article_number else "Điều"
        if article_title:
            header += f" {article_title}"
        parts.append(header.strip())

        intro_text = str(article.get("intro_text") or "").strip()
        if intro_text:
            parts.append(intro_text)

        article_content = str(article.get("article_content") or "").strip()
        if article_content:
            parts.append(article_content)

        for clause in article.get("clauses", []):
            number = clause.get("clause_number")
            content = str(clause.get("content") or "").strip()
            parts.append(f"{number}. {content}".strip())

        return "\n".join(part for part in parts if part)

    @staticmethod
    def _make_chunk_id(document_id: Any, local_id: str) -> str:
        document_prefix = str(document_id or "document")
        return f"{document_prefix}:{local_id}"

    @staticmethod
    def _approximate_token_count(text: str) -> int:
        """Dependency-free fallback token approximation.

        This is intentionally only a fallback. For final experiments, inject the
        tokenizer used by the embedding model so the threshold is exact.
        """

        return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))
