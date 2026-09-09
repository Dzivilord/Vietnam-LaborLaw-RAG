"""Parser for Vietnamese labor-law text.

The parser converts cleaned plain text into a hierarchical structure:
Document -> Chapter -> Section -> Article -> Clause.

Points such as ``a)``, ``b)`` remain inside their parent clause content.
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any


class LaborLawParser:
    """Parse cleaned Vietnamese legal text down to clause level."""

    CHAPTER_RE = re.compile(
        r"^Chương\s+(?P<number>[IVXLCDM]+|\d+)\s*$",
        flags=re.IGNORECASE,
    )
    SECTION_RE = re.compile(
        r"^Mục\s+(?P<number>\d+[A-Za-z]?)\.\s*(?P<title>.*)$",
        flags=re.IGNORECASE,
    )
    ARTICLE_RE = re.compile(
        r"^Điều\s+(?P<number>\d+[A-Za-z]?)\.\s*(?P<title>.*)$",
        flags=re.IGNORECASE,
    )
    CLAUSE_RE = re.compile(
        r"^(?P<number>\d+)\.\s+(?P<content>.+)$"
    )
    POINT_RE = re.compile(r"^[a-zđ]\)\s+", flags=re.IGNORECASE)
    POSTAMBLE_RE = re.compile(
        r"^Bộ luật này được Quốc hội nước Cộng hòa xã hội chủ nghĩa Việt Nam",
        flags=re.IGNORECASE,
    )

    def parse( self, text: str, document_metadata: dict[str, Any] | None = None, ) -> dict[str, Any]:
        """Parse text and return a nested JSON-serializable dictionary.

        Args:
            text: Cleaned text extracted from the source document.
            document_metadata: Optional metadata merged into the root object.

        Raises:
            ValueError: If the input is empty or no article is detected.
        """

        lines = self._prepare_lines(text)
        if not lines:
            raise ValueError("Input text is empty after normalization.")

        lines, postamble_lines = self._split_postamble(lines)

        document: dict[str, Any] = {
            "document_id": None,
            "document_title": None,
            "document_number": None,
            "document_type": None,
            "issued_date": None,
            "effective_date": None,
            "source_file": None,
            "preamble": None,
            "postamble": None,
            "chapters": [],
        }
        if document_metadata:
            document.update(deepcopy(document_metadata))

        preamble_lines: list[str] = []
        current_chapter: dict[str, Any] | None = None
        current_section: dict[str, Any] | None = None
        current_article: dict[str, Any] | None = None
        article_body_lines: list[str] = []

        index = 0
        while index < len(lines):
            line = lines[index]

            chapter_match = self.CHAPTER_RE.match(line)
            if chapter_match:
                self._finalize_article(current_article, article_body_lines)
                current_article = None
                article_body_lines = []
                current_section = None

                title_lines, next_index = self._consume_uppercase_title(
                    lines,
                    index + 1,
                )
                chapter_number = chapter_match.group("number").upper()
                current_chapter = {
                    "chapter_id": self._chapter_id(chapter_number),
                    "chapter_number": chapter_number,
                    "chapter_title": self._join_inline(title_lines) or None,
                    "intro_text": None,
                    "articles": [],
                    "sections": [],
                }
                document["chapters"].append(current_chapter)
                index = next_index
                continue

            section_match = self.SECTION_RE.match(line)
            if section_match:
                self._finalize_article(current_article, article_body_lines)
                current_article = None
                article_body_lines = []

                if current_chapter is None:
                    current_chapter = self._create_unassigned_chapter()
                    document["chapters"].append(current_chapter)

                title_lines = [section_match.group("title").strip()]
                continuation, next_index = self._consume_uppercase_title(
                    lines,
                    index + 1,
                )
                title_lines.extend(continuation)

                section_number = section_match.group("number")
                current_section = {
                    "section_id": (
                        f"{current_chapter['chapter_id']}-section-"
                        f"{section_number.lower()}"
                    ),
                    "section_number": section_number,
                    "section_title": self._join_inline(title_lines) or None,
                    "intro_text": None,
                    "articles": [],
                }
                current_chapter["sections"].append(current_section)
                index = next_index
                continue

            article_match = self.ARTICLE_RE.match(line)
            if article_match:
                self._finalize_article(current_article, article_body_lines)
                article_body_lines = []

                if current_chapter is None:
                    current_chapter = self._create_unassigned_chapter()
                    document["chapters"].append(current_chapter)

                article_number = article_match.group("number")
                article_title = article_match.group("title").strip()
                current_article = {
                    "article_id": f"article-{article_number.lower()}",
                    "article_number": article_number,
                    "article_title": article_title or None,
                    "intro_text": None,
                    "article_content": None,
                    "clauses": [],
                    "raw_text": None,
                }

                target_articles = (
                    current_section["articles"]
                    if current_section is not None
                    else current_chapter["articles"]
                )
                target_articles.append(current_article)
                index += 1
                continue

            if current_article is not None:
                article_body_lines.append(line)
            elif current_chapter is None:
                preamble_lines.append(line)
            elif current_section is not None:
                current_section["intro_text"] = self._append_text(
                    current_section.get("intro_text"),
                    line,
                )
            else:
                current_chapter["intro_text"] = self._append_text(
                    current_chapter.get("intro_text"),
                    line,
                )

            index += 1

        self._finalize_article(current_article, article_body_lines)
        document["preamble"] = self._join_content_lines(preamble_lines) or None
        document["postamble"] = (
            self._join_content_lines(postamble_lines) or None
        )

        self._infer_document_metadata(document)
        self._validate(document)
        return document

    def summarize(self, document: dict[str, Any]) -> dict[str, int]:
        """Return structural counts useful for CLI reporting and validation."""

        chapters = document.get("chapters", [])
        section_count = 0
        article_count = 0
        clause_count = 0
        articles_without_clauses = 0

        for chapter in chapters:
            direct_articles = chapter.get("articles", [])
            article_count += len(direct_articles)
            clause_count += sum(len(a.get("clauses", [])) for a in direct_articles)
            articles_without_clauses += sum(
                not a.get("clauses") for a in direct_articles
            )

            for section in chapter.get("sections", []):
                section_count += 1
                articles = section.get("articles", [])
                article_count += len(articles)
                clause_count += sum(len(a.get("clauses", [])) for a in articles)
                articles_without_clauses += sum(
                    not a.get("clauses") for a in articles
                )

        return {
            "chapters": len(chapters),
            "sections": section_count,
            "articles": article_count,
            "clauses": clause_count,
            "articles_without_clauses": articles_without_clauses,
        }


    def _split_postamble(
        self,
        lines: list[str],
    ) -> tuple[list[str], list[str]]:
        """Separate the enactment/signature block after the final article."""

        for index, line in enumerate(lines):
            if self.POSTAMBLE_RE.match(line):
                return lines[:index], lines[index:]
        return lines, []

    def _finalize_article(
        self,
        article: dict[str, Any] | None,
        body_lines: list[str],
    ) -> None:
        if article is None:
            return

        title_continuation, content_lines = self._split_title_continuation(
            body_lines
        )
        if title_continuation:
            article["article_title"] = self._join_inline(
                [article.get("article_title") or "", *title_continuation]
            )

        intro_lines, clauses = self._split_clauses(
            content_lines,
            article_number=str(article["article_number"]),
        )

        if clauses:
            article["intro_text"] = self._join_content_lines(intro_lines) or None
            article["clauses"] = clauses
            article["article_content"] = None
        else:
            article["intro_text"] = None
            article["article_content"] = (
                self._join_content_lines(content_lines) or None
            )
            article["clauses"] = []

        raw_parts = [
            f"Điều {article['article_number']}. {article.get('article_title') or ''}".rstrip()
        ]
        raw_parts.extend(content_lines)
        article["raw_text"] = "\n".join(raw_parts).strip()

    def _split_clauses(
        self,
        lines: list[str],
        article_number: str,
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """Split top-level clauses while ignoring numbering inside quotations."""

        intro_lines: list[str] = []
        clauses: list[dict[str, Any]] = []
        current_number: int | None = None
        current_lines: list[str] = []
        expected_number = 1
        quote_depth = 0

        for line in lines:
            match = self.CLAUSE_RE.match(line) if quote_depth == 0 else None
            candidate_number = int(match.group("number")) if match else None

            is_next_clause = (
                match is not None and candidate_number == expected_number
            )

            if is_next_clause:
                if current_number is not None:
                    clauses.append(
                        self._build_clause(
                            article_number,
                            current_number,
                            current_lines,
                        )
                    )
                current_number = candidate_number
                current_lines = [match.group("content").strip()]
                expected_number += 1
            elif current_number is None:
                intro_lines.append(line)
            else:
                current_lines.append(line)

            quote_depth = max(
                0,
                quote_depth + line.count("“") - line.count("”"),
            )

        if current_number is not None:
            clauses.append(
                self._build_clause(
                    article_number,
                    current_number,
                    current_lines,
                )
            )

        # A numbered list is considered clauses only when it starts from clause 1.
        if not clauses:
            return lines, []

        return intro_lines, clauses

    def _build_clause(
        self,
        article_number: str,
        clause_number: int,
        lines: list[str],
    ) -> dict[str, Any]:
        return {
            "clause_id": (
                f"article-{article_number.lower()}-clause-{clause_number}"
            ),
            "clause_number": str(clause_number),
            "content": self._join_content_lines(lines),
        }

    def _split_title_continuation(
        self,
        body_lines: list[str],
    ) -> tuple[list[str], list[str]]:
        """Recover article titles wrapped by PDF extraction.

        In this source, wrapped title lines start with a lowercase letter, while
        article bodies normally start with an uppercase letter or a clause marker.
        """

        continuation: list[str] = []
        index = 0
        while index < len(body_lines):
            line = body_lines[index]
            first_alpha = self._first_alpha(line)
            if first_alpha is None or not first_alpha.islower():
                break
            continuation.append(line)
            index += 1

        return continuation, body_lines[index:]

    def _consume_uppercase_title(
        self,
        lines: list[str],
        start_index: int,
    ) -> tuple[list[str], int]:
        title_lines: list[str] = []
        index = start_index

        while index < len(lines):
            line = lines[index]
            if self._is_structural_line(line):
                break
            if not self._is_uppercase_heading(line):
                break
            title_lines.append(line)
            index += 1

        return title_lines, index

    def _prepare_lines(self, text: str) -> list[str]:
        text = text.replace("\ufeff", "").replace("\u00a0", " ")
        lines: list[str] = []
        for raw_line in text.splitlines():
            line = re.sub(r"[ \t]+", " ", raw_line).strip()
            if not line:
                continue
            if re.fullmatch(r"=+\s*PAGE\s+\d+\s*=+", line, re.IGNORECASE):
                continue
            lines.append(line)
        return lines

    def _join_content_lines(self, lines: list[str]) -> str:
        """Join PDF-wrapped lines while retaining real paragraphs and point lists."""

        paragraphs: list[str] = []
        current = ""

        for line in lines:
            line = line.strip()
            if not line:
                continue

            begins_list_item = bool(
                self.POINT_RE.match(line)
                or re.match(r"^\d+[a-z]?\.\s+", line, re.IGNORECASE)
                or line.startswith("“Điều ")
            )
            begins_new_sentence = (
                current.endswith((".", ";", ":", "?", "!", ".”", ";”"))
                and self._starts_uppercase(line)
            )

            if current and (begins_list_item or begins_new_sentence):
                paragraphs.append(current.strip())
                current = line
            elif current:
                current = f"{current} {line}"
            else:
                current = line

        if current:
            paragraphs.append(current.strip())

        return "\n".join(paragraphs)

    @staticmethod
    def _join_inline(lines: list[str]) -> str:
        return " ".join(line.strip() for line in lines if line and line.strip())

    @staticmethod
    def _append_text(existing: str | None, line: str) -> str:
        if not existing:
            return line.strip()
        return f"{existing} {line.strip()}"

    def _is_structural_line(self, line: str) -> bool:
        return bool(
            self.CHAPTER_RE.match(line)
            or self.SECTION_RE.match(line)
            or self.ARTICLE_RE.match(line)
        )

    @staticmethod
    def _first_alpha(text: str) -> str | None:
        for character in text:
            if character.isalpha():
                return character
        return None

    def _starts_uppercase(self, text: str) -> bool:
        first_alpha = self._first_alpha(text)
        return bool(first_alpha and first_alpha.isupper())

    @staticmethod
    def _is_uppercase_heading(text: str) -> bool:
        letters = [character for character in text if character.isalpha()]
        return bool(letters) and all(
            not character.islower() for character in letters
        )

    @staticmethod
    def _chapter_id(chapter_number: str) -> str:
        return f"chapter-{chapter_number.lower()}"

    @staticmethod
    def _create_unassigned_chapter() -> dict[str, Any]:
        return {
            "chapter_id": "chapter-unassigned",
            "chapter_number": None,
            "chapter_title": None,
            "intro_text": None,
            "articles": [],
            "sections": [],
        }

    def _infer_document_metadata(self, document: dict[str, Any]) -> None:
        preamble = document.get("preamble") or ""

        if not document.get("document_number"):
            match = re.search(
                r"Bộ luật số:\s*([^\s]+)",
                preamble,
                flags=re.IGNORECASE,
            )
            if match:
                document["document_number"] = match.group(1)

        if not document.get("document_type"):
            document["document_type"] = "Bộ luật"

        if not document.get("document_title"):
            document["document_title"] = "Bộ luật Lao động 2019"

        if not document.get("document_id"):
            document["document_id"] = "bo-luat-lao-dong-2019"

        if not document.get("issued_date"):
            date_match = re.search(
                r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})",
                preamble,
                flags=re.IGNORECASE,
            )
            if date_match:
                day, month, year = date_match.groups()
                document["issued_date"] = (
                    f"{year}-{int(month):02d}-{int(day):02d}"
                )

    def _validate(self, document: dict[str, Any]) -> None:
        stats = self.summarize(document)
        if stats["articles"] == 0:
            raise ValueError("No top-level article heading was detected.")

        article_numbers: list[str] = []
        for chapter in document["chapters"]:
            article_numbers.extend(
                article["article_number"]
                for article in chapter.get("articles", [])
            )
            for section in chapter.get("sections", []):
                article_numbers.extend(
                    article["article_number"]
                    for article in section.get("articles", [])
                )

        duplicates = sorted(
            number
            for number in set(article_numbers)
            if article_numbers.count(number) > 1
        )
        if duplicates:
            raise ValueError(
                "Duplicate top-level article numbers detected: "
                + ", ".join(duplicates)
            )