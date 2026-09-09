"""Hierarchy-aware, generation-ready context builder.

This module deliberately does *not* infer legal structure from chunk text.
It consumes structure already attached to each ``RetrievalResult.metadata``
(e.g. document/chapter/section/article/clause/point identifiers) and uses that
metadata to select, de-duplicate, group, order, and render evidence.

Design goals
------------
1. Keep retrieval/reranker diagnostics separate from the text shown to the LLM.
2. Preserve evidence boundaries instead of concatenating candidates into one
   indistinguishable paragraph.
3. Use hierarchy metadata rather than regex/text syntax to understand source
   structure.
4. Avoid parent/child redundancy when the same legal unit is retrieved at
   multiple chunk granularities.
5. Keep generation context compact with grouping and token budgeting.
6. Preserve rank/score/provenance in structured diagnostics for logging,
   evaluation, and optional prompt use later.

The builder remains LLM-free and deterministic. Semantic compression,
semantic deduplication, contradiction detection, and dependency expansion are
intentionally left for later modules because they require semantic models or
additional corpus access.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal, Sequence

from labor_rag.models.rag import EvidenceItem
from labor_rag.retrieval.base import RetrievalResult


ContextStatus = Literal["ready", "insufficient_context"]
ContextOrder = Literal["legal", "relevance"]
HierarchyPreference = Literal["higher_score", "more_specific", "parent"]
RankSource = Literal["reranker", "retrieval"]


@dataclass(slots=True, frozen=True)
class ContextBuilderConfig:
    """Configuration for context selection and rendering."""

    # Applied only on the reranker path.
    min_score: float = 0.60
    min_relative_to_top: float | None = None
    score_gap: float | None = None
    min_chunks_before_gap: int = 1

    # Shared selection budget.
    max_chunks: int = 8
    max_tokens: int | None = 3500
    metadata_token_overhead: int = 24

    # Presentation / hierarchy behavior.
    order_by: ContextOrder = "legal"
    include_source_header: bool = True
    include_document_metadata: bool = True
    group_by_hierarchy: bool = True
    compact_hierarchy_headers: bool = True
    hierarchical_deduplication: bool = True
    hierarchy_preference: HierarchyPreference = "higher_score"

    # Scores/ranks are kept in structured diagnostics by default. Enabling
    # these options exposes them to the generation LLM as text.
    include_rank_in_context: bool = False
    include_score_in_context: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_score <= 1.0:
            raise ValueError("min_score must be between 0 and 1.")
        if self.min_relative_to_top is not None and not (
            0.0 <= self.min_relative_to_top <= 1.0
        ):
            raise ValueError("min_relative_to_top must be between 0 and 1.")
        if self.score_gap is not None and self.score_gap < 0.0:
            raise ValueError("score_gap must be non-negative.")
        if self.min_chunks_before_gap <= 0:
            raise ValueError("min_chunks_before_gap must be greater than 0.")
        if self.max_chunks <= 0:
            raise ValueError("max_chunks must be greater than 0.")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("max_tokens must be greater than 0 when provided.")
        if self.metadata_token_overhead < 0:
            raise ValueError("metadata_token_overhead must be non-negative.")
        if self.order_by not in {"legal", "relevance"}:
            raise ValueError("order_by must be 'legal' or 'relevance'.")
        if self.hierarchy_preference not in {
            "higher_score",
            "more_specific",
            "parent",
        }:
            raise ValueError(
                "hierarchy_preference must be 'higher_score', "
                "'more_specific', or 'parent'."
            )


@dataclass(slots=True)
class ContextSelection:
    """One selected evidence chunk.

    ``rerank_rank`` is kept for backward compatibility. On the no-rerank path
    it stores retrieval rank and ``rank_source`` makes that explicit.
    """

    result: RetrievalResult
    rerank_rank: int
    estimated_tokens: int
    rank_source: RankSource = "reranker"
    evidence_id: str = ""

    @property
    def source_rank(self) -> int:
        return self.rerank_rank


@dataclass(slots=True)
class ContextBuildResult:
    """Structured output consumed by pipeline/generation/API layers."""

    status: ContextStatus
    query: str
    context: str
    selected: list[ContextSelection]
    evidence: list[EvidenceItem]
    rejected: list[dict[str, Any]]
    diagnostics: dict[str, Any]


class RerankedContextBuilder:
    """Build compact structured evidence from retrieval/reranking results."""

    _LEVEL_DEPTH = {
        "document": 0,
        "chapter": 1,
        "section": 2,
        "article": 3,
        "clause": 4,
        "point": 5,
    }

    def __init__(self, config: ContextBuilderConfig | None = None) -> None:
        self.config = config or ContextBuilderConfig()

    # ------------------------------------------------------------------
    # Public API: reranker path
    # ------------------------------------------------------------------
    def build(
        self,
        query: str,
        results: Sequence[RetrievalResult],
    ) -> ContextBuildResult:
        query = self._validate(query, results)
        if not results:
            return self._empty_result(query, input_count=0, rank_source="reranker")

        ranked = sorted(results, key=lambda item: item.score, reverse=True)
        rejected: list[dict[str, Any]] = []

        # 1) Absolute relevance gate.
        thresholded: list[tuple[int, RetrievalResult]] = []
        for rank, result in enumerate(ranked, start=1):
            if float(result.score) < self.config.min_score:
                rejected.append(
                    self._rejection(result, rank, "below_min_score", "reranker")
                )
                continue
            thresholded.append((rank, result))

        if not thresholded:
            return self._empty_result(
                query,
                input_count=len(ranked),
                rejected=rejected,
                top_score=float(ranked[0].score),
                rank_source="reranker",
            )

        top_score = float(thresholded[0][1].score)

        # 2) Optional relative-to-top filter.
        if self.config.min_relative_to_top is not None:
            floor = top_score * self.config.min_relative_to_top
            kept: list[tuple[int, RetrievalResult]] = []
            for rank, result in thresholded:
                if float(result.score) < floor:
                    rejected.append(
                        self._rejection(
                            result, rank, "below_relative_to_top", "reranker"
                        )
                    )
                    continue
                kept.append((rank, result))
            thresholded = kept

        if not thresholded:
            return self._empty_result(
                query,
                input_count=len(ranked),
                rejected=rejected,
                top_score=top_score,
                rank_source="reranker",
            )

        # 3) Optional score-gap experiment. Disabled by default.
        gap_cut_index: int | None = None
        gap_value: float | None = None
        if self.config.score_gap is not None:
            for index in range(len(thresholded) - 1):
                if index + 1 < self.config.min_chunks_before_gap:
                    continue
                current_score = float(thresholded[index][1].score)
                next_score = float(thresholded[index + 1][1].score)
                gap = current_score - next_score
                if gap >= self.config.score_gap:
                    gap_cut_index = index + 1
                    gap_value = gap
                    break

        if gap_cut_index is not None:
            for rank, result in thresholded[gap_cut_index:]:
                rejected.append(
                    self._rejection(result, rank, "after_score_gap", "reranker")
                )
            thresholded = thresholded[:gap_cut_index]

        return self._select_and_render(
            query=query,
            ranked_input=ranked,
            candidates=thresholded,
            rejected=rejected,
            rank_source="reranker",
            top_score=top_score,
            gap_cut_index=gap_cut_index,
            gap_value=gap_value,
        )

    # ------------------------------------------------------------------
    # Public API: no-reranker path
    # ------------------------------------------------------------------
    def build_from_retrieval(
        self,
        query: str,
        results: Sequence[RetrievalResult],
        *,
        top_k: int = 5,
    ) -> ContextBuildResult:
        """Build context directly from retrieval rank when reranking is off.

        Retrieval scores are deliberately not compared with ``min_score``
        because dense/BM25/RRF scores live on different scales.
        """
        query = self._validate_retrieval(query, results)
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")
        if not results:
            return self._empty_result(query, input_count=0, rank_source="retrieval")

        ranked = list(results)
        rejected: list[dict[str, Any]] = []
        limit = min(top_k, self.config.max_chunks)
        candidates: list[tuple[int, RetrievalResult]] = []

        for rank, result in enumerate(ranked, start=1):
            if rank <= limit:
                candidates.append((rank, result))
            else:
                rejected.append(
                    self._rejection(
                        result,
                        rank,
                        "retrieval_top_k_fallback",
                        "retrieval",
                    )
                )

        return self._select_and_render(
            query=query,
            ranked_input=ranked,
            candidates=candidates,
            rejected=rejected,
            rank_source="retrieval",
            top_score=float(ranked[0].score),
            retrieval_context_k=top_k,
        )

    # ------------------------------------------------------------------
    # Shared selection logic
    # ------------------------------------------------------------------
    def _select_and_render(
        self,
        *,
        query: str,
        ranked_input: Sequence[RetrievalResult],
        candidates: list[tuple[int, RetrievalResult]],
        rejected: list[dict[str, Any]],
        rank_source: RankSource,
        top_score: float | None,
        gap_cut_index: int | None = None,
        gap_value: float | None = None,
        retrieval_context_k: int | None = None,
    ) -> ContextBuildResult:
        # 4) Exact chunk/content deduplication.
        candidates = self._exact_deduplicate(
            candidates,
            rejected=rejected,
            rank_source=rank_source,
        )

        # 5) Metadata-only hierarchical deduplication.
        if self.config.hierarchical_deduplication:
            candidates = self._hierarchical_deduplicate(
                candidates,
                rejected=rejected,
                rank_source=rank_source,
            )

        # 6) Safety cap.
        if len(candidates) > self.config.max_chunks:
            for rank, result in candidates[self.config.max_chunks :]:
                rejected.append(
                    self._rejection(result, rank, "max_chunks", rank_source)
                )
            candidates = candidates[: self.config.max_chunks]

        # 7) Token-budget-aware selection.
        selected: list[ContextSelection] = []
        used_tokens = 0
        for rank, result in candidates:
            estimated_tokens = self._estimate_tokens(result)
            if (
                self.config.max_tokens is not None
                and selected
                and used_tokens + estimated_tokens > self.config.max_tokens
            ):
                rejected.append(
                    self._rejection(result, rank, "token_budget", rank_source)
                )
                continue

            selected.append(
                ContextSelection(
                    result=result,
                    rerank_rank=rank,
                    estimated_tokens=estimated_tokens,
                    rank_source=rank_source,
                )
            )
            used_tokens += estimated_tokens

        if not selected:
            return self._empty_result(
                query,
                input_count=len(ranked_input),
                rejected=rejected,
                top_score=top_score,
                rank_source=rank_source,
            )

        # 8) Presentation ordering does not destroy original rank metadata.
        formatted_selection = list(selected)
        if self.config.order_by == "legal":
            formatted_selection.sort(key=self._legal_sort_key)
        else:
            formatted_selection.sort(key=lambda item: item.source_rank)

        for index, item in enumerate(formatted_selection, start=1):
            item.evidence_id = f"E{index}"

        context = self._format_context(formatted_selection)
        evidence = [self._evidence_item(item) for item in formatted_selection]

        diagnostics: dict[str, Any] = {
            "input_count": len(ranked_input),
            "selected_count": len(selected),
            "rejected_count": len(rejected),
            "rank_source": rank_source,
            "top_score": top_score,
            "lowest_selected_score": min(
                float(item.result.score) for item in selected
            ),
            "estimated_context_tokens": used_tokens,
            "score_gap_cut_applied": gap_cut_index is not None,
            "score_gap_value": gap_value,
            "evidence_ids": [item.evidence_id for item in evidence],
            "selection_methods": self._selection_methods(rank_source),
            "config": self._config_diagnostics(retrieval_context_k),
        }

        return ContextBuildResult(
            status="ready",
            query=query,
            context=context,
            selected=selected,
            evidence=evidence,
            rejected=rejected,
            diagnostics=diagnostics,
        )

    def _exact_deduplicate(
        self,
        candidates: list[tuple[int, RetrievalResult]],
        *,
        rejected: list[dict[str, Any]],
        rank_source: RankSource,
    ) -> list[tuple[int, RetrievalResult]]:
        kept: list[tuple[int, RetrievalResult]] = []
        seen_ids: set[str] = set()
        seen_contents: set[str] = set()

        for rank, result in candidates:
            normalized_content = self._normalize_text(result.content)
            if result.chunk_id in seen_ids:
                rejected.append(
                    self._rejection(result, rank, "duplicate_chunk_id", rank_source)
                )
                continue
            if normalized_content and normalized_content in seen_contents:
                rejected.append(
                    self._rejection(result, rank, "duplicate_content", rank_source)
                )
                continue

            seen_ids.add(result.chunk_id)
            if normalized_content:
                seen_contents.add(normalized_content)
            kept.append((rank, result))

        return kept

    def _hierarchical_deduplicate(
        self,
        candidates: list[tuple[int, RetrievalResult]],
        *,
        rejected: list[dict[str, Any]],
        rank_source: RankSource,
    ) -> list[tuple[int, RetrievalResult]]:
        """Remove explicit parent/child redundancy using metadata only.

        This method never parses legal text. A relationship is recognized only
        when metadata explicitly states it (``parent_id``) or when both chunks
        identify the same article/clause and one is structurally broader.
        """
        kept: list[tuple[int, RetrievalResult]] = []

        for candidate in candidates:
            conflicts = [
                index
                for index, existing in enumerate(kept)
                if self._hierarchy_relation(existing[1], candidate[1]) is not None
            ]
            if not conflicts:
                kept.append(candidate)
                continue

            # Compare against each conflicting representative. In normal legal
            # chunking there is usually only one explicit parent/child conflict.
            candidate_survives = True
            for index in reversed(conflicts):
                existing = kept[index]
                winner = self._choose_hierarchy_winner(
                    existing,
                    candidate,
                    rank_source=rank_source,
                )
                if winner is existing:
                    rejected.append(
                        self._rejection(
                            candidate[1],
                            candidate[0],
                            f"hierarchy_redundant_with:{existing[1].chunk_id}",
                            rank_source,
                        )
                    )
                    candidate_survives = False
                    break

                rejected.append(
                    self._rejection(
                        existing[1],
                        existing[0],
                        f"hierarchy_redundant_with:{candidate[1].chunk_id}",
                        rank_source,
                    )
                )
                kept.pop(index)

            if candidate_survives:
                kept.append(candidate)

        kept.sort(key=lambda pair: pair[0])
        return kept

    def _choose_hierarchy_winner(
        self,
        a: tuple[int, RetrievalResult],
        b: tuple[int, RetrievalResult],
        *,
        rank_source: RankSource,
    ) -> tuple[int, RetrievalResult]:
        preference = self.config.hierarchy_preference
        a_depth = self._hierarchy_depth(a[1])
        b_depth = self._hierarchy_depth(b[1])

        if preference == "more_specific":
            if a_depth != b_depth:
                return a if a_depth > b_depth else b
            return a if a[0] <= b[0] else b

        if preference == "parent":
            if a_depth != b_depth:
                return a if a_depth < b_depth else b
            return a if a[0] <= b[0] else b

        # higher_score: reranker scores are meaningful for pairwise preference;
        # on the no-rerank path, rank is safer than comparing heterogeneous
        # dense/BM25/RRF score scales.
        if rank_source == "reranker":
            a_score = float(a[1].score)
            b_score = float(b[1].score)
            if a_score != b_score:
                return a if a_score > b_score else b
        if a[0] != b[0]:
            return a if a[0] < b[0] else b
        return a if a_depth >= b_depth else b

    # ------------------------------------------------------------------
    # Metadata hierarchy helpers
    # ------------------------------------------------------------------
    def _hierarchy_relation(
        self,
        a: RetrievalResult,
        b: RetrievalResult,
    ) -> str | None:
        if a.chunk_id == b.chunk_id:
            return "same_chunk"

        a_meta = a.metadata
        b_meta = b.metadata
        a_parent = self._text(a_meta.get("parent_id"))
        b_parent = self._text(b_meta.get("parent_id"))
        if a_parent and a_parent == b.chunk_id:
            return "a_child_of_b"
        if b_parent and b_parent == a.chunk_id:
            return "b_child_of_a"

        # Metadata-only fallback when parent_id was not propagated through a
        # retriever payload. Same article + article-vs-clause/point is enough to
        # establish structural containment.
        if not self._same_document(a, b):
            return None
        if not self._same_nonempty_meta(a, b, "article_id"):
            return None

        a_level = self._chunk_level(a)
        b_level = self._chunk_level(b)
        if a_level == b_level:
            # Same clause/point only counts as hierarchy redundancy when the
            # corresponding identifier is also identical.
            if a_level == "clause" and self._same_nonempty_meta(a, b, "clause_id"):
                return "same_clause"
            if a_level == "point" and self._same_nonempty_meta(a, b, "point_id"):
                return "same_point"
            return None

        a_depth = self._hierarchy_depth(a)
        b_depth = self._hierarchy_depth(b)
        if {a_level, b_level} <= {"article", "clause", "point"}:
            # For point-vs-clause, require same clause when IDs exist.
            if "point" in {a_level, b_level} and "clause" in {a_level, b_level}:
                if not self._same_nonempty_meta(a, b, "clause_id"):
                    return None
            return "a_broader" if a_depth < b_depth else "b_broader"

        return None

    @staticmethod
    def _same_document(a: RetrievalResult, b: RetrievalResult) -> bool:
        a_doc = str(a.metadata.get("document_id") or "").strip()
        b_doc = str(b.metadata.get("document_id") or "").strip()
        return bool(a_doc and b_doc and a_doc == b_doc)

    @staticmethod
    def _same_nonempty_meta(
        a: RetrievalResult,
        b: RetrievalResult,
        key: str,
    ) -> bool:
        av = str(a.metadata.get(key) or "").strip()
        bv = str(b.metadata.get(key) or "").strip()
        return bool(av and bv and av == bv)

    def _chunk_level(self, result: RetrievalResult) -> str:
        level = str(result.metadata.get("chunk_level") or "").strip().lower()
        if level in self._LEVEL_DEPTH:
            return level
        # Infer only from explicit metadata fields, never from text syntax.
        if result.metadata.get("point_id") or result.metadata.get("point_number"):
            return "point"
        if result.metadata.get("clause_id") or result.metadata.get("clause_number"):
            return "clause"
        if result.metadata.get("article_id") or result.metadata.get("article_number"):
            return "article"
        if result.metadata.get("section_id") or result.metadata.get("section_number"):
            return "section"
        if result.metadata.get("chapter_id") or result.metadata.get("chapter_number"):
            return "chapter"
        return "document"

    def _hierarchy_depth(self, result: RetrievalResult) -> int:
        return self._LEVEL_DEPTH.get(self._chunk_level(result), 0)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _format_context(self, selected: Sequence[ContextSelection]) -> str:
        if not selected:
            return ""
        if not self.config.group_by_hierarchy:
            return "\n\n".join(self._format_flat_item(item) for item in selected).strip()

        # Group by document. In compact mode, chapter/section headers are emitted
        # only when the hierarchy actually changes, instead of repeating the full
        # path before every evidence item. Evidence boundaries remain explicit.
        by_document: dict[str, list[ContextSelection]] = {}
        document_order: list[str] = []
        for item in selected:
            key = self._document_group_key(item.result)
            if key not in by_document:
                by_document[key] = []
                document_order.append(key)
            by_document[key].append(item)

        document_blocks: list[str] = []
        for doc_key in document_order:
            items = by_document[doc_key]
            lines: list[str] = []
            if self.config.include_source_header:
                source_header = self._format_document_header(items[0].result)
                if source_header:
                    lines.append(source_header)

            if self.config.compact_hierarchy_headers:
                previous_chapter: tuple[str, str] | None = None
                previous_section: tuple[str, str] | None = None

                for item in items:
                    metadata = item.result.metadata
                    chapter = (
                        self._text(metadata.get("chapter_number")),
                        self._text(metadata.get("chapter_title")),
                    )
                    section = (
                        self._text(metadata.get("section_number")),
                        self._text(metadata.get("section_title")),
                    )

                    if chapter != previous_chapter:
                        chapter_header = self._format_compact_chapter_header(item.result)
                        if chapter_header:
                            lines.append(chapter_header)
                        previous_chapter = chapter
                        previous_section = None

                    if section != previous_section:
                        section_header = self._format_compact_section_header(item.result)
                        if section_header:
                            lines.append(section_header)
                        previous_section = section

                    lines.append(self._format_compact_evidence_item(item))
            else:
                by_article: dict[str, list[ContextSelection]] = {}
                article_order: list[str] = []
                for item in items:
                    key = self._article_group_key(item.result)
                    if key not in by_article:
                        by_article[key] = []
                        article_order.append(key)
                    by_article[key].append(item)

                for article_key in article_order:
                    article_items = by_article[article_key]
                    article_header = self._format_article_header(article_items[0].result)
                    if article_header:
                        lines.append(article_header)

                    for item in article_items:
                        lines.append(self._format_evidence_item(item))

            document_blocks.append("\n".join(part for part in lines if part).strip())

        return "\n\n".join(block for block in document_blocks if block).strip()

    def _format_compact_chapter_header(self, result: RetrievalResult) -> str:
        metadata = result.metadata
        number = self._text(metadata.get("chapter_number"))
        title = self._text(metadata.get("chapter_title"))
        if not number and not title:
            return ""
        text = "Chương"
        if number:
            text += f" {number}"
        if title:
            text += f". {title}"
        return f"## {text}"

    def _format_compact_section_header(self, result: RetrievalResult) -> str:
        metadata = result.metadata
        number = self._text(metadata.get("section_number"))
        title = self._text(metadata.get("section_title"))
        if not number and not title:
            return ""
        text = "Mục"
        if number:
            text += f" {number}"
        if title:
            text += f". {title}"
        return f"### {text}"

    def _format_compact_evidence_item(self, item: ContextSelection) -> str:
        result = item.result
        metadata = result.metadata
        marker_parts = [item.evidence_id]

        article = self._text(metadata.get("article_number"))
        clause = self._text(metadata.get("clause_number"))
        point = self._text(metadata.get("point_number"))
        if article:
            marker_parts.append(f"Điều {article}")
        if clause:
            marker_parts.append(f"Khoản {clause}")
        if point:
            marker_parts.append(f"Điểm {point}")

        marker_parts.extend(self._diagnostic_header_parts(item))
        marker = f"[{' | '.join(marker_parts)}]"
        return f"{marker}\n{result.content.strip()}"

    def _format_flat_item(self, item: ContextSelection) -> str:
        result = item.result
        header_parts = [item.evidence_id]
        legal_ref = self._legal_reference(result, include_article_title=True)
        document_title = self._text(result.metadata.get("document_title"))
        if self.config.include_source_header and document_title:
            header_parts.append(document_title)
        if legal_ref:
            header_parts.append(legal_ref)
        header_parts.extend(self._diagnostic_header_parts(item))
        return f"[{' | '.join(header_parts)}]\n{result.content.strip()}"

    def _format_document_header(self, result: RetrievalResult) -> str:
        metadata = result.metadata
        title = self._text(metadata.get("document_title"))
        if not title:
            return ""

        if not self.config.include_document_metadata:
            return f"=== {title} ==="

        details: list[str] = [title]
        number = self._text(metadata.get("document_number"))
        doc_type = self._text(metadata.get("document_type"))
        effective = self._text(metadata.get("effective_date"))
        if doc_type and doc_type.casefold() not in title.casefold():
            details.append(doc_type)
        if number:
            details.append(f"Số {number}")
        if effective:
            details.append(f"Hiệu lực {effective}")
        return f"=== {' | '.join(details)} ==="

    def _format_article_header(self, result: RetrievalResult) -> str:
        metadata = result.metadata
        parts: list[str] = []

        chapter_number = self._text(metadata.get("chapter_number"))
        chapter_title = self._text(metadata.get("chapter_title"))
        if chapter_number or chapter_title:
            chapter = "Chương"
            if chapter_number:
                chapter += f" {chapter_number}"
            if chapter_title:
                chapter += f". {chapter_title}"
            parts.append(chapter)

        section_number = self._text(metadata.get("section_number"))
        section_title = self._text(metadata.get("section_title"))
        if section_number or section_title:
            section = "Mục"
            if section_number:
                section += f" {section_number}"
            if section_title:
                section += f". {section_title}"
            parts.append(section)

        article_number = self._text(metadata.get("article_number"))
        article_title = self._text(metadata.get("article_title"))
        if article_number or article_title:
            article = "Điều"
            if article_number:
                article += f" {article_number}"
            if article_title:
                article += f". {article_title}"
            parts.append(article)

        if not parts:
            return ""
        return f"[{ ' > '.join(parts) }]"

    def _format_evidence_item(self, item: ContextSelection) -> str:
        result = item.result
        metadata = result.metadata
        marker_parts = [item.evidence_id]

        level = self._chunk_level(result)
        if level == "clause":
            clause = self._text(metadata.get("clause_number"))
            if clause:
                marker_parts.append(f"Khoản {clause}")
        elif level == "point":
            clause = self._text(metadata.get("clause_number"))
            point = self._text(metadata.get("point_number"))
            if clause:
                marker_parts.append(f"Khoản {clause}")
            if point:
                marker_parts.append(f"Điểm {point}")
        elif level not in {"article", "document"}:
            marker_parts.append(level)

        marker_parts.extend(self._diagnostic_header_parts(item))
        marker = f"[{' | '.join(marker_parts)}]"
        return f"{marker}\n{result.content.strip()}"

    def _diagnostic_header_parts(self, item: ContextSelection) -> list[str]:
        parts: list[str] = []
        if self.config.include_rank_in_context:
            label = "rerank" if item.rank_source == "reranker" else "retrieval"
            parts.append(f"{label}_rank={item.source_rank}")
        if self.config.include_score_in_context:
            score_label = "reranker_score" if item.rank_source == "reranker" else "retrieval_score"
            parts.append(f"{score_label}={float(item.result.score):.6f}")
        return parts

    # ------------------------------------------------------------------
    # Diagnostics / provenance
    # ------------------------------------------------------------------
    def _evidence_item(self, item: ContextSelection) -> EvidenceItem:
        result = item.result
        metadata = dict(result.metadata)

        retrieval_score = metadata.get("retrieval_score")
        if item.rank_source == "retrieval":
            retrieval_score = result.score

        reranker_score = result.score if item.rank_source == "reranker" else None

        def optional_float(value: Any) -> float | None:
            try:
                return None if value is None else float(value)
            except (TypeError, ValueError):
                return None

        return EvidenceItem(
            evidence_id=item.evidence_id,
            chunk_id=result.chunk_id,
            content=result.content,
            rank_source=item.rank_source,
            rank=item.source_rank,
            score=float(result.score),
            estimated_tokens=item.estimated_tokens,
            retrieval_score=optional_float(retrieval_score),
            reranker_score=optional_float(reranker_score),
            dense_score=optional_float(metadata.get("dense_score")),
            bm25_score=optional_float(metadata.get("bm25_score")),
            rrf_score=optional_float(metadata.get("rrf_score")),
            chunk_level=self._chunk_level(result),
            parent_id=self._text(metadata.get("parent_id")),
            document_id=self._text(metadata.get("document_id")),
            document_title=self._text(metadata.get("document_title")),
            document_number=self._text(metadata.get("document_number")),
            document_type=self._text(metadata.get("document_type")),
            issued_date=self._text(metadata.get("issued_date")),
            effective_date=self._text(metadata.get("effective_date")),
            source_file=self._text(metadata.get("source_file")),
            chapter_id=self._text(metadata.get("chapter_id")),
            chapter_number=self._text(metadata.get("chapter_number")),
            chapter_title=self._text(metadata.get("chapter_title")),
            section_id=self._text(metadata.get("section_id")),
            section_number=self._text(metadata.get("section_number")),
            section_title=self._text(metadata.get("section_title")),
            article_id=self._text(metadata.get("article_id")),
            article_number=self._text(metadata.get("article_number")),
            article_title=self._text(metadata.get("article_title")),
            clause_id=self._text(metadata.get("clause_id")),
            clause_number=self._text(metadata.get("clause_number")),
            point_id=self._text(metadata.get("point_id")),
            point_number=self._text(metadata.get("point_number")),
            metadata=metadata,
        )

    def _selection_methods(self, rank_source: RankSource) -> list[str]:
        methods: list[str] = []
        if rank_source == "reranker":
            methods.append("absolute_score_threshold")
            methods.append(
                "relative_to_top_threshold"
                if self.config.min_relative_to_top is not None
                else "relative_to_top_threshold_disabled"
            )
            methods.append(
                "adaptive_score_gap_cutoff"
                if self.config.score_gap is not None
                else "adaptive_score_gap_cutoff_disabled"
            )
        else:
            methods.append("retrieval_rank_selection")

        methods.extend(
            [
                "exact_deduplication",
                "metadata_hierarchy_deduplication"
                if self.config.hierarchical_deduplication
                else "metadata_hierarchy_deduplication_disabled",
                "max_chunk_cap",
                "token_budget",
                "hierarchy_grouped_rendering"
                if self.config.group_by_hierarchy
                else "flat_rendering",
                f"{self.config.order_by}_ordering",
                "typed_evidence_schema",
                "abstention_if_empty",
            ]
        )
        return methods

    def _config_diagnostics(self, retrieval_context_k: int | None) -> dict[str, Any]:
        return {
            "min_score": self.config.min_score,
            "min_relative_to_top": self.config.min_relative_to_top,
            "score_gap": self.config.score_gap,
            "min_chunks_before_gap": self.config.min_chunks_before_gap,
            "max_chunks": self.config.max_chunks,
            "max_tokens": self.config.max_tokens,
            "metadata_token_overhead": self.config.metadata_token_overhead,
            "order_by": self.config.order_by,
            "group_by_hierarchy": self.config.group_by_hierarchy,
            "compact_hierarchy_headers": self.config.compact_hierarchy_headers,
            "hierarchical_deduplication": self.config.hierarchical_deduplication,
            "hierarchy_preference": self.config.hierarchy_preference,
            "include_rank_in_context": self.config.include_rank_in_context,
            "include_score_in_context": self.config.include_score_in_context,
            "retrieval_context_k": retrieval_context_k,
        }

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_retrieval(
        query: str,
        results: Sequence[RetrievalResult],
    ) -> str:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")
        for index, result in enumerate(results):
            if not isinstance(result, RetrievalResult):
                raise TypeError(
                    f"results[{index}] must be a RetrievalResult, "
                    f"got {type(result).__name__}."
                )
            try:
                float(result.score)
            except (TypeError, ValueError) as exc:
                raise TypeError(f"results[{index}].score must be numeric.") from exc
        return query.strip()

    @staticmethod
    def _validate(
        query: str,
        results: Sequence[RetrievalResult],
    ) -> str:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")
        for index, result in enumerate(results):
            if not isinstance(result, RetrievalResult):
                raise TypeError(
                    f"results[{index}] must be a RetrievalResult, "
                    f"got {type(result).__name__}."
                )
            if not 0.0 <= float(result.score) <= 1.0:
                raise ValueError(
                    "RerankedContextBuilder expects normalized reranker scores "
                    "in [0, 1] on build(). Use build_from_retrieval() when "
                    "reranking is disabled."
                )
        return query.strip()

    @staticmethod
    def _rejection(
        result: RetrievalResult,
        rank: int,
        reason: str,
        rank_source: RankSource,
    ) -> dict[str, Any]:
        return {
            "chunk_id": result.chunk_id,
            "rank_source": rank_source,
            "rank": rank,
            "score": float(result.score),
            "reason": reason,
        }

    def _empty_result(
        self,
        query: str,
        *,
        input_count: int,
        rank_source: RankSource,
        rejected: list[dict[str, Any]] | None = None,
        top_score: float | None = None,
    ) -> ContextBuildResult:
        rejected = rejected or []
        return ContextBuildResult(
            status="insufficient_context",
            query=query,
            context="",
            selected=[],
            evidence=[],
            rejected=rejected,
            diagnostics={
                "input_count": input_count,
                "selected_count": 0,
                "rejected_count": len(rejected),
                "rank_source": rank_source,
                "top_score": top_score,
                "evidence_ids": [],
                "selection_methods": self._selection_methods(rank_source),
                "config": self._config_diagnostics(None),
            },
        )

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip().casefold()

    def _estimate_tokens(self, result: RetrievalResult) -> int:
        metadata_count = result.metadata.get("token_count")
        if isinstance(metadata_count, int) and metadata_count > 0:
            base = metadata_count
        elif isinstance(metadata_count, float) and metadata_count > 0:
            base = int(round(metadata_count))
        else:
            words = re.findall(r"\S+", result.content)
            base = max(1, int(round(len(words) * 1.35)))
        return base + self.config.metadata_token_overhead

    @classmethod
    def _number(cls, value: Any) -> int:
        """Parse numeric or Roman hierarchy numbers without reading chunk text."""
        if isinstance(value, int):
            return value
        text = str(value or "").strip().upper()
        match = re.search(r"\d+", text)
        if match:
            return int(match.group())
        if text and re.fullmatch(r"[IVXLCDM]+", text):
            roman = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
            total = 0
            previous = 0
            for char in reversed(text):
                current = roman[char]
                if current < previous:
                    total -= current
                else:
                    total += current
                    previous = current
            return total
        return 10**9

    def _legal_sort_key(self, item: ContextSelection) -> tuple[Any, ...]:
        metadata = item.result.metadata
        return (
            str(metadata.get("document_id") or metadata.get("document_title") or ""),
            self._number(metadata.get("chapter_number")),
            self._number(metadata.get("section_number")),
            self._number(metadata.get("article_number")),
            self._number(metadata.get("clause_number")),
            self._number(metadata.get("point_number")),
            item.source_rank,
        )

    @staticmethod
    def _text(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _document_group_key(self, result: RetrievalResult) -> str:
        metadata = result.metadata
        return str(
            metadata.get("document_id")
            or metadata.get("document_title")
            or result.chunk_id
        )

    def _article_group_key(self, result: RetrievalResult) -> str:
        metadata = result.metadata
        return str(
            metadata.get("article_id")
            or metadata.get("article_number")
            or metadata.get("section_id")
            or metadata.get("chapter_id")
            or result.chunk_id
        )

    def _legal_reference(
        self,
        result: RetrievalResult,
        *,
        include_article_title: bool,
    ) -> str:
        metadata = result.metadata
        article = self._text(metadata.get("article_number"))
        clause = self._text(metadata.get("clause_number"))
        point = self._text(metadata.get("point_number"))
        title = self._text(metadata.get("article_title"))

        parts: list[str] = []
        if article:
            parts.append(f"Điều {article}")
        if clause:
            parts.append(f"Khoản {clause}")
        if point:
            parts.append(f"Điểm {point}")
        reference = ", ".join(parts)
        if include_article_title and title:
            reference = f"{reference}. {title}" if reference else title
        return reference or result.chunk_id
