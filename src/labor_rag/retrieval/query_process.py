"""Offline query understanding for Vietnamese legal questions.

Three deterministic steps, no LLM and no network:

1. **Article references** - ``Điều 35``, ``khoản 2 Điều 107``, ``Điều 98 khoản 1``
   are parsed so the matching chunk can be pinned instead of hoping that
   semantic search finds it.
2. **Normalisation** - common abbreviations (``NLĐ``, ``cty``, ``ko``) are
   expanded and, for queries typed without diacritics, diacritics are restored
   from n-grams observed in the corpus itself.
3. **Legal-term expansion** - everyday wording (``đuổi việc``, ``nghỉ đẻ``,
   ``tăng ca``) is mapped to the terminology the statute actually uses and
   appended to the search text.

``QueryAwareRetriever`` wraps any ``BaseRetriever``.  The reranker and the
generator should keep receiving the *original* user query; only retrieval uses
``ProcessedQuery.search_text``.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .base import BaseRetriever, RetrievalResult

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def fold(text: str) -> str:
    """Lower-case and strip Vietnamese diacritics (``đ`` -> ``d``)."""
    lowered = unicodedata.normalize("NFC", text).lower().replace("đ", "d")
    decomposed = unicodedata.normalize("NFD", lowered)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


# ----------------------------------------------------------------------------
# Resources (kept small and explicit so they can be reviewed by a lawyer).
# ----------------------------------------------------------------------------

# Keys are folded tokens.
ABBREVIATIONS: dict[str, str] = {
    "cty": "công ty",
    "nld": "người lao động",
    "nsdld": "người sử dụng lao động",
    "hdld": "hợp đồng lao động",
    "hd": "hợp đồng",
    "blld": "bộ luật lao động",
    "bhxh": "bảo hiểm xã hội",
    "bhyt": "bảo hiểm y tế",
    "bhtn": "bảo hiểm thất nghiệp",
    "ko": "không",
    "khong": "không",
    "dc": "được",
}

# (folded regex, legal terms appended to the search text).  Matching is done on
# the folded query so it works with or without diacritics.
LEGAL_TERM_EXPANSIONS: tuple[tuple[str, str], ...] = (
    (r"\b(?:duoi viec|cho nghi viec|cho thoi viec|duoi)\b", "đơn phương chấm dứt hợp đồng lao động"),
    (r"\b(?:nghi de|nghi sinh|sinh con)\b", "nghỉ thai sản"),
    (r"\b(?:tang ca|lam them|ot)\b", "làm thêm giờ"),
    (r"\b(?:nghi phep|phep nam)\b", "nghỉ hằng năm"),
    (r"\b(?:nghi ma|dam tang|nguoi than mat|dam cuoi|ket hon)\b", "nghỉ việc riêng"),
    (r"\b(?:tien coc|dat coc|the chap)\b", "biện pháp bảo đảm bằng tiền hoặc tài sản"),
    (r"\b(?:giu chung minh|giu bang goc|giu bang|giu cmnd|giu can cuoc)\b", "giữ bản chính giấy tờ tùy thân văn bằng chứng chỉ"),
    (r"\b(?:nghi ngang|bo viec|nghi viec ngay)\b", "đơn phương chấm dứt hợp đồng lao động báo trước"),
    (r"\b(?:tru luong|cat luong|khau tru)\b", "khấu trừ tiền lương"),
    (r"\b(?:no luong|tra luong cham|cham luong|tre luong)\b", "không được trả đủ lương trả lương không đúng thời hạn"),
    (r"\b(?:nghi cho viec|cho viec|ngung viec)\b", "tiền lương ngừng việc"),
    (r"\b(?:giup viec|osin)\b", "lao động là người giúp việc gia đình"),
    # "thuong" alone is avoided: folded, it collides with "thương lượng".
    (r"\b(?:thuong tet|thang 13|tien thuong|quy che thuong)\b", "thưởng quy chế thưởng"),
    (r"\bchu nhat\b", "ngày nghỉ hằng tuần"),
    (r"\b(?:ngay le|nghi le|le tet)\b", "ngày nghỉ lễ tết"),
    (r"\b(?:ca dem|lam dem)\b", "làm việc vào ban đêm"),
    (r"\b(?:bao hiem|bhxh)\b", "bảo hiểm xã hội bảo hiểm y tế bảo hiểm thất nghiệp"),
    (r"\b(?:doi luong|kien cong ty|khoi kien)\b", "tranh chấp lao động"),
    (r"\b(?:ve huu|nghi huu)\b", "tuổi nghỉ hưu"),
    (r"\b(?:nghi tap the)\b", "đình công"),
    (r"\b(?:cong tac vien|freelance)\b", "hợp đồng lao động việc làm có trả công"),
    (r"\b(?:nghi trua|nghi giua gio)\b", "nghỉ giữa giờ làm việc"),
    (r"\b(?:tai nan lao dong|bi tai nan)\b", "tai nạn lao động điều trị"),
)

# folded "dieu N", with optional "khoan M" before or after, optional "diem x".
_REF_CLAUSE_FIRST = re.compile(r"\bkhoan\s+(\d{1,2})\s+(?:cua\s+)?dieu\s+(\d{1,3})\b")
_REF_ARTICLE_FIRST = re.compile(r"\bdieu\s+(\d{1,3})\s*(?:,|\s)?\s*khoan\s+(\d{1,2})\b")
_REF_ARTICLE_ONLY = re.compile(r"\bdieu\s+(\d{1,3})\b")


@dataclass(frozen=True, slots=True)
class ArticleRef:
    article: str
    clause: str | None = None


@dataclass(frozen=True, slots=True)
class ProcessedQuery:
    original: str
    search_text: str
    article_refs: tuple[ArticleRef, ...] = ()
    expansions: tuple[str, ...] = ()
    abbreviations_expanded: bool = False
    diacritics_restored: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "search_text": self.search_text,
            "article_refs": [{"article": r.article, "clause": r.clause} for r in self.article_refs],
            "expansions": list(self.expansions),
            "abbreviations_expanded": self.abbreviations_expanded,
            "diacritics_restored": self.diacritics_restored,
        }


def parse_article_refs(query: str) -> tuple[ArticleRef, ...]:
    """Extract ``Điều N`` / ``khoản M Điều N`` references from a query."""
    text = fold(query)
    refs: list[ArticleRef] = []
    consumed: list[tuple[int, int]] = []

    def free(span: tuple[int, int]) -> bool:
        return all(span[1] <= a or span[0] >= b for a, b in consumed)

    for pattern, clause_group, article_group in (
        (_REF_CLAUSE_FIRST, 1, 2),
        (_REF_ARTICLE_FIRST, 2, 1),
    ):
        for match in pattern.finditer(text):
            if free(match.span()):
                consumed.append(match.span())
                refs.append(ArticleRef(match.group(article_group), match.group(clause_group)))
    for match in _REF_ARTICLE_ONLY.finditer(text):
        if free(match.span()):
            consumed.append(match.span())
            refs.append(ArticleRef(match.group(1)))
    return tuple(dict.fromkeys(refs))


class QueryProcessor:
    """Normalise and expand a query; learn diacritic restoration from the corpus."""

    def __init__(
        self,
        corpus_texts: Iterable[str] = (),
        *,
        expand_terms: bool = True,
        restore_diacritics: bool = True,
        expand_abbreviations: bool = True,
        max_ngram: int = 3,
        min_vocab_coverage: float = 0.6,
    ) -> None:
        self.expand_terms = expand_terms
        self.restore_diacritics = restore_diacritics
        self.expand_abbreviations = expand_abbreviations
        self.max_ngram = max_ngram
        self.min_vocab_coverage = min_vocab_coverage
        self._ngrams: dict[str, Counter[str]] = defaultdict(Counter)
        self._compiled = [(re.compile(p), terms) for p, terms in LEGAL_TERM_EXPANSIONS]
        if restore_diacritics:
            self._learn(corpus_texts)

    # ---- corpus statistics ------------------------------------------------
    def _learn(self, texts: Iterable[str]) -> None:
        for text in texts:
            tokens = _tokens(text)
            for n in range(1, self.max_ngram + 1):
                for i in range(len(tokens) - n + 1):
                    window = tokens[i : i + n]
                    self._ngrams[" ".join(fold(t) for t in window)][" ".join(window)] += 1

    @staticmethod
    def _has_diacritics(text: str) -> bool:
        return fold(text) != unicodedata.normalize("NFC", text).lower()

    def _restore(self, tokens: list[str]) -> list[str]:
        known = sum(1 for t in tokens if fold(t) in self._ngrams)
        if len(tokens) < 2 or not self._ngrams or known / len(tokens) < self.min_vocab_coverage:
            return tokens
        out: list[str] = []
        i = 0
        while i < len(tokens):
            for n in range(min(self.max_ngram, len(tokens) - i), 0, -1):
                key = " ".join(fold(t) for t in tokens[i : i + n])
                candidates = self._ngrams.get(key)
                if candidates:
                    out.extend(candidates.most_common(1)[0][0].split())
                    i += n
                    break
            else:
                out.append(tokens[i])
                i += 1
        return out

    # ---- public API -------------------------------------------------------
    def process(self, query: str) -> ProcessedQuery:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")
        original = " ".join(query.split())
        refs = parse_article_refs(original)

        tokens = _tokens(original)
        had_diacritics = self._has_diacritics(original)

        abbreviated = False
        if self.expand_abbreviations:
            expanded: list[str] = []
            for token in tokens:
                replacement = ABBREVIATIONS.get(fold(token))
                # "khong"/"dc" etc. are only treated as abbreviations when the
                # user clearly typed without diacritics.
                if replacement and (fold(token) not in {"khong"} or not had_diacritics):
                    if replacement != token:
                        abbreviated = True
                    expanded.extend(replacement.split())
                else:
                    expanded.append(token)
            tokens = expanded

        restored = False
        if self.restore_diacritics and not had_diacritics:
            new_tokens = self._restore(tokens)
            restored = new_tokens != tokens
            tokens = new_tokens

        base_text = " ".join(tokens)
        folded_text = fold(base_text)
        added: list[str] = []
        if self.expand_terms:
            for pattern, terms in self._compiled:
                if pattern.search(folded_text) and fold(terms) not in folded_text:
                    added.append(terms)
        search_text = " ".join([base_text, *added]).strip()
        return ProcessedQuery(
            original=original,
            search_text=search_text,
            article_refs=refs,
            expansions=tuple(added),
            abbreviations_expanded=abbreviated,
            diacritics_restored=restored,
        )


class QueryAwareRetriever(BaseRetriever):
    """Wrap a retriever with query rewriting and direct article lookup."""

    def __init__(
        self,
        base: BaseRetriever,
        chunks: Sequence[dict[str, Any]],
        *,
        processor: QueryProcessor | None = None,
        article_lookup: bool = True,
        pinned_score: float = 1.0,
    ) -> None:
        self.base = base
        self.processor = processor or QueryProcessor(c.get("content", "") for c in chunks)
        self.article_lookup = article_lookup
        self.pinned_score = pinned_score
        self._by_article: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for chunk in chunks:
            number = str(chunk.get("article_number") or "").strip()
            if number:
                self._by_article[number].append(dict(chunk))
        self.last_processed: ProcessedQuery | None = None

    def retrieve(self, query: str, *, top_k: int = 5) -> list[RetrievalResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")
        processed = self.processor.process(query)
        self.last_processed = processed

        pinned: list[RetrievalResult] = []
        if self.article_lookup and processed.article_refs:
            for ref in processed.article_refs:
                pinned.extend(self._resolve(ref))
            if not pinned:
                # The user named a provision this corpus does not contain
                # (e.g. "Điều 300"): returning look-alike chunks would mislead.
                return []

        results = self.base.retrieve(processed.search_text, top_k=top_k)
        seen = {item.chunk_id for item in pinned}
        merged = pinned + [item for item in results if item.chunk_id not in seen]
        for item in merged:
            item.metadata.setdefault("query_processing", processed.as_dict())
        return merged[:top_k]

    def _resolve(self, ref: ArticleRef) -> list[RetrievalResult]:
        chunks = self._by_article.get(ref.article, [])
        if not chunks:
            return []
        if ref.clause:
            exact = [c for c in chunks if str(c.get("clause_number") or "") == ref.clause]
            # Fall back to the whole article when it is stored as one chunk.
            selected = exact or [c for c in chunks if c.get("chunk_level") == "article"] or []
        else:
            selected = [c for c in chunks if c.get("chunk_level") == "article"] or chunks
        results = []
        for chunk in selected:
            metadata = dict(chunk)
            metadata["retrieval_method"] = "article_lookup"
            results.append(
                RetrievalResult(
                    chunk_id=str(chunk["chunk_id"]),
                    content=str(chunk.get("content") or ""),
                    score=self.pinned_score,
                    metadata=metadata,
                )
            )
        return results
