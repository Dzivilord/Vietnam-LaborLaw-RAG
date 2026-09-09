"""Optional query-ambiguity classifier grounded by retrieved legal topics.

The classifier runs only after reranking/context relevance filtering has shown
that several strong chunks exist. Its job is not to judge whether evidence is
relevant; reranking has already done that. Instead it compares the user's query
against the *facets exposed by the relevant chunk titles* and decides whether
the query is:

* ``specific``      - it points to a concrete facet/question;
* ``comprehensive`` - it explicitly asks for all/multiple facets;
* ``ambiguous``     - it names a broad topic while several distinct facets are
  available and the intended facet is still unclear.

The default implementation uses a small multilingual NLI model. Model loading
is lazy, so no extra model cost is paid unless the pipeline activates the
ambiguity feature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

from labor_rag.retrieval.base import RetrievalResult


AmbiguityLabel = Literal["specific", "comprehensive", "ambiguous"]


@dataclass(slots=True, frozen=True)
class QueryAmbiguityConfig:
    """Configuration for optional topic-grounded zero-shot classification."""

    model_name: str = "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"
    device: str | int | None = None
    min_decision_score: float = 0.45
    min_decision_margin: float = 0.08
    max_titles: int = 8
    include_legal_reference: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_decision_score <= 1.0:
            raise ValueError("min_decision_score must be between 0 and 1.")
        if not 0.0 <= self.min_decision_margin <= 1.0:
            raise ValueError("min_decision_margin must be between 0 and 1.")
        if self.max_titles <= 0:
            raise ValueError("max_titles must be greater than 0.")


@dataclass(slots=True, frozen=True)
class Suggestion:
    """Grounded clarification suggestion derived from one retrieved title."""

    label: str
    chunk_id: str
    reranker_score: float
    article_number: str | None = None
    clause_number: str | None = None


@dataclass(slots=True, frozen=True)
class AmbiguityDecision:
    """Structured classifier output consumed by pipeline orchestration."""

    # Final pipeline-safe decision. Low-confidence predictions fall back to
    # ambiguous so the system asks for clarification instead of guessing.
    label: AmbiguityLabel
    score: float
    raw_scores: dict[AmbiguityLabel, float]
    model_name: str
    low_confidence: bool

    # Diagnostics: what the model itself preferred before safety fallback and
    # what topic evidence it was given.
    predicted_label: AmbiguityLabel
    predicted_score: float
    decision_margin: float
    topics: tuple[str, ...]
    suggestions: tuple[Suggestion, ...] = ()

    @property
    def requires_clarification(self) -> bool:
        return self.label == "ambiguous"


class QueryAmbiguityClassifier:
    """Classify specificity by explicitly comparing query with chunk topics."""

    # These labels intentionally describe the relation between QUERY and TOPICS,
    # not just generic linguistic properties of the query.
    _LABEL_TEXT: dict[AmbiguityLabel, str] = {
        "specific": (
            "người dùng đã chỉ rõ một câu hỏi hoặc một khía cạnh cụ thể trong "
            "các chủ đề tài liệu được liệt kê, nên có thể trả lời trực tiếp"
        ),
        "comprehensive": (
            "người dùng chủ động yêu cầu tổng hợp toàn bộ hoặc nhiều khía cạnh "
            "trong các chủ đề tài liệu được liệt kê"
        ),
        "ambiguous": (
            "người dùng chỉ nêu chủ đề chung hoặc chưa chỉ rõ muốn khía cạnh nào, "
            "trong khi các tài liệu liên quan cho thấy có nhiều khía cạnh khác nhau"
        ),
    }

    def __init__(self, config: QueryAmbiguityConfig | None = None) -> None:
        self.config = config or QueryAmbiguityConfig()
        self._classifier: Any | None = None

    def _ensure_model(self) -> None:
        if self._classifier is not None:
            return

        try:
            import torch
            from transformers import pipeline
        except ImportError as exc:
            raise ImportError(
                "QueryAmbiguityClassifier requires torch and transformers."
            ) from exc

        device = self.config.device
        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        elif isinstance(device, str):
            normalized = device.strip().lower()
            if normalized == "cpu":
                device = -1
            elif normalized == "cuda":
                device = 0
            elif normalized.startswith("cuda:"):
                device = int(normalized.split(":", 1)[1])
            else:
                raise ValueError(
                    "device must be None, an integer, 'cpu', 'cuda', or 'cuda:N'."
                )

        self._classifier = pipeline(
            "zero-shot-classification",
            model=self.config.model_name,
            device=device,
        )

    def classify(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
    ) -> AmbiguityDecision:
        query = self._validate(query, candidates)
        if not candidates:
            raise ValueError("candidates must not be empty.")

        self._ensure_model()
        if self._classifier is None:
            raise RuntimeError("Ambiguity classifier was not initialized correctly.")

        topics = tuple(self._candidate_titles(candidates))
        sequence = self._build_sequence(query, topics)

        # multi_label=True is deliberate. With multi_label=False, transformers
        # normalizes all three labels into a forced sum of 1, which often gives
        # misleading ~0.33/~0.33/~0.33 outputs even when the model is simply
        # uncertain. Independent entailment scores are more useful here.
        candidate_labels = list(self._LABEL_TEXT.values())
        output = self._classifier(
            sequence,
            candidate_labels=candidate_labels,
            multi_label=True,
            hypothesis_template=(
                "Kết luận đúng khi so sánh truy vấn với các chủ đề tài liệu là: {}."
            ),
        )

        score_by_text = {
            str(label): float(score)
            for label, score in zip(
                output.get("labels", []),
                output.get("scores", []),
                strict=True,
            )
        }
        raw_scores: dict[AmbiguityLabel, float] = {
            label: float(score_by_text.get(text, 0.0))
            for label, text in self._LABEL_TEXT.items()
        }

        ranked = sorted(raw_scores.items(), key=lambda item: item[1], reverse=True)
        predicted_label, predicted_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        decision_margin = max(0.0, float(predicted_score - second_score))

        low_confidence = (
            predicted_score < self.config.min_decision_score
            or decision_margin < self.config.min_decision_margin
        )

        # Fail safe: when a classifier invoked specifically because many strong
        # facets exist cannot confidently distinguish them, clarification is
        # safer than silently forcing SPECIFIC and sending broad context to the
        # generator.
        final_label: AmbiguityLabel = (
            "ambiguous" if low_confidence else predicted_label
        )
        final_score = raw_scores[final_label]

        suggestions: tuple[Suggestion, ...] = ()
        if final_label == "ambiguous":
            suggestions = tuple(self.build_suggestions(candidates))

        return AmbiguityDecision(
            label=final_label,
            score=float(final_score),
            raw_scores=raw_scores,
            model_name=self.config.model_name,
            low_confidence=low_confidence,
            predicted_label=predicted_label,
            predicted_score=float(predicted_score),
            decision_margin=decision_margin,
            topics=topics,
            suggestions=suggestions,
        )

    def build_suggestions(
        self,
        candidates: Sequence[RetrievalResult],
    ) -> list[Suggestion]:
        """Build deterministic suggestions from grounded metadata titles only."""
        suggestions: list[Suggestion] = []
        seen: set[str] = set()

        for candidate in candidates[: self.config.max_titles]:
            metadata = candidate.metadata
            title = str(
                metadata.get("article_title")
                or metadata.get("title")
                or metadata.get("section_title")
                or ""
            ).strip()
            if not title:
                continue

            normalized = " ".join(title.casefold().split())
            if normalized in seen:
                continue
            seen.add(normalized)

            article = self._optional_string(metadata.get("article_number"))
            clause = self._optional_string(metadata.get("clause_number"))
            label = title
            if self.config.include_legal_reference and article:
                legal_ref = f"Điều {article}"
                if clause:
                    legal_ref += f", khoản {clause}"
                label = f"{title} ({legal_ref})"

            suggestions.append(
                Suggestion(
                    label=label,
                    chunk_id=candidate.chunk_id,
                    reranker_score=float(candidate.score),
                    article_number=article,
                    clause_number=clause,
                )
            )

        return suggestions

    def _candidate_titles(
        self,
        candidates: Sequence[RetrievalResult],
    ) -> list[str]:
        return [suggestion.label for suggestion in self.build_suggestions(candidates)]

    @staticmethod
    def _build_sequence(query: str, topics: Sequence[str]) -> str:
        rendered_topics = "\n".join(
            f"{index}. {topic}" for index, topic in enumerate(topics, start=1)
        )
        if not rendered_topics:
            rendered_topics = "1. Không có tiêu đề metadata."

        return (
            "Hãy SO SÁNH trực tiếp truy vấn của người dùng với danh sách các chủ đề "
            "mà hệ thống đã tìm thấy là liên quan.\n\n"
            f"TRUY VẤN:\n{query}\n\n"
            "CÁC CHỦ ĐỀ/FACET CÓ THỂ TRẢ LỜI TỪ TÀI LIỆU:\n"
            f"{rendered_topics}\n\n"
            "Quy tắc đánh giá:\n"
            "- SPECIFIC nếu truy vấn đã chỉ rõ câu hỏi hoặc facet cần biết.\n"
            "- COMPREHENSIVE nếu truy vấn chủ động yêu cầu toàn bộ/tổng hợp nhiều facet.\n"
            "- AMBIGUOUS nếu truy vấn chỉ nêu chủ đề chung, trong khi danh sách trên "
            "có nhiều facet và chưa biết người dùng muốn facet nào.\n"
            "Không đánh giá độ đúng của tài liệu và không tự trả lời câu hỏi."
        )

    @staticmethod
    def _validate(
        query: str,
        candidates: Sequence[RetrievalResult],
    ) -> str:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")
        for index, candidate in enumerate(candidates):
            if not isinstance(candidate, RetrievalResult):
                raise TypeError(
                    f"candidates[{index}] must be a RetrievalResult, "
                    f"got {type(candidate).__name__}."
                )
        return query.strip()

    @staticmethod
    def _optional_string(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None
