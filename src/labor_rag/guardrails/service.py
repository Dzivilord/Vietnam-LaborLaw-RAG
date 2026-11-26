"""Policy engine composing rules, an optional JEV classifier, and RAG gates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from labor_rag.generation.schemas import GenerationResult
from labor_rag.models.rag import EvidenceItem

from .classifier import BaseQueryClassifier, RuleBasedClassifier
from .schemas import GuardrailDecision


# clarify                 - stop before retrieval with a generic "add details" message
# clarify_after_retrieval - flag the query, still retrieve, then return grounded topic
#                           suggestions + the related evidence instead of an answer
# broad_context           - ignore the ambiguity label and answer from broad context
AmbiguityMode = Literal["clarify", "clarify_after_retrieval", "broad_context"]
# abstain  - questions labelled "unknown" (need a source outside the corpus) stop early
# continue - let them reach retrieval; the evidence gate may still abstain
UnknownAction = Literal["abstain", "continue"]


@dataclass(frozen=True, slots=True)
class GuardrailSettings:
    enabled: bool = True
    scope_enabled: bool = True
    sensitive_enabled: bool = True
    rules_enabled: bool = True
    classifier_threshold: float = 0.80
    # Per-label overrides of classifier_threshold, e.g. {"unknown": 0.35}. A
    # 5-way classifier is naturally less confident on the subtle labels.
    label_thresholds: dict[str, float] = field(default_factory=dict)
    ambiguity_enabled: bool = True
    ambiguity_mode: AmbiguityMode = "broad_context"
    unknown_action: UnknownAction = "abstain"
    evidence_quality_enabled: bool = True
    handoff_eligible: bool = True
    output_enabled: bool = True
    require_citations: bool = True
    reject_invalid_citations: bool = True


class GuardrailService:
    """Decide whether a query may proceed, and whether an answer may ship."""

    def __init__(self, settings: GuardrailSettings | None = None, classifier: BaseQueryClassifier | None = None) -> None:
        self.settings = settings or GuardrailSettings()
        self.classifier = classifier or RuleBasedClassifier()
        self.rule_classifier = RuleBasedClassifier()

    def threshold_for(self, label: str) -> float:
        return float(self.settings.label_thresholds.get(label, self.settings.classifier_threshold))

    def check_input(self, query: str) -> GuardrailDecision:
        if not self.settings.enabled:
            return GuardrailDecision("continue", "disabled", "guardrails_disabled")

        # Rules are evaluated first because they are high precision and should
        # override a semantic model only for explicit prohibited intent.
        rule_result = self.rule_classifier.classify(query) if self.settings.rules_enabled else None
        classified = rule_result if rule_result and rule_result.confidence else self.classifier.classify(query)
        metadata = {"classifier": classified.source, "scores": dict(classified.scores), "matched_rules": list(classified.matched_rules)}

        confident = classified.confidence >= self.threshold_for(classified.label)
        if self.settings.sensitive_enabled and classified.label == "sensitive_or_disallowed" and confident:
            return GuardrailDecision("refuse", classified.label, "sensitive_or_disallowed", classified.confidence, metadata=metadata)
        if self.settings.scope_enabled and classified.label == "out_of_scope" and confident:
            return GuardrailDecision("abstain", classified.label, "out_of_scope", classified.confidence, metadata=metadata)
        if self.settings.scope_enabled and self.settings.unknown_action == "abstain" and classified.label == "unknown" and confident:
            return GuardrailDecision(
                "abstain", "unknown", "requires_source_outside_corpus", classified.confidence,
                handoff_eligible=self.settings.handoff_eligible, metadata=metadata,
            )
        if self.settings.ambiguity_enabled and classified.label == "ambiguous" and confident:
            if self.settings.ambiguity_mode == "clarify":
                return GuardrailDecision("clarify", classified.label, "ambiguous_query", classified.confidence, metadata=metadata)
            if self.settings.ambiguity_mode == "clarify_after_retrieval":
                # Continue so retrieval can ground the clarification in real articles;
                # the application pipeline turns this flag into clarification_required.
                return GuardrailDecision(
                    "continue", classified.label, "ambiguous_clarify_after_retrieval", classified.confidence,
                    metadata={**metadata, "clarify_after_retrieval": True},
                )
        # broad_context: ambiguity is not blocked; evidence may still support a general answer.
        return GuardrailDecision("continue", classified.label, "input_allowed", classified.confidence, metadata=metadata)

    def check_evidence(self, evidence: list[EvidenceItem]) -> GuardrailDecision:
        if not self.settings.enabled or not self.settings.evidence_quality_enabled:
            return GuardrailDecision("continue", "disabled", "evidence_gate_disabled")
        if not evidence:
            return GuardrailDecision(
                "abstain", "insufficient_evidence", "no_evidence_after_context_selection",
                handoff_eligible=self.settings.handoff_eligible,
            )
        return GuardrailDecision("continue", "sufficient_evidence", "evidence_available", metadata={"evidence_count": len(evidence)})

    def check_output(self, result: GenerationResult) -> GuardrailDecision:
        if not self.settings.enabled or not self.settings.output_enabled:
            return GuardrailDecision("continue", "disabled", "output_gate_disabled")
        warnings = set(result.citation_warnings)
        has_invalid = any(warning.startswith("invalid_citations:") for warning in warnings)
        has_missing = "missing_citations" in warnings
        if self.settings.reject_invalid_citations and has_invalid:
            return GuardrailDecision("abstain", "invalid_citation", "citation_not_in_evidence", metadata={"warnings": sorted(warnings)})
        if self.settings.require_citations and has_missing:
            return GuardrailDecision("abstain", "missing_citation", "answer_missing_required_citation", metadata={"warnings": sorted(warnings)})
        return GuardrailDecision("continue", "output_valid", "citations_valid", metadata={"warnings": sorted(warnings)})


def user_message_for(decision: GuardrailDecision) -> str:
    if decision.category == "out_of_scope":
        return "Hệ thống này chỉ hỗ trợ truy vấn Bộ luật Lao động 2019 và không hỗ trợ các nội dung không liên quan."
    if decision.category == "sensitive_or_disallowed":
        return "Tôi không thể hỗ trợ yêu cầu nhằm né tránh nghĩa vụ hoặc gây hại. Tôi có thể hỗ trợ giải thích quy định và phương án tuân thủ pháp luật."
    if decision.category == "unknown":
        return ("Câu hỏi này cần căn cứ ngoài Bộ luật Lao động 2019 (ví dụ nghị định hướng dẫn, mức phạt, "
                "bảo hiểm xã hội hoặc số liệu cập nhật), nên hệ thống không trả lời để tránh thông tin sai.")
    if decision.action == "clarify" or decision.category == "ambiguous":
        return "Câu hỏi cần thêm thông tin tình huống để áp dụng quy định phù hợp. Vui lòng nêu rõ loại hợp đồng, thời điểm và hoàn cảnh liên quan."
    if decision.category in {"insufficient_evidence", "invalid_citation", "missing_citation"}:
        return "Không tìm thấy đủ căn cứ trong Bộ luật Lao động 2019 để trả lời nội dung này một cách đáng tin cậy."
    return "Yêu cầu không thể được xử lý ở thời điểm này."
