"""Classifier adapters, including OpenRouter's typed Jev Decisions API."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .rules import OUT_OF_SCOPE_PATTERNS, SENSITIVE_EVASION_PATTERNS, matched_patterns
from .schemas import QueryClassification


class BaseQueryClassifier(ABC):
    @abstractmethod
    def classify(self, query: str) -> QueryClassification:
        """Return a normalized label and confidence for one user query."""


class RuleBasedClassifier(BaseQueryClassifier):
    """Safe local baseline with only high-precision patterns."""

    def classify(self, query: str) -> QueryClassification:
        sensitive = matched_patterns(query, SENSITIVE_EVASION_PATTERNS)
        if sensitive:
            return QueryClassification(
                label="sensitive_or_disallowed", confidence=0.99,
                scores={"sensitive_or_disallowed": 0.99}, source="rules", matched_rules=sensitive,
            )
        out_of_scope = matched_patterns(query, OUT_OF_SCOPE_PATTERNS)
        if out_of_scope:
            return QueryClassification(
                label="out_of_scope", confidence=0.99,
                scores={"out_of_scope": 0.99}, source="rules", matched_rules=out_of_scope,
            )
        return QueryClassification(
            label="in_scope", confidence=0.0, scores={"in_scope": 0.0}, source="rules",
        )


class JEVClassifier(BaseQueryClassifier):
    """Adapter for a JEV classifier provided as ``query -> scores``.

    ``predict`` may call a local Python model, a local HTTP service, or a
    transformer pipeline. It must return label-to-score mappings using the
    standard guardrail labels. No hosted provider is required or assumed.
    """

    _ALIASES = {
        "in_scope": "in_scope", "labor_law": "in_scope",
        "out_of_scope": "out_of_scope", "sensitive": "sensitive_or_disallowed",
        "sensitive_or_disallowed": "sensitive_or_disallowed", "ambiguous": "ambiguous",
        "unknown": "unknown", "unknown_or_insufficient_evidence": "unknown",
    }

    def __init__(self, predict: Callable[[str], Mapping[str, float]]) -> None:
        self._predict = predict

    def classify(self, query: str) -> QueryClassification:
        raw_scores = self._predict(query)
        scores = {self._ALIASES.get(str(label), str(label)): float(score) for label, score in raw_scores.items()}
        allowed = {label: score for label, score in scores.items() if label in self._ALIASES.values()}
        if not allowed:
            raise ValueError("JEV classifier returned no supported guardrail label.")
        label, confidence = max(allowed.items(), key=lambda item: item[1])
        return QueryClassification(label=label, confidence=max(0.0, min(1.0, confidence)), scores=allowed, source="jev")


class OpenRouterJEVClassifier(BaseQueryClassifier):
    """Semantic input classifier backed by OpenRouter's Decisions API.

    The API returns a typed ``choice`` and calibrated ``confidence`` instead
    of generated prose. API credentials are read only from the process
    environment (normally ``OPENROUTER_API_KEY`` loaded from ``.env``).
    """

    DEFAULT_URL = "https://openrouter.ai/api/alpha/decisions"
    DEFAULT_MODEL = "typesafe/jev-1.13"
    QUESTION_ID = "guardrail_category"
    CRITERIA = {
        "in_scope": "A legitimate question about Vietnamese labor law, employment rights, obligations, or an employment relationship.",
        "out_of_scope": "Not related to Vietnamese labor law or employment, such as weather, cooking, travel, programming, sport, or medical advice.",
        "sensitive_or_disallowed": "Seeks to evade legal duties, conceal violations, coerce or discriminate against workers, forge documents, or otherwise cause harm.",
        "ambiguous": "Related to labor law but lacks material facts needed to apply law to the person's situation.",
        "unknown": "Related to labor law but likely requires a source outside the configured Labor Code corpus, such as a future schedule, a calculation under another law, or an administrative fine amount.",
    }

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 15.0,
        referer: str | None = None,
        title: str | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("Missing OPENROUTER_API_KEY for OpenRouter Jev guardrail.")
        self.model = model or self.DEFAULT_MODEL
        self.base_url = (base_url or self.DEFAULT_URL).rstrip("/")
        self.timeout = timeout
        self.referer = referer
        self.title = title

    def classify(self, query: str) -> QueryClassification:
        payload = {
            "model": self.model,
            "state": {
                "query": query,
                "system_scope": "The application answers questions only from the Vietnamese Labor Code 2019 corpus. It has no web lookup or human review enabled.",
            },
            "questions": {
                self.QUESTION_ID: {
                    "type": "choice",
                    "instructions": "Classify the user's query into exactly one guardrail category. Choose sensitive_or_disallowed only for harmful or evasive intent, not for ordinary legal compliance questions.",
                    "criteria": self.CRITERIA,
                }
            },
        }
        response = self._post(payload)
        answer = (response.get("answers") or {}).get(self.QUESTION_ID) or {}
        choice = str(answer.get("choice") or "")
        if choice not in self.CRITERIA:
            raise RuntimeError(f"OpenRouter Jev returned unsupported choice: {choice or '<empty>'}.")
        confidence = float(answer.get("confidence", answer.get("score", 0.0)))
        raw_scores = answer.get("probabilities") or answer.get("scores") or {}
        scores = {str(label): float(score) for label, score in raw_scores.items() if str(label) in self.CRITERIA}
        scores.setdefault(choice, confidence)
        return QueryClassification(
            label=choice, confidence=max(0.0, min(1.0, confidence)), scores=scores,
            source="openrouter_jev",
        )

    def _post(self, payload: dict) -> dict:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        if self.referer:
            headers["HTTP-Referer"] = self.referer
        if self.title:
            headers["X-Title"] = self.title
        request = Request(self.base_url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        try:
            with urlopen(request, timeout=self.timeout) as raw_response:
                return json.loads(raw_response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenRouter Jev HTTP {exc.code}: {detail or exc.reason}") from exc
        except URLError as exc:
            raise RuntimeError(f"Cannot connect to OpenRouter Jev: {exc.reason}") from exc
