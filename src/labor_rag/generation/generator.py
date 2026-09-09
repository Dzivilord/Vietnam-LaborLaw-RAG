"""Model-independent RAG answer generation."""

from __future__ import annotations

import re
import time

from .base import BaseLLMProvider
from .prompts import SYSTEM_PROMPT, build_user_prompt
from .schemas import GenerationConfig, GenerationRequest, GenerationResult


_EVIDENCE_RE = re.compile(r"\[(E\d+)\]", re.IGNORECASE)


class RAGGenerator:
    """Generate a grounded answer from a finished Context Builder result."""

    def __init__(
        self,
        provider: BaseLLMProvider,
        config: GenerationConfig | None = None,
    ) -> None:
        self.provider = provider
        self.config = config or GenerationConfig()

    def generate(self, request: GenerationRequest) -> GenerationResult:
        query = request.query.strip()
        context = request.context.strip()

        if not context:
            return GenerationResult(
                status="abstained",
                query=query,
                answer=(
                    "Không có đủ ngữ cảnh đã được xác thực để tạo câu trả lời. "
                    "Generation không gọi mô hình."
                ),
                provider=self.provider.provider_name,
                model=self.provider.model_name,
                latency_ms=0.0,
                usage={},
                evidence=list(request.evidence),
                citations_used=[],
                citation_warnings=["empty_context"],
                source_metadata=dict(request.source_metadata),
            )

        started = time.perf_counter()
        response = self.provider.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=build_user_prompt(query=query, context=context),
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_output_tokens,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0

        answer = response.text.strip()
        if not answer:
            raise RuntimeError(
                f"{response.provider} returned an empty generation response."
            )

        available_ids = {item.evidence_id.upper() for item in request.evidence}
        if not available_ids:
            available_ids = {
                match.upper() for match in _EVIDENCE_RE.findall(context)
            }

        citations_used, warnings = self._validate_citations(answer, available_ids)
        status = "generated_with_warnings" if warnings else "generated"

        return GenerationResult(
            status=status,
            query=query,
            answer=answer,
            provider=response.provider,
            model=response.model,
            latency_ms=latency_ms,
            usage=dict(response.usage),
            evidence=list(request.evidence),
            citations_used=citations_used,
            citation_warnings=warnings,
            source_metadata=dict(request.source_metadata),
        )

    def _validate_citations(
        self,
        answer: str,
        available_ids: set[str],
    ) -> tuple[list[str], list[str]]:
        citations = [match.upper() for match in _EVIDENCE_RE.findall(answer)]
        citations = list(dict.fromkeys(citations))
        warnings: list[str] = []

        invalid = [citation for citation in citations if citation not in available_ids]
        if invalid:
            warnings.append("invalid_citations:" + ",".join(invalid))

        if self.config.require_citations and available_ids and not citations:
            warnings.append("missing_citations")

        return citations, warnings
