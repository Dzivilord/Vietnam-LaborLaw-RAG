"""Application runtime layer for serving the Labor Law RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import RAGRuntimeConfig


@dataclass
class RuntimeResult:
    answer: Any
    metadata: dict
    evidence: list[Any]


class RAGRuntime:
    """Stable entry point between API/CLI and the RAG pipeline.

    Runtime owns orchestration concerns:
    - pipeline lifecycle
    - request execution
    - response normalization
    - error boundary
    """

    def __init__(
        self,
        pipeline,
        config: RAGRuntimeConfig,
    ):
        self.pipeline = pipeline
        self.config = config

    def chat(self, query: str, **kwargs) -> RuntimeResult:
        try:
            result = self.pipeline.run(query, **kwargs)

            return RuntimeResult(
                answer=result.answer,
                metadata={
                    "status": result.status,
                    "citations": [
                        evidence.evidence_id for evidence in result.evidence
                    ],
                    "diagnostics": result.diagnostics,
                },
                evidence=list(result.evidence),
            )

        except Exception as exc:
            return RuntimeResult(
                answer=None,
                metadata={
                    "status": "error",
                    "error": str(exc),
                },
                evidence=[],
            )
