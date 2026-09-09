from __future__ import annotations

from labor_rag.generation import RAGGenerator
from labor_rag.generation.base import BaseLLMProvider, ProviderResponse
from labor_rag.pipelines.application_pipeline import LaborLawRAGPipeline
from labor_rag.pipelines.rag_pipeline import (
    RetrievalContextPipeline,
    RetrievalContextPipelineConfig,
)
from labor_rag.reranking.base import BaseReranker
from labor_rag.reranking.context_builder import RerankedContextBuilder
from labor_rag.retrieval.base import BaseRetriever, RetrievalResult


class _Retriever(BaseRetriever):
    def retrieve(self, query: str, *, top_k: int = 5) -> list[RetrievalResult]:
        return [
            RetrievalResult(
                chunk_id="article-24",
                content="Điều 24. Thử việc",
                score=0.80,
                metadata={
                    "document_title": "Bộ luật Lao động 2019",
                    "article_number": "24",
                    "article_title": "Thử việc",
                    "source_file": "labor_law.txt",
                },
            )
        ]


class _Reranker(BaseReranker):
    def rerank(self, query, candidates, *, top_k=None):
        item = candidates[0]
        return [
            RetrievalResult(
                chunk_id=item.chunk_id,
                content=item.content,
                score=0.95,
                metadata={**item.metadata, "retrieval_score": item.score},
            )
        ]


class _Provider(BaseLLMProvider):
    provider_name = "fake"
    model_name = "fake-model"

    def generate(self, **kwargs) -> ProviderResponse:
        return ProviderResponse(
            text="Quy định thử việc được dẫn tại [E1].",
            provider=self.provider_name,
            model=self.model_name,
            usage={"eval_count": 10},
        )


def test_context_exposes_typed_evidence() -> None:
    context_builder = RerankedContextBuilder()
    result = context_builder.build("thử việc", _Reranker().rerank("thử việc", _Retriever().retrieve("thử việc")))

    assert result.status == "ready"
    assert result.evidence[0].evidence_id == "E1"
    assert result.evidence[0].content == "Điều 24. Thử việc"
    assert result.evidence[0].article_number == "24"
    assert result.evidence[0].retrieval_score == 0.80
    assert result.evidence[0].reranker_score == 0.95


def test_end_to_end_pipeline_passes_objects_without_jsonl() -> None:
    retrieval_pipeline = RetrievalContextPipeline(
        retriever=_Retriever(),
        reranker=_Reranker(),
        context_builder=RerankedContextBuilder(),
        config=RetrievalContextPipelineConfig(
            retrieval_top_k=5,
            rerank_top_k=5,
        ),
    )
    pipeline = LaborLawRAGPipeline(
        retrieval_pipeline=retrieval_pipeline,
        generator=RAGGenerator(_Provider()),
    )

    result = pipeline.run("thử việc")

    assert result.status == "generated"
    assert result.answer == "Quy định thử việc được dẫn tại [E1]."
    assert result.evidence[0].evidence_id == "E1"
    assert result.generation is not None
    assert result.generation.citations_used == ["E1"]
