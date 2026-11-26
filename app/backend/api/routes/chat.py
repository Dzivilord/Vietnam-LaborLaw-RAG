from time import perf_counter
from uuid import uuid4

from fastapi import APIRouter, Request
from app.backend.api.schemas import ChatRequest, ChatResponse, ClarificationSuggestion, EvidenceCitation

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("")
def chat(body: ChatRequest, request: Request):
    request_id = str(uuid4())
    started = perf_counter()
    runtime = request.app.state.rag_runtime
    options = body.options
    result = runtime.chat(
        body.query,
        enable_rerank=options.enable_rerank if options else None,
        generation_overrides=(
            options.generation.model_dump(exclude_none=True, exclude_defaults=True)
            if options
            else {}
        ),
    )
    diagnostics = result.metadata.get("diagnostics", {})
    request.app.state.trace_recorder.record(
        request_id=request_id,
        query=body.query,
        status=result.metadata.get("status", "error"),
        total_latency_ms=(perf_counter() - started) * 1000.0,
        diagnostics=diagnostics,
        evidence_count=len(result.evidence),
        citations=result.metadata.get("citations", []),
        error=result.metadata.get("error"),
    )

    return ChatResponse(
        request_id=request_id,
        answer=result.answer,
        status=result.metadata.get("status"),
        citations=result.metadata.get("citations", []),
        suggestions=[ClarificationSuggestion(**item) for item in result.metadata.get("suggestions", [])],
        evidence=[
            EvidenceCitation(
                evidence_id=item.evidence_id,
                chunk_id=item.chunk_id,
                content=item.content,
                document_title=item.document_title,
                document_number=item.document_number,
                source_file=item.source_file,
                page=(
                    str(item.metadata.get("page_number"))
                    if item.metadata.get("page_number") is not None
                    else (
                        str(item.metadata.get("page"))
                        if item.metadata.get("page") is not None
                        else None
                    )
                ),
                chapter_number=item.chapter_number,
                chapter_title=item.chapter_title,
                section_number=item.section_number,
                section_title=item.section_title,
                article_number=item.article_number,
                article_title=item.article_title,
                clause_number=item.clause_number,
                point_number=item.point_number,
            )
            for item in result.evidence
        ],
        debug=result.metadata,
    )
