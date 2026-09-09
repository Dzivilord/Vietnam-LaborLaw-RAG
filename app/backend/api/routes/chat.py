from fastapi import APIRouter, Request
from app.backend.api.schemas import ChatRequest, ChatResponse, EvidenceCitation

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("")
def chat(body: ChatRequest, request: Request):
    runtime = request.app.state.rag_runtime
    result = runtime.chat(body.query)

    return ChatResponse(
        answer=result.answer,
        citations=result.metadata.get("citations", []),
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
