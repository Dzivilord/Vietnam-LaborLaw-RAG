from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str


class EvidenceCitation(BaseModel):
    evidence_id: str
    chunk_id: str
    content: str
    document_title: str | None = None
    document_number: str | None = None
    source_file: str | None = None
    page: str | None = None
    chapter_number: str | None = None
    chapter_title: str | None = None
    section_number: str | None = None
    section_title: str | None = None
    article_number: str | None = None
    article_title: str | None = None
    clause_number: str | None = None
    point_number: str | None = None


class ChatResponse(BaseModel):
    answer: str | None = None
    citations: list[str] = Field(default_factory=list)
    evidence: list[EvidenceCitation] = Field(default_factory=list)
    debug: dict = Field(default_factory=dict)
