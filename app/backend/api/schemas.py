from pydantic import BaseModel, Field


class GenerationOverrides(BaseModel):
    model: str | None = Field(default=None, min_length=1, max_length=200)
    fallback_models: list[str] = Field(default_factory=list, max_length=5)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_output_tokens: int | None = Field(default=None, ge=64, le=4096)


class ChatRuntimeOptions(BaseModel):
    enable_rerank: bool | None = None
    generation: GenerationOverrides = Field(default_factory=GenerationOverrides)


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    options: ChatRuntimeOptions | None = None


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


class ClarificationSuggestion(BaseModel):
    label: str
    article_number: str | None = None
    clause_number: str | None = None


class ChatResponse(BaseModel):
    request_id: str
    answer: str | None = None
    status: str | None = None
    citations: list[str] = Field(default_factory=list)
    evidence: list[EvidenceCitation] = Field(default_factory=list)
    suggestions: list[ClarificationSuggestion] = Field(default_factory=list)
    debug: dict = Field(default_factory=dict)
