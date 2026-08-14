from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=20)


class Citation(BaseModel):
    chunk_id: str
    document: str
    page: int
    text: str
    score: float


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]


class CompareRequest(BaseModel):
    question: str
    documents: list[str]
    top_k_per_document: int = Field(default=3, ge=1, le=10)


class CompareDocumentResult(BaseModel):
    document: str
    evidence: list[dict]


class PaperSummary(BaseModel):
    document: str
    top_evidence: list[str]
    pages: list[int]


class ComparisonSummary(BaseModel):
    papers: list[PaperSummary]
    shared_terms: list[str]
    note: str


class CompareResponse(BaseModel):
    question: str
    documents: list[CompareDocumentResult]
    summary: ComparisonSummary