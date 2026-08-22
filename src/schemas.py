from typing import Literal

from pydantic import BaseModel, Field


RetrievalMode = Literal[
    "fast",
    "balanced",
    "quality",
]


class SearchRequest(BaseModel):
    query: str = Field(
        min_length=2,
        max_length=2000,
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
    )

    retrieval_mode: RetrievalMode = "quality"


class AskRequest(BaseModel):
    question: str = Field(
        min_length=2,
        max_length=4000,
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
    )

    retrieval_mode: RetrievalMode = "quality"


class Citation(BaseModel):
    chunk_id: str
    document: str
    page: int
    text: str
    score: float
    reranker_score: float | None = None


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]


class CompareRequest(BaseModel):
    question: str

    documents: list[str]

    top_k_per_document: int = Field(
        default=3,
        ge=1,
        le=10,
    )


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