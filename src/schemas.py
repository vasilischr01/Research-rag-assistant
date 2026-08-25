from typing import Literal

from pydantic import BaseModel, Field

RetrievalMode = Literal[
    "fast",
    "balanced",
    "quality",
]

RetrievalStrategy = Literal[
    "dense",
    "bm25",
    "hybrid",
    "dense_reranked",
    "hybrid_reranked",
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

    retrieval_strategy: RetrievalStrategy = "hybrid"


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

    retrieval_strategy: RetrievalStrategy = "hybrid"


class Citation(BaseModel):
    chunk_id: str
    document: str
    page: int
    text: str

    # Dense or BM25 first-stage score.
    score: float | None = None

    # Hybrid Reciprocal Rank Fusion score.
    rrf_score: float | None = None

    # Original component scores retained by hybrid retrieval.
    dense_score: float | None = None
    bm25_score: float | None = None

    # Present only for reranked strategies.
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