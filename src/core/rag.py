import re
from collections import Counter
from pathlib import Path

from src.core.bm25 import BM25Retriever
from src.core.chunking import chunk_pages
from src.core.config import settings
from src.core.embeddings import embed_texts
from src.core.generation import generate_answer
from src.core.hybrid import reciprocal_rank_fusion
from src.core.pdf import extract_pdf_pages
from src.core.reranker import rerank
from src.core.vector_store import VectorStore

RETRIEVAL_MODES = {
    "fast": 5,
    "balanced": 10,
    "quality": 20,
}

RETRIEVAL_STRATEGIES = {
    "dense",
    "bm25",
    "hybrid",
    "dense_reranked",
    "hybrid_reranked",
}


class RAGService:
    def __init__(self):
        self.store = VectorStore()
        self.store.load()

        self.bm25 = BM25Retriever()
        self._rebuild_bm25()

    def _rebuild_bm25(self) -> None:
        self.bm25.build(
            self.store.chunks
        )

    def ingest_pdf(
        self,
        path: Path,
    ) -> dict:
        pages = extract_pdf_pages(path)

        chunks = chunk_pages(
            pages,
            settings.chunk_size,
            settings.chunk_overlap,
        )

        self.store.add(
            chunks,
            embed_texts(
                [
                    chunk.text
                    for chunk in chunks
                ]
            ),
        )

        self._rebuild_bm25()

        return {
            "document": path.name,
            "pages": len(pages),
            "chunks_added": len(chunks),
            "index_size": self.store.size,
        }

    # --------------------------------------------------
    # First-stage retrievers
    # --------------------------------------------------

    def dense_search(
        self,
        query: str,
        candidate_k: int,
    ) -> list[dict]:
        embedding = embed_texts(
            [query]
        )

        return self.store.search(
            embedding,
            candidate_k,
        )

    def bm25_search(
        self,
        query: str,
        candidate_k: int,
    ) -> list[dict]:
        return self.bm25.search(
            query=query,
            top_k=candidate_k,
        )

    def hybrid_search(
        self,
        query: str,
        candidate_k: int,
    ) -> list[dict]:
        dense_results = self.dense_search(
            query=query,
            candidate_k=candidate_k,
        )

        bm25_results = self.bm25_search(
            query=query,
            candidate_k=candidate_k,
        )

        return reciprocal_rank_fusion(
            ranked_lists=[
                dense_results,
                bm25_results,
            ],
            top_k=candidate_k,
        )

    # --------------------------------------------------
    # Unified retrieval interface
    # --------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int,
        candidate_k: int | None = None,
        retrieval_strategy: str = "hybrid",
    ) -> list[dict]:
        if candidate_k is None:
            candidate_k = max(
                top_k * 4,
                20,
            )

        if candidate_k < top_k:
            raise ValueError(
                "candidate_k must be greater than "
                "or equal to top_k."
            )

        if retrieval_strategy not in RETRIEVAL_STRATEGIES:
            raise ValueError(
                "retrieval_strategy must be one of: "
                "'dense', 'bm25', 'hybrid', "
                "'dense_reranked', "
                "'hybrid_reranked'."
            )

        # Dense FAISS only
        if retrieval_strategy == "dense":
            candidates = self.dense_search(
                query=query,
                candidate_k=candidate_k,
            )

            return candidates[:top_k]

        # BM25 only
        if retrieval_strategy == "bm25":
            candidates = self.bm25_search(
                query=query,
                candidate_k=candidate_k,
            )

            return candidates[:top_k]

        # Dense + BM25 + RRF
        # Benchmark-selected default.
        if retrieval_strategy == "hybrid":
            candidates = self.hybrid_search(
                query=query,
                candidate_k=candidate_k,
            )

            return candidates[:top_k]

        # Dense + CrossEncoder
        if retrieval_strategy == "dense_reranked":
            candidates = self.dense_search(
                query=query,
                candidate_k=candidate_k,
            )

            return rerank(
                query=query,
                results=candidates,
                top_k=top_k,
            )

        # Dense + BM25 + RRF + CrossEncoder
        candidates = self.hybrid_search(
            query=query,
            candidate_k=candidate_k,
        )

        return rerank(
            query=query,
            results=candidates,
            top_k=top_k,
        )

    def search_without_reranking(
        self,
        query: str,
        top_k: int,
        retrieval_strategy: str = "hybrid",
    ) -> list[dict]:
        if retrieval_strategy not in {
            "dense",
            "bm25",
            "hybrid",
        }:
            raise ValueError(
                "retrieval_strategy must be one of: "
                "'dense', 'bm25', 'hybrid'."
            )

        return self.search(
            query=query,
            top_k=top_k,
            candidate_k=top_k,
            retrieval_strategy=retrieval_strategy,
        )

    # --------------------------------------------------
    # Retrieval modes
    # --------------------------------------------------

    def search_with_mode(
        self,
        query: str,
        top_k: int,
        mode: str = "quality",
        retrieval_strategy: str = "hybrid",
    ) -> list[dict]:
        if mode not in RETRIEVAL_MODES:
            raise ValueError(
                f"Unknown retrieval mode: {mode}. "
                f"Choose from "
                f"{list(RETRIEVAL_MODES)}."
            )

        candidate_k = max(
            top_k,
            RETRIEVAL_MODES[mode],
        )

        return self.search(
            query=query,
            top_k=top_k,
            candidate_k=candidate_k,
            retrieval_strategy=retrieval_strategy,
        )

    # --------------------------------------------------
    # Question answering
    # --------------------------------------------------

    def ask(
        self,
        question: str,
        top_k: int,
        mode: str = "quality",
        retrieval_strategy: str = "hybrid",
    ) -> dict:
        hits = self.search_with_mode(
            query=question,
            top_k=top_k,
            mode=mode,
            retrieval_strategy=retrieval_strategy,
        )

        return {
            "answer": generate_answer(
                question,
                hits,
            ),
            "citations": hits,
        }

    # --------------------------------------------------
    # Multi-document comparison
    # --------------------------------------------------

    def compare_documents(
        self,
        question: str,
        documents: list[str],
        top_k_per_document: int = 3,
    ) -> dict:
        candidate_k = max(
            top_k_per_document * 8,
            30,
        )

        # Hybrid retrieval improves candidate coverage.
        candidates = self.hybrid_search(
            query=question,
            candidate_k=candidate_k,
        )

        grouped = {}

        for document in documents:
            document_candidates = [
                item
                for item in candidates
                if item["document"] == document
            ]

            # Comparison still uses the CrossEncoder
            # inside each document to rank evidence.
            ranked = rerank(
                query=question,
                results=document_candidates,
                top_k=top_k_per_document,
            )

            formatted = []

            for rank, item in enumerate(
                ranked,
                start=1,
            ):
                retrieval_score = item.get(
                    "rrf_score",
                    item.get(
                        "score",
                        0.0,
                    ),
                )

                formatted.append(
                    {
                        "rank": rank,
                        "document": item["document"],
                        "page": item["page"],
                        "chunk_id": item["chunk_id"],
                        "retrieval_score": round(
                            float(retrieval_score),
                            4,
                        ),
                        "reranker_score": round(
                            float(
                                item[
                                    "reranker_score"
                                ]
                            ),
                            4,
                        ),
                        "citation": (
                            f'[{item["document"]}, '
                            f'p. {item["page"]}]'
                        ),
                        "text": item["text"],
                    }
                )

            grouped[document] = formatted

        summary = self._build_comparison_summary(
            grouped
        )

        return {
            "question": question,
            "documents": [
                {
                    "document": document,
                    "evidence": grouped.get(
                        document,
                        [],
                    ),
                }
                for document in documents
            ],
            "summary": summary,
        }

    def _build_comparison_summary(
        self,
        grouped: dict,
    ) -> dict:
        papers = []
        all_words = []

        stopwords = {
            "the",
            "and",
            "of",
            "to",
            "in",
            "a",
            "for",
            "with",
            "was",
            "were",
            "is",
            "are",
            "on",
            "that",
            "this",
            "using",
            "used",
            "from",
            "as",
            "by",
            "we",
            "our",
            "an",
            "be",
            "it",
            "at",
        }

        for document, evidence in grouped.items():
            snippets = []
            pages = []

            for item in evidence:
                text = item["text"].strip()

                first_sentence = re.split(
                    r"(?<=[.!?])\s+",
                    text,
                )[0]

                snippets.append(
                    first_sentence
                )

                pages.append(
                    item["page"]
                )

                words = re.findall(
                    r"\b[a-zA-Z]"
                    r"[a-zA-Z-]{3,}\b",
                    text.lower(),
                )

                all_words.extend(
                    word
                    for word in words
                    if word not in stopwords
                )

            papers.append(
                {
                    "document": document,
                    "top_evidence": snippets[:3],
                    "pages": sorted(
                        set(pages)
                    ),
                }
            )

        frequencies = Counter(
            all_words
        )

        shared_terms = [
            word
            for word, _
            in frequencies.most_common(10)
        ]

        return {
            "papers": papers,
            "shared_terms": shared_terms,
            "note": (
                "Extractive comparison summary generated "
                "from the highest-ranked evidence chunks; "
                "no external LLM used."
            ),
        }


rag_service = RAGService()