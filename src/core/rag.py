from pathlib import Path

import re
from collections import Counter
from src.core.chunking import chunk_pages
from src.core.config import settings
from src.core.embeddings import embed_texts
from src.core.generation import generate_answer
from src.core.pdf import extract_pdf_pages
from src.core.reranker import rerank
from src.core.vector_store import VectorStore

class RAGService:
    def __init__(self):
        self.store = VectorStore()
        self.store.load()

    def ingest_pdf(self, path: Path):
        pages = extract_pdf_pages(path)
        chunks = chunk_pages(pages, settings.chunk_size, settings.chunk_overlap)
        self.store.add(chunks, embed_texts([c.text for c in chunks]))
        return {"document": path.name, "pages": len(pages), "chunks_added": len(chunks), "index_size": self.store.size}

    def search(self, query, top_k):
        candidate_k = max(top_k * 4, 20)

        embedding = embed_texts([query])

        candidates = self.store.search(
            embedding,
            candidate_k,
        )

        return rerank(
            query=query,
            results=candidates,
            top_k=top_k,
        )

    def ask(self, question, top_k):
        hits = self.search(question, top_k)
        return {"answer": generate_answer(question, hits), "citations": hits}

    def compare_documents(
        self,
        question: str,
        documents: list[str],
        top_k_per_document: int = 3,
    ) -> dict:
        candidate_k = max(top_k_per_document * 8, 30)

        embedding = embed_texts([question])

        candidates = self.store.search(
            embedding,
            candidate_k,
        )

        grouped = {}

        for document in documents:
            document_candidates = [
                item
                for item in candidates
                if item["document"] == document
            ]

            ranked = rerank(
                query=question,
                results=document_candidates,
                top_k=top_k_per_document,
            )
            formatted = []

            for rank, item in enumerate(ranked, start=1):
                formatted.append(
                    {
                        "rank": rank,
                        "document": item["document"],
                        "page": item["page"],
                        "chunk_id": item["chunk_id"],
                        "retrieval_score": round(float(item["score"]), 4),
                        "reranker_score": round(
                            float(item["reranker_score"]),
                            4,
                    ),
                    "citation": (
                        f'[{item["document"]}, p. {item["page"]}]'
                    ),
                    "text": item["text"],
                }
            )

            grouped[document] = formatted

            summary = self._build_comparison_summary(grouped)

            return {
                "question": question,
                "documents": [
                    {
                        "document": document,
                        "evidence": grouped.get(document, []),
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

                snippets.append(first_sentence)
                pages.append(item["page"])

                words = re.findall(
                    r"\b[a-zA-Z][a-zA-Z-]{3,}\b",
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
                    "pages": sorted(set(pages)),
                }
            )

        frequencies = Counter(all_words)

        shared_terms = [
            word
            for word, _ in frequencies.most_common(10)
        ]

        return {
            "papers": papers,
            "shared_terms": shared_terms,
            "note": (
                "Extractive comparison summary generated from "
                "the highest-ranked evidence chunks; no external LLM used."
            ),
        }
rag_service = RAGService()

