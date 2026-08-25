from __future__ import annotations

import math
import re
from collections import Counter

from src.core.chunking import Chunk

TOKEN_PATTERN = re.compile(
    r"\b[a-zA-Z0-9][a-zA-Z0-9_-]*\b"
)


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(
        text.lower()
    )


class BM25Retriever:
    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.k1 = k1
        self.b = b

        self.chunks: list[Chunk] = []
        self.tokenized_docs: list[list[str]] = []
        self.doc_frequencies: Counter[str] = Counter()

        self.avg_doc_length = 0.0

    def build(
        self,
        chunks: list[Chunk],
    ) -> None:
        self.chunks = list(chunks)

        self.tokenized_docs = [
            tokenize(chunk.text)
            for chunk in self.chunks
        ]

        if not self.tokenized_docs:
            self.avg_doc_length = 0.0
            self.doc_frequencies.clear()
            return

        self.avg_doc_length = (
            sum(
                len(tokens)
                for tokens in self.tokenized_docs
            )
            / len(self.tokenized_docs)
        )

        self.doc_frequencies.clear()

        for tokens in self.tokenized_docs:
            for token in set(tokens):
                self.doc_frequencies[token] += 1

    def _idf(
        self,
        term: str,
    ) -> float:
        total_docs = len(self.chunks)
        doc_freq = self.doc_frequencies.get(
            term,
            0,
        )

        return math.log(
            1
            + (
                total_docs
                - doc_freq
                + 0.5
            )
            / (
                doc_freq
                + 0.5
            )
        )

    def search(
        self,
        query: str,
        top_k: int,
    ) -> list[dict]:
        if not self.chunks:
            return []

        query_terms = tokenize(query)

        if not query_terms:
            return []

        scored = []

        for chunk, tokens in zip(
            self.chunks,
            self.tokenized_docs,
        ):
            frequencies = Counter(tokens)

            doc_length = len(tokens)

            score = 0.0

            for term in query_terms:
                term_freq = frequencies.get(
                    term,
                    0,
                )

                if term_freq == 0:
                    continue

                idf = self._idf(term)

                denominator = (
                    term_freq
                    + self.k1
                    * (
                        1
                        - self.b
                        + self.b
                        * (
                            doc_length
                            / self.avg_doc_length
                        )
                    )
                )

                score += (
                    idf
                    * (
                        term_freq
                        * (
                            self.k1
                            + 1
                        )
                    )
                    / denominator
                )

            if score > 0:
                scored.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "document": chunk.document,
                        "page": chunk.page,
                        "text": chunk.text,
                        "score": float(score),
                    }
                )

        scored.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return scored[:top_k]