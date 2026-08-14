from __future__ import annotations

import json
from pathlib import Path

from src.core.rag import rag_service


EVAL_PATH = Path("evaluation/retrieval_eval.json")


def recall_at_k(
    retrieved_pages: list[int],
    relevant_pages: list[int],
    k: int,
) -> float:
    retrieved = set(retrieved_pages[:k])
    relevant = set(relevant_pages)

    if not relevant:
        return 0.0

    return len(retrieved & relevant) / len(relevant)


def reciprocal_rank(
    retrieved_pages: list[int],
    relevant_pages: list[int],
) -> float:
    relevant = set(relevant_pages)

    for rank, page in enumerate(retrieved_pages, start=1):
        if page in relevant:
            return 1.0 / rank

    return 0.0


def evaluate_retrieval() -> dict:
    examples = json.loads(
        EVAL_PATH.read_text(encoding="utf-8")
    )

    results = []

    for example in examples:
        query = example["query"]
        relevant_pages = example["relevant_pages"]

        hits = rag_service.search(
            query=query,
            top_k=10,
        )

        retrieved_pages = [
            hit["page"]
            for hit in hits
        ]

        result = {
            "query": query,
            "recall@3": round(
                recall_at_k(
                    retrieved_pages,
                    relevant_pages,
                    3,
                ),
                4,
            ),
            "recall@5": round(
                recall_at_k(
                    retrieved_pages,
                    relevant_pages,
                    5,
                ),
                4,
            ),
            "reciprocal_rank": round(
                reciprocal_rank(
                    retrieved_pages,
                    relevant_pages,
                ),
                4,
            ),
        }

        results.append(result)

    mean_recall_3 = sum(
        item["recall@3"]
        for item in results
    ) / len(results)

    mean_recall_5 = sum(
        item["recall@5"]
        for item in results
    ) / len(results)

    mrr = sum(
        item["reciprocal_rank"]
        for item in results
    ) / len(results)

    return {
        "queries": len(results),
        "mean_recall@3": round(mean_recall_3, 4),
        "mean_recall@5": round(mean_recall_5, 4),
        "mrr": round(mrr, 4),
        "results": results,
    }


if __name__ == "__main__":
    report = evaluate_retrieval()

    print(
        json.dumps(
            report,
            indent=2,
        )
    )