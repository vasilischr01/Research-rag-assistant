from __future__ import annotations

from functools import lru_cache

from sentence_transformers import CrossEncoder

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder(RERANKER_MODEL)


def rerank(
    query: str,
    results: list[dict],
    top_k: int,
) -> list[dict]:
    if not results:
        return []

    model = get_reranker()

    pairs = [
        [query, item["text"]]
        for item in results
    ]

    scores = model.predict(pairs)

    reranked = []

    for item, score in zip(results, scores):
        result = dict(item)
        result["reranker_score"] = float(score)
        reranked.append(result)

    reranked.sort(
        key=lambda item: item["reranker_score"],
        reverse=True,
    )

    return reranked[:top_k]