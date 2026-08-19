from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

from src.core.embeddings import embed_texts
from src.core.rag import rag_service
from src.core.reranker import get_reranker, rerank


EVAL_PATH = Path("evaluation/retrieval_eval.json")
OUTPUT_PATH = Path("evaluation/retrieval_benchmark.json")

TOP_K_VALUES = (3, 5)
CANDIDATE_K = 20


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


def percentage_change(
    baseline: float,
    new_value: float,
) -> float | None:
    if baseline == 0:
        return None

    return ((new_value - baseline) / baseline) * 100


def evaluate_result_set(
    hits: list[dict],
    relevant_pages: list[int],
) -> dict:
    retrieved_pages = [
        hit["page"]
        for hit in hits
    ]

    metrics = {
        f"recall@{k}": round(
            recall_at_k(
                retrieved_pages,
                relevant_pages,
                k,
            ),
            4,
        )
        for k in TOP_K_VALUES
    }

    metrics["reciprocal_rank"] = round(
        reciprocal_rank(
            retrieved_pages,
            relevant_pages,
        ),
        4,
    )

    return metrics


def mean_metric(
    results: list[dict],
    system: str,
    metric: str,
) -> float:
    if not results:
        return 0.0

    return round(
        sum(
            result[system][metric]
            for result in results
        )
        / len(results),
        4,
    )


def evaluate_retrieval() -> dict:
    examples = json.loads(
        EVAL_PATH.read_text(
            encoding="utf-8",
        )
    )

    if not examples:
        raise ValueError(
            "Evaluation dataset is empty."
        )

    if rag_service.store.size == 0:
        raise RuntimeError(
            "Vector store is empty. "
            "Ingest documents before running evaluation."
        )

    # Warm up the CrossEncoder before timing.
    # This avoids counting model loading time as reranking latency.
    get_reranker()

    results = []

    dense_latencies = []
    rerank_latencies = []
    total_latencies = []

    for example in examples:
        query = example["query"]
        relevant_pages = example["relevant_pages"]

        # --------------------------------------------------
        # Dense retrieval
        # --------------------------------------------------

        dense_start = perf_counter()

        query_embedding = embed_texts(
            [query]
        )

        dense_candidates = rag_service.store.search(
            query_embedding,
            CANDIDATE_K,
        )

        dense_end = perf_counter()

        dense_latency_ms = (
            dense_end - dense_start
        ) * 1000

        # Dense baseline uses the original FAISS ranking.
        dense_hits = dense_candidates[: max(TOP_K_VALUES)]

        # --------------------------------------------------
        # Cross-encoder reranking
        # --------------------------------------------------

        rerank_start = perf_counter()

        reranked_hits = rerank(
            query=query,
            results=dense_candidates,
            top_k=max(TOP_K_VALUES),
        )

        rerank_end = perf_counter()

        rerank_latency_ms = (
            rerank_end - rerank_start
        ) * 1000

        total_latency_ms = (
            dense_latency_ms
            + rerank_latency_ms
        )

        dense_latencies.append(
            dense_latency_ms
        )

        rerank_latencies.append(
            rerank_latency_ms
        )

        total_latencies.append(
            total_latency_ms
        )

        dense_metrics = evaluate_result_set(
            dense_hits,
            relevant_pages,
        )

        reranked_metrics = evaluate_result_set(
            reranked_hits,
            relevant_pages,
        )

        results.append(
            {
                "query": query,
                "relevant_pages": relevant_pages,
                "dense": {
                    **dense_metrics,
                    "latency_ms": round(
                        dense_latency_ms,
                        2,
                    ),
                    "retrieved_pages": [
                        hit["page"]
                        for hit in dense_hits
                    ],
                },
                "reranked": {
                    **reranked_metrics,
                    "rerank_latency_ms": round(
                        rerank_latency_ms,
                        2,
                    ),
                    "total_latency_ms": round(
                        total_latency_ms,
                        2,
                    ),
                    "retrieved_pages": [
                        hit["page"]
                        for hit in reranked_hits
                    ],
                },
            }
        )

    dense_recall_3 = mean_metric(
        results,
        "dense",
        "recall@3",
    )

    dense_recall_5 = mean_metric(
        results,
        "dense",
        "recall@5",
    )

    dense_mrr = mean_metric(
        results,
        "dense",
        "reciprocal_rank",
    )

    reranked_recall_3 = mean_metric(
        results,
        "reranked",
        "recall@3",
    )

    reranked_recall_5 = mean_metric(
        results,
        "reranked",
        "recall@5",
    )

    reranked_mrr = mean_metric(
        results,
        "reranked",
        "reciprocal_rank",
    )

    report = {
        "queries": len(results),
        "candidate_k": CANDIDATE_K,
        "dense": {
            "mean_recall@3": dense_recall_3,
            "mean_recall@5": dense_recall_5,
            "mrr": dense_mrr,
            "avg_latency_ms": round(
                sum(dense_latencies)
                / len(dense_latencies),
                2,
            ),
        },
        "reranked": {
            "mean_recall@3": reranked_recall_3,
            "mean_recall@5": reranked_recall_5,
            "mrr": reranked_mrr,
            "avg_rerank_latency_ms": round(
                sum(rerank_latencies)
                / len(rerank_latencies),
                2,
            ),
            "avg_total_latency_ms": round(
                sum(total_latencies)
                / len(total_latencies),
                2,
            ),
        },
        "improvement": {
            "recall@3_absolute": round(
                reranked_recall_3
                - dense_recall_3,
                4,
            ),
            "recall@3_percent": (
                None
                if dense_recall_3 == 0
                else round(
                    percentage_change(
                        dense_recall_3,
                        reranked_recall_3,
                    ),
                    2,
                )
            ),
            "recall@5_absolute": round(
                reranked_recall_5
                - dense_recall_5,
                4,
            ),
            "recall@5_percent": (
                None
                if dense_recall_5 == 0
                else round(
                    percentage_change(
                        dense_recall_5,
                        reranked_recall_5,
                    ),
                    2,
                )
            ),
            "mrr_absolute": round(
                reranked_mrr
                - dense_mrr,
                4,
            ),
            "mrr_percent": (
                None
                if dense_mrr == 0
                else round(
                    percentage_change(
                        dense_mrr,
                        reranked_mrr,
                    ),
                    2,
                )
            ),
        },
        "results": results,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    return report


if __name__ == "__main__":
    report = evaluate_retrieval()

    print(
        json.dumps(
            report,
            indent=2,
        )
    )

    print(
        f"\nBenchmark saved to: {OUTPUT_PATH}"
    )