from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, median
from time import perf_counter

from src.core.embeddings import embed_texts
from src.core.rag import rag_service
from src.core.reranker import get_reranker, rerank


EVAL_PATH = Path("evaluation/retrieval_eval.json")
OUTPUT_PATH = Path("evaluation/candidate_k_ablation.json")

CANDIDATE_K_VALUES = (5, 10, 20, 40)
TOP_K = 5
RUNS_PER_SETTING = 3


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


def warm_up(examples: list[dict]) -> None:
    """
    Warm up embedding generation, FAISS search, and CrossEncoder inference
    so that one-time initialization costs do not distort latency results.
    """
    query = examples[0]["query"]

    embedding = embed_texts([query])
    candidates = rag_service.store.search(
        embedding,
        max(CANDIDATE_K_VALUES),
    )

    rerank(
        query=query,
        results=candidates,
        top_k=TOP_K,
    )


def evaluate_candidate_k_once(
    examples: list[dict],
    candidate_k: int,
) -> dict:
    recall_5_scores = []
    reciprocal_ranks = []

    retrieval_latencies = []
    rerank_latencies = []
    total_latencies = []

    for example in examples:
        query = example["query"]
        relevant_pages = example["relevant_pages"]

        total_start = perf_counter()

        retrieval_start = perf_counter()

        query_embedding = embed_texts([query])

        candidates = rag_service.store.search(
            query_embedding,
            candidate_k,
        )

        retrieval_ms = (
            perf_counter() - retrieval_start
        ) * 1000

        rerank_start = perf_counter()

        reranked = rerank(
            query=query,
            results=candidates,
            top_k=TOP_K,
        )

        rerank_ms = (
            perf_counter() - rerank_start
        ) * 1000

        total_ms = (
            perf_counter() - total_start
        ) * 1000

        retrieved_pages = [
            item["page"]
            for item in reranked
        ]

        recall_5_scores.append(
            recall_at_k(
                retrieved_pages,
                relevant_pages,
                TOP_K,
            )
        )

        reciprocal_ranks.append(
            reciprocal_rank(
                retrieved_pages,
                relevant_pages,
            )
        )

        retrieval_latencies.append(
            retrieval_ms
        )

        rerank_latencies.append(
            rerank_ms
        )

        total_latencies.append(
            total_ms
        )

    return {
        "recall@5": mean(recall_5_scores),
        "mrr": mean(reciprocal_ranks),
        "avg_retrieval_latency_ms": mean(
            retrieval_latencies
        ),
        "avg_rerank_latency_ms": mean(
            rerank_latencies
        ),
        "avg_total_latency_ms": mean(
            total_latencies
        ),
    }


def evaluate_candidate_k(
    examples: list[dict],
    candidate_k: int,
) -> dict:
    runs = []

    for run_number in range(
        1,
        RUNS_PER_SETTING + 1,
    ):
        print(
            f"  Run {run_number}/"
            f"{RUNS_PER_SETTING}"
        )

        result = evaluate_candidate_k_once(
            examples,
            candidate_k,
        )

        runs.append(result)

    recall_scores = [
        run["recall@5"]
        for run in runs
    ]

    mrr_scores = [
        run["mrr"]
        for run in runs
    ]

    retrieval_latencies = [
        run["avg_retrieval_latency_ms"]
        for run in runs
    ]

    rerank_latencies = [
        run["avg_rerank_latency_ms"]
        for run in runs
    ]

    total_latencies = [
        run["avg_total_latency_ms"]
        for run in runs
    ]

    return {
        "candidate_k": candidate_k,

        "recall@5": round(
            mean(recall_scores),
            4,
        ),

        "mrr": round(
            mean(mrr_scores),
            4,
        ),

        "median_retrieval_latency_ms": round(
            median(retrieval_latencies),
            2,
        ),

        "median_rerank_latency_ms": round(
            median(rerank_latencies),
            2,
        ),

        "median_total_latency_ms": round(
            median(total_latencies),
            2,
        ),

        "runs": [
            {
                "run": index + 1,
                "avg_retrieval_latency_ms": round(
                    run["avg_retrieval_latency_ms"],
                    2,
                ),
                "avg_rerank_latency_ms": round(
                    run["avg_rerank_latency_ms"],
                    2,
                ),
                "avg_total_latency_ms": round(
                    run["avg_total_latency_ms"],
                    2,
                ),
            }
            for index, run in enumerate(runs)
        ],
    }


def run_ablation() -> dict:
    examples = json.loads(
        EVAL_PATH.read_text(
            encoding="utf-8"
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

    # Load CrossEncoder model.
    get_reranker()

    print("Warming up models...")
    warm_up(examples)
    print("Warm-up complete.\n")

    results = []

    for candidate_k in CANDIDATE_K_VALUES:
        print(
            f"Evaluating candidate_k="
            f"{candidate_k}"
        )

        result = evaluate_candidate_k(
            examples,
            candidate_k,
        )

        results.append(result)

        print(
            f"  Recall@5: "
            f"{result['recall@5']}"
        )

        print(
            f"  MRR: "
            f"{result['mrr']}"
        )

        print(
            f"  Median total latency: "
            f"{result['median_total_latency_ms']} ms\n"
        )

    report = {
        "queries": len(examples),
        "top_k": TOP_K,
        "runs_per_setting": RUNS_PER_SETTING,
        "candidate_k_values": list(
            CANDIDATE_K_VALUES
        ),
        "results": results,
    }

    OUTPUT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    return report


if __name__ == "__main__":
    report = run_ablation()

    print(
        json.dumps(
            report,
            indent=2,
        )
    )

    print(
        "\nAblation report saved to: "
        f"{OUTPUT_PATH}"
    )