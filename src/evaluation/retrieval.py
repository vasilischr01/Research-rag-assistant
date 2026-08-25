from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

from src.core.rag import rag_service
from src.core.reranker import get_reranker, rerank

EVAL_PATH = Path("evaluation/retrieval_eval.json")
OUTPUT_PATH = Path("evaluation/retrieval_benchmark.json")

TOP_K_VALUES = (3, 5)
FINAL_TOP_K = max(TOP_K_VALUES)
CANDIDATE_K = 20


SYSTEMS = (
    "bm25",
    "dense",
    "hybrid",
    "dense_reranked",
    "hybrid_reranked",
)


def relevant_keys(example: dict) -> set:
    """
    Supports two evaluation formats.

    Preferred:
        "relevant_locations": [
            {"document": "paper.pdf", "page": 4}
        ]

    Backwards-compatible:
        "relevant_pages": [4, 5]

    The preferred document+page representation avoids false positives
    when multiple indexed PDFs contain the same page number.
    """
    if "relevant_locations" in example:
        return {
            (
                item["document"],
                item["page"],
            )
            for item in example[
                "relevant_locations"
            ]
        }

    return set(
        example.get(
            "relevant_pages",
            [],
        )
    )


def hit_key(
    hit: dict,
    use_document: bool,
):
    if use_document:
        return (
            hit["document"],
            hit["page"],
        )

    return hit["page"]


def recall_at_k(
    hits: list[dict],
    relevant: set,
    k: int,
    use_document: bool,
) -> float:
    if not relevant:
        return 0.0

    retrieved = {
        hit_key(
            hit,
            use_document,
        )
        for hit in hits[:k]
    }

    return len(
        retrieved & relevant
    ) / len(relevant)


def reciprocal_rank(
    hits: list[dict],
    relevant: set,
    use_document: bool,
) -> float:
    for rank, hit in enumerate(
        hits,
        start=1,
    ):
        if (
            hit_key(
                hit,
                use_document,
            )
            in relevant
        ):
            return 1.0 / rank

    return 0.0


def evaluate_result_set(
    hits: list[dict],
    example: dict,
) -> dict:
    relevant = relevant_keys(
        example
    )

    use_document = (
        "relevant_locations"
        in example
    )

    metrics = {
        f"recall@{k}": round(
            recall_at_k(
                hits=hits,
                relevant=relevant,
                k=k,
                use_document=use_document,
            ),
            4,
        )
        for k in TOP_K_VALUES
    }

    metrics[
        "reciprocal_rank"
    ] = round(
        reciprocal_rank(
            hits=hits,
            relevant=relevant,
            use_document=use_document,
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
            item[system][metric]
            for item in results
        )
        / len(results),
        4,
    )


def mean_latency(
    results: list[dict],
    system: str,
) -> float:
    if not results:
        return 0.0

    return round(
        sum(
            item[system][
                "latency_ms"
            ]
            for item in results
        )
        / len(results),
        2,
    )


def format_hits(
    hits: list[dict],
) -> list[dict]:
    return [
        {
            "document": hit[
                "document"
            ],
            "page": hit["page"],
            "chunk_id": hit[
                "chunk_id"
            ],
        }
        for hit in hits
    ]


def warm_up(
    examples: list[dict],
) -> None:
    """
    Warm up embedding and CrossEncoder inference so model initialization
    is not counted as benchmark latency.
    """
    query = examples[0]["query"]

    dense_candidates = (
        rag_service.dense_search(
            query=query,
            candidate_k=CANDIDATE_K,
        )
    )

    if dense_candidates:
        rerank(
            query=query,
            results=dense_candidates,
            top_k=FINAL_TOP_K,
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
            "Ingest documents before "
            "running evaluation."
        )

    # Ensure the reranker model is loaded.
    get_reranker()

    print(
        "Warming up embedding and "
        "reranking models..."
    )

    warm_up(examples)

    print(
        "Warm-up complete.\n"
    )

    results = []

    for index, example in enumerate(
        examples,
        start=1,
    ):
        query = example["query"]

        print(
            f"Evaluating query "
            f"{index}/{len(examples)}"
        )

        # --------------------------------------------------
        # 1. BM25
        # --------------------------------------------------

        start = perf_counter()

        bm25_candidates = (
            rag_service.bm25_search(
                query=query,
                candidate_k=CANDIDATE_K,
            )
        )

        bm25_latency = (
            perf_counter() - start
        ) * 1000

        bm25_hits = (
            bm25_candidates[
                :FINAL_TOP_K
            ]
        )

        # --------------------------------------------------
        # 2. Dense FAISS
        # --------------------------------------------------

        start = perf_counter()

        dense_candidates = (
            rag_service.dense_search(
                query=query,
                candidate_k=CANDIDATE_K,
            )
        )

        dense_latency = (
            perf_counter() - start
        ) * 1000

        dense_hits = (
            dense_candidates[
                :FINAL_TOP_K
            ]
        )

        # --------------------------------------------------
        # 3. Hybrid RRF
        # --------------------------------------------------

        start = perf_counter()

        hybrid_candidates = (
            rag_service.hybrid_search(
                query=query,
                candidate_k=CANDIDATE_K,
            )
        )

        hybrid_latency = (
            perf_counter() - start
        ) * 1000

        hybrid_hits = (
            hybrid_candidates[
                :FINAL_TOP_K
            ]
        )

        # --------------------------------------------------
        # 4. Dense + CrossEncoder
        # --------------------------------------------------

        start = perf_counter()

        dense_reranked_hits = rerank(
            query=query,
            results=dense_candidates,
            top_k=FINAL_TOP_K,
        )

        dense_rerank_latency = (
            perf_counter() - start
        ) * 1000

        dense_reranked_total = (
            dense_latency
            + dense_rerank_latency
        )

        # --------------------------------------------------
        # 5. Hybrid + CrossEncoder
        # --------------------------------------------------

        start = perf_counter()

        hybrid_reranked_hits = (
            rerank(
                query=query,
                results=hybrid_candidates,
                top_k=FINAL_TOP_K,
            )
        )

        hybrid_rerank_latency = (
            perf_counter() - start
        ) * 1000

        hybrid_reranked_total = (
            hybrid_latency
            + hybrid_rerank_latency
        )

        systems = {
            "bm25": (
                bm25_hits,
                bm25_latency,
            ),
            "dense": (
                dense_hits,
                dense_latency,
            ),
            "hybrid": (
                hybrid_hits,
                hybrid_latency,
            ),
            "dense_reranked": (
                dense_reranked_hits,
                dense_reranked_total,
            ),
            "hybrid_reranked": (
                hybrid_reranked_hits,
                hybrid_reranked_total,
            ),
        }

        query_result = {
            "query": query,
        }

        if (
            "relevant_locations"
            in example
        ):
            query_result[
                "relevant_locations"
            ] = example[
                "relevant_locations"
            ]
        else:
            query_result[
                "relevant_pages"
            ] = example.get(
                "relevant_pages",
                [],
            )

        for (
            system_name,
            (
                hits,
                latency,
            ),
        ) in systems.items():
            metrics = (
                evaluate_result_set(
                    hits=hits,
                    example=example,
                )
            )

            query_result[
                system_name
            ] = {
                **metrics,
                "latency_ms": round(
                    latency,
                    2,
                ),
                "retrieved": (
                    format_hits(hits)
                ),
            }

        query_result[
            "dense_reranked"
        ][
            "rerank_only_latency_ms"
        ] = round(
            dense_rerank_latency,
            2,
        )

        query_result[
            "hybrid_reranked"
        ][
            "rerank_only_latency_ms"
        ] = round(
            hybrid_rerank_latency,
            2,
        )

        results.append(
            query_result
        )

    # --------------------------------------------------
    # Aggregate results
    # --------------------------------------------------

    summary = {}

    for system in SYSTEMS:
        summary[system] = {
            "mean_recall@3": (
                mean_metric(
                    results,
                    system,
                    "recall@3",
                )
            ),
            "mean_recall@5": (
                mean_metric(
                    results,
                    system,
                    "recall@5",
                )
            ),
            "mrr": (
                mean_metric(
                    results,
                    system,
                    "reciprocal_rank",
                )
            ),
            "avg_latency_ms": (
                mean_latency(
                    results,
                    system,
                )
            ),
        }

    report = {
        "queries": len(results),
        "candidate_k": CANDIDATE_K,
        "top_k_values": list(
            TOP_K_VALUES
        ),
        "evaluation_unit": (
            "document+page"
            if all(
                "relevant_locations"
                in example
                for example in examples
            )
            else "page"
        ),
        "systems": summary,
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
        "\n===== BENCHMARK SUMMARY =====\n"
    )

    print(
        json.dumps(
            {
                "queries": report[
                    "queries"
                ],
                "candidate_k": report[
                    "candidate_k"
                ],
                "evaluation_unit": report[
                    "evaluation_unit"
                ],
                "systems": report[
                    "systems"
                ],
            },
            indent=2,
        )
    )

    print(
        f"\nBenchmark saved to: "
        f"{OUTPUT_PATH}"
    )