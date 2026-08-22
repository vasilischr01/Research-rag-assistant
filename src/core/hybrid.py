from __future__ import annotations


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    top_k: int,
    rrf_k: int = 60,
) -> list[dict]:
    fused = {}

    for source_index, ranked_list in enumerate(
        ranked_lists
    ):
        for rank, item in enumerate(
            ranked_list,
            start=1,
        ):
            chunk_id = item["chunk_id"]

            if chunk_id not in fused:
                fused[chunk_id] = {
                    "chunk_id": item["chunk_id"],
                    "document": item["document"],
                    "page": item["page"],
                    "text": item["text"],
                    "rrf_score": 0.0,
                    "dense_score": None,
                    "bm25_score": None,
                }

            fused[chunk_id][
                "rrf_score"
            ] += 1.0 / (
                rrf_k + rank
            )

            if source_index == 0:
                fused[chunk_id][
                    "dense_score"
                ] = float(
                    item.get(
                        "score",
                        0.0,
                    )
                )

            elif source_index == 1:
                fused[chunk_id][
                    "bm25_score"
                ] = float(
                    item.get(
                        "score",
                        0.0,
                    )
                )

    results = list(
        fused.values()
    )

    results.sort(
        key=lambda item: item[
            "rrf_score"
        ],
        reverse=True,
    )

    return results[:top_k]