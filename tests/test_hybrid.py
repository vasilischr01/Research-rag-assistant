from src.core.hybrid import reciprocal_rank_fusion


def test_rrf_combines_ranked_lists():
    dense_results = [
        {
            "chunk_id": "a",
            "document": "paper.pdf",
            "page": 1,
            "text": "A",
            "score": 0.9,
        },
        {
            "chunk_id": "b",
            "document": "paper.pdf",
            "page": 2,
            "text": "B",
            "score": 0.8,
        },
    ]

    bm25_results = [
        {
            "chunk_id": "b",
            "document": "paper.pdf",
            "page": 2,
            "text": "B",
            "score": 4.0,
        },
        {
            "chunk_id": "c",
            "document": "paper.pdf",
            "page": 3,
            "text": "C",
            "score": 3.0,
        },
    ]

    results = reciprocal_rank_fusion(
        ranked_lists=[
            dense_results,
            bm25_results,
        ],
        top_k=3,
    )

    assert len(results) == 3
    assert results[0]["chunk_id"] == "b"
    assert "rrf_score" in results[0]
    assert results[0]["rrf_score"] > 0


def test_rrf_removes_duplicate_chunks():
    first = [
        {
            "chunk_id": "a",
            "document": "paper.pdf",
            "page": 1,
            "text": "A",
            "score": 0.9,
        }
    ]

    second = [
        {
            "chunk_id": "a",
            "document": "paper.pdf",
            "page": 1,
            "text": "A",
            "score": 5.0,
        }
    ]

    results = reciprocal_rank_fusion(
        ranked_lists=[
            first,
            second,
        ],
        top_k=10,
    )

    assert len(results) == 1
    assert results[0]["chunk_id"] == "a"


def test_rrf_respects_top_k():
    ranked_list = [
        {
            "chunk_id": str(i),
            "document": "paper.pdf",
            "page": i,
            "text": str(i),
            "score": 1.0,
        }
        for i in range(10)
    ]

    results = reciprocal_rank_fusion(
        ranked_lists=[ranked_list],
        top_k=3,
    )

    assert len(results) == 3


def test_rrf_empty_input():
    results = reciprocal_rank_fusion(
        ranked_lists=[],
        top_k=5,
    )

    assert results == []