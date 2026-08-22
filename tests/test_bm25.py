from src.core.bm25 import BM25Retriever, tokenize
from src.core.chunking import Chunk


def test_tokenize_lowercases_and_splits():
    tokens = tokenize(
        "Migraine Forecasting with GRU-D Models"
    )

    assert tokens == [
        "migraine",
        "forecasting",
        "with",
        "gru-d",
        "models",
    ]


def test_bm25_returns_relevant_chunk_first():
    chunks = [
        Chunk(
            chunk_id="doc.pdf:p1:c0",
            document="doc.pdf",
            page=1,
            text=(
                "Migraine forecasting was performed "
                "with time-series neural networks."
            ),
        ),
        Chunk(
            chunk_id="doc.pdf:p2:c0",
            document="doc.pdf",
            page=2,
            text=(
                "The study reported demographic "
                "information about participants."
            ),
        ),
    ]

    retriever = BM25Retriever()
    retriever.build(chunks)

    results = retriever.search(
        query="migraine forecasting",
        top_k=2,
    )

    assert len(results) >= 1
    assert results[0]["chunk_id"] == "doc.pdf:p1:c0"
    assert results[0]["page"] == 1
    assert results[0]["score"] > 0


def test_bm25_respects_top_k():
    chunks = [
        Chunk(
            chunk_id=f"doc.pdf:p{i}:c0",
            document="doc.pdf",
            page=i,
            text="migraine forecasting model",
        )
        for i in range(1, 6)
    ]

    retriever = BM25Retriever()
    retriever.build(chunks)

    results = retriever.search(
        query="migraine",
        top_k=2,
    )

    assert len(results) == 2


def test_bm25_returns_empty_for_unknown_terms():
    chunks = [
        Chunk(
            chunk_id="doc.pdf:p1:c0",
            document="doc.pdf",
            page=1,
            text="migraine forecasting model",
        )
    ]

    retriever = BM25Retriever()
    retriever.build(chunks)

    results = retriever.search(
        query="quantum astronomy",
        top_k=5,
    )

    assert results == []


def test_bm25_returns_empty_when_index_is_empty():
    retriever = BM25Retriever()

    results = retriever.search(
        query="migraine",
        top_k=5,
    )

    assert results == []