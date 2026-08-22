from fastapi.testclient import TestClient

from src.api.main import app, rag_service


client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_upload_rejects_non_pdf():
    r = client.post(
        "/documents/upload",
        files={
            "file": (
                "notes.txt",
                b"hello",
                "text/plain",
            )
        },
    )

    assert r.status_code == 400


def test_search_mock(monkeypatch):
    monkeypatch.setattr(
        rag_service,
        "search",
        lambda query, top_k, candidate_k=None, retrieval_strategy="hybrid": [
            {
                "chunk_id": "x",
                "document": "paper.pdf",
                "page": 1,
                "text": "evidence",
                "score": 0.9,
                "reranker_score": 1.2,
            }
        ],
    )

    r = client.post(
        "/search",
        json={
            "query": "method",
            "top_k": 3,
        },
    )

    assert r.status_code == 200
    assert r.json()[0]["document"] == "paper.pdf"


def test_search_fast_mode_uses_candidate_k_5(
    monkeypatch,
):
    captured = {}

    def fake_search(
        query,
        top_k,
        candidate_k=None,
        retrieval_strategy="hybrid",
    ):
        captured["query"] = query
        captured["top_k"] = top_k
        captured["candidate_k"] = candidate_k
        captured["retrieval_strategy"] = retrieval_strategy

        return [
            {
                "chunk_id": "x",
                "document": "paper.pdf",
                "page": 1,
                "text": "evidence",
                "score": 0.9,
                "reranker_score": 1.0,
            }
        ]

    monkeypatch.setattr(
        rag_service,
        "search",
        fake_search,
    )

    response = client.post(
        "/search",
        json={
            "query": "method",
            "top_k": 5,
            "retrieval_mode": "fast",
        },
    )

    assert response.status_code == 200
    assert captured["candidate_k"] == 5
    assert captured["retrieval_strategy"] == "hybrid"


def test_search_quality_mode_uses_candidate_k_20(
    monkeypatch,
):
    captured = {}

    def fake_search(
        query,
        top_k,
        candidate_k=None,
        retrieval_strategy="hybrid",
    ):
        captured["query"] = query
        captured["top_k"] = top_k
        captured["candidate_k"] = candidate_k
        captured["retrieval_strategy"] = retrieval_strategy

        return [
            {
                "chunk_id": "x",
                "document": "paper.pdf",
                "page": 1,
                "text": "evidence",
                "score": 0.9,
                "reranker_score": 1.0,
            }
        ]

    monkeypatch.setattr(
        rag_service,
        "search",
        fake_search,
    )

    response = client.post(
        "/search",
        json={
            "query": "method",
            "top_k": 5,
            "retrieval_mode": "quality",
        },
    )

    assert response.status_code == 200
    assert captured["candidate_k"] == 20
    assert captured["retrieval_strategy"] == "hybrid"


def test_invalid_retrieval_mode_is_rejected():
    response = client.post(
        "/search",
        json={
            "query": "method",
            "top_k": 5,
            "retrieval_mode": "turbo",
        },
    )

    assert response.status_code == 422


def test_compare_documents_returns_summary():
    payload = {
        "question": (
            "Compare the data sources and methods "
            "used to study migraine."
        ),
        "documents": [
            "Forecasting migraine with time-series.pdf",
            "Machine Diagnostics and Machine Phenotyping of migraine.pdf",
        ],
        "top_k_per_document": 3,
    }

    response = client.post(
        "/compare",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert "question" in data
    assert "documents" in data
    assert "summary" in data

    summary = data["summary"]

    assert "papers" in summary
    assert "shared_terms" in summary
    assert "note" in summary

    assert isinstance(
        summary["papers"],
        list,
    )

    assert isinstance(
        summary["shared_terms"],
        list,
    )

    assert isinstance(
        summary["note"],
        str,
    )


def test_compare_documents_evidence_ranking():
    payload = {
        "question": (
            "Compare migraine forecasting methods."
        ),
        "documents": [
            "Forecasting migraine with time-series.pdf",
            "Machine Diagnostics and Machine Phenotyping of migraine.pdf",
        ],
        "top_k_per_document": 3,
    }

    response = client.post(
        "/compare",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    for document_result in data["documents"]:
        evidence = document_result["evidence"]

        for item in evidence:
            assert "rank" in item
            assert "retrieval_score" in item
            assert "reranker_score" in item
            assert "citation" in item
            assert "text" in item