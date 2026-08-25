from fastapi.testclient import TestClient

from src.api import main as api_main
from src.api.main import app, rag_service

client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_upload_rejects_non_pdf():
    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "notes.txt",
                b"hello",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400


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

    response = client.post(
        "/search",
        json={
            "query": "method",
            "top_k": 3,
        },
    )

    assert response.status_code == 200
    assert (
        response.json()[0]["document"]
        == "paper.pdf"
    )


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
        captured["retrieval_strategy"] = (
            retrieval_strategy
        )

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
    assert (
        captured["retrieval_strategy"]
        == "hybrid"
    )


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
        captured["retrieval_strategy"] = (
            retrieval_strategy
        )

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
    assert (
        captured["retrieval_strategy"]
        == "hybrid"
    )


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
            (
                "Forecasting migraine "
                "with time-series.pdf"
            ),
            (
                "Machine Diagnostics and Machine "
                "Phenotyping of migraine.pdf"
            ),
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
            (
                "Forecasting migraine "
                "with time-series.pdf"
            ),
            (
                "Machine Diagnostics and Machine "
                "Phenotyping of migraine.pdf"
            ),
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


def test_security_headers_are_present():
    response = client.get("/health")

    assert response.status_code == 200

    assert (
        response.headers[
            "X-Content-Type-Options"
        ]
        == "nosniff"
    )

    assert (
        response.headers[
            "X-Frame-Options"
        ]
        == "DENY"
    )

    assert (
        response.headers[
            "Referrer-Policy"
        ]
        == "no-referrer"
    )

    assert (
        response.headers[
            "Cache-Control"
        ]
        == "no-store"
    )

    assert (
        "camera=()"
        in response.headers[
            "Permissions-Policy"
        ]
    )


def test_upload_rejects_fake_pdf(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        api_main,
        "UPLOAD_DIR",
        tmp_path,
    )

    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "fake.pdf",
                b"This is not really a PDF.",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": (
            "Uploaded file is not a valid PDF."
        ),
    }

    assert not (
        tmp_path / "fake.pdf"
    ).exists()


def test_upload_rejects_duplicate_filename(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        api_main,
        "UPLOAD_DIR",
        tmp_path,
    )

    existing_file = (
        tmp_path / "duplicate.pdf"
    )

    existing_file.write_bytes(
        b"%PDF-existing"
    )

    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "duplicate.pdf",
                b"%PDF-new-content",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 409
    assert response.json() == {
        "detail": (
            "A document with this "
            "filename already exists."
        ),
    }

    assert (
        existing_file.read_bytes()
        == b"%PDF-existing"
    )


def test_upload_rejects_oversized_pdf(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        api_main,
        "UPLOAD_DIR",
        tmp_path,
    )

    oversized_pdf = (
        b"%PDF-"
        + b"x" * api_main.MAX_PDF_BYTES
    )

    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "oversized.pdf",
                oversized_pdf,
                "application/pdf",
            )
        },
    )

    assert response.status_code == 413

    assert response.json() == {
        "detail": (
            "PDF exceeds the "
            "20 MB size limit."
        ),
    }

    assert not (
        tmp_path / "oversized.pdf"
    ).exists()


def test_rate_limit_is_enforced():
    api_main._request_history.clear()

    try:
        last_response = None

        for _ in range(
            api_main.RATE_LIMIT_REQUESTS + 1
        ):
            last_response = client.get(
                "/health"
            )

        assert last_response is not None
        assert last_response.status_code == 429

        assert last_response.json() == {
            "detail": "Too many requests",
        }

        assert (
            last_response.headers[
                "Retry-After"
            ]
            == str(
                api_main
                .RATE_LIMIT_WINDOW_SECONDS
            )
        )

    finally:
        api_main._request_history.clear()


def test_search_value_error_is_sanitized(
    monkeypatch,
):
    secret_message = (
        "private retrieval implementation "
        "detail"
    )

    def fail_search(**kwargs):
        raise ValueError(
            secret_message
        )

    monkeypatch.setattr(
        rag_service,
        "search_with_mode",
        fail_search,
    )

    response = client.post(
        "/search",
        json={
            "query": "test query",
            "top_k": 3,
        },
    )

    assert response.status_code == 400

    assert response.json() == {
        "detail": "Invalid search request.",
    }

    assert (
        secret_message
        not in response.text
    )


def test_search_internal_error_is_sanitized(
    monkeypatch,
):
    secret_message = (
        "super-secret internal "
        "retrieval traceback"
    )

    def fail_search(**kwargs):
        raise RuntimeError(
            secret_message
        )

    monkeypatch.setattr(
        rag_service,
        "search_with_mode",
        fail_search,
    )

    response = client.post(
        "/search",
        json={
            "query": "test query",
            "top_k": 3,
        },
    )

    assert response.status_code == 500

    assert response.json() == {
        "detail": "Search failed.",
    }

    assert (
        secret_message
        not in response.text
    )