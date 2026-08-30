from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Annotated

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from src.core.rag import rag_service
from src.schemas import (
    AskRequest,
    AskResponse,
    Citation,
    CompareRequest,
    CompareResponse,
    SearchRequest,
)

logger = logging.getLogger(__name__)

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MAX_PDF_BYTES = 20 * 1024 * 1024
MAX_JSON_REQUEST_BYTES = 64 * 1024
MAX_UPLOAD_REQUEST_BYTES = 21 * 1024 * 1024

RATE_LIMIT_REQUESTS = 60
RATE_LIMIT_WINDOW_SECONDS = 60

_request_history: dict[
    str,
    deque[float],
] = defaultdict(deque)


def _add_security_headers(
    response,
):
    response.headers[
        "X-Content-Type-Options"
    ] = "nosniff"

    response.headers[
        "X-Frame-Options"
    ] = "DENY"

    response.headers[
        "Referrer-Policy"
    ] = "no-referrer"

    response.headers[
        "Permissions-Policy"
    ] = (
        "camera=(), "
        "microphone=(), "
        "geolocation=()"
    )

    response.headers[
        "Cache-Control"
    ] = "no-store"

    return response


class SecurityMiddleware(
    BaseHTTPMiddleware
):
    async def dispatch(
        self,
        request: Request,
        call_next,
    ):
        content_length = (
            request.headers.get(
                "content-length"
            )
        )

        if content_length is not None:
            try:
                request_size = int(
                    content_length
                )

            except ValueError:
                response = JSONResponse(
                    status_code=400,
                    content={
                        "detail": (
                            "Invalid Content-Length "
                            "header"
                        ),
                    },
                )

                return _add_security_headers(
                    response
                )

            request_limit = (
                MAX_UPLOAD_REQUEST_BYTES
                if request.url.path
                == "/documents/upload"
                else MAX_JSON_REQUEST_BYTES
            )

            if request_size > request_limit:
                response = JSONResponse(
                    status_code=413,
                    content={
                        "detail": (
                            "Request body too large"
                        ),
                    },
                )

                return _add_security_headers(
                    response
                )

        client_host = (
            request.client.host
            if request.client
            else "unknown"
        )

        now = time.monotonic()

        history = _request_history[
            client_host
        ]

        while (
            history
            and now - history[0]
            >= RATE_LIMIT_WINDOW_SECONDS
        ):
            history.popleft()

        if (
            len(history)
            >= RATE_LIMIT_REQUESTS
        ):
            response = JSONResponse(
                status_code=429,
                content={
                    "detail": (
                        "Too many requests"
                    ),
                },
                headers={
                    "Retry-After": str(
                        RATE_LIMIT_WINDOW_SECONDS
                    ),
                },
            )

            return _add_security_headers(
                response
            )

        history.append(
            now
        )

        response = await call_next(
            request
        )

        return _add_security_headers(
            response
        )


app = FastAPI(
    title="Research RAG Assistant",
    version="0.4.0",
)

@app.get("/")
def root():
    return {
        "name": "Research RAG Assistant",
        "status": "online",
        "version": "0.4.0",
        "documentation": "/docs",
        "health": "/health",
    }

app.add_middleware(
    SecurityMiddleware
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "indexed_chunks": (
            rag_service.store.size
        ),
    }


@app.post("/documents/upload")
def upload_document(
    file: Annotated[
        UploadFile,
        File(),
    ],
):
    filename = Path(
        file.filename
        or "document.pdf"
    ).name

    suffix = Path(
        filename
    ).suffix.lower()

    if suffix != ".pdf":
        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail=(
                "Only PDF files are supported."
            ),
        )

    allowed_content_types = {
        "application/pdf",
        "application/octet-stream",
    }

    if (
        file.content_type
        and file.content_type
        not in allowed_content_types
    ):
        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail=(
                "Invalid PDF content type."
            ),
        )

    destination = (
        UPLOAD_DIR / filename
    )

    if destination.exists():
        raise HTTPException(
            status_code=(
                status.HTTP_409_CONFLICT
            ),
            detail=(
                "A document with this "
                "filename already exists."
            ),
        )

    try:
        header = file.file.read(5)

        if header != b"%PDF-":
            raise HTTPException(
                status_code=(
                    status.HTTP_400_BAD_REQUEST
                ),
                detail=(
                    "Uploaded file is not "
                    "a valid PDF."
                ),
            )

        file.file.seek(0)

        total_bytes = 0
        chunk_size = 1024 * 1024

        with destination.open(
            "xb"
        ) as output:
            while True:
                chunk = file.file.read(
                    chunk_size
                )

                if not chunk:
                    break

                total_bytes += len(
                    chunk
                )

                if (
                    total_bytes
                    > MAX_PDF_BYTES
                ):
                    raise HTTPException(
                        status_code=(
                            status
                            .HTTP_413_REQUEST_ENTITY_TOO_LARGE
                        ),
                        detail=(
                            "PDF exceeds the "
                            "20 MB size limit."
                        ),
                    )

                output.write(
                    chunk
                )

        try:
            return rag_service.ingest_pdf(
                destination
            )

        except ValueError as exc:
            logger.warning(
                "pdf_ingestion_rejected",
                exc_info=exc,
            )

            raise HTTPException(
                status_code=(
                    status.HTTP_400_BAD_REQUEST
                ),
                detail=(
                    "The PDF could not "
                    "be processed."
                ),
            ) from exc

        except Exception as exc:
            logger.exception(
                "pdf_ingestion_failed"
            )

            raise HTTPException(
                status_code=(
                    status.HTTP_500_INTERNAL_SERVER_ERROR
                ),
                detail=(
                    "Document processing failed."
                ),
            ) from exc

    except HTTPException:
        if destination.exists():
            destination.unlink()

        raise

    except Exception as exc:
        if destination.exists():
            destination.unlink()

        logger.exception(
            "document_upload_failed"
        )

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "Document upload failed."
            ),
        ) from exc


@app.post(
    "/search",
    response_model=list[Citation],
)
def search(
    request: SearchRequest,
):
    try:
        return rag_service.search_with_mode(
            query=request.query,
            top_k=request.top_k,
            mode=request.retrieval_mode,
            retrieval_strategy=(
                request.retrieval_strategy
            ),
        )

    except ValueError as exc:
        logger.warning(
            "search_request_rejected",
            exc_info=exc,
        )

        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail=(
                "Invalid search request."
            ),
        ) from exc

    except Exception as exc:
        logger.exception(
            "search_failed"
        )

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail="Search failed.",
        ) from exc


@app.post(
    "/ask",
    response_model=AskResponse,
)
def ask(
    request: AskRequest,
):
    try:
        return rag_service.ask(
            question=request.question,
            top_k=request.top_k,
            mode=request.retrieval_mode,
            retrieval_strategy=(
                request.retrieval_strategy
            ),
        )

    except ValueError as exc:
        logger.warning(
            "ask_request_rejected",
            exc_info=exc,
        )

        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail=(
                "Invalid question request."
            ),
        ) from exc

    except Exception as exc:
        logger.exception(
            "ask_failed"
        )

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "Question processing failed."
            ),
        ) from exc


@app.post(
    "/compare",
    response_model=CompareResponse,
)
def compare(
    request: CompareRequest,
):
    try:
        return (
            rag_service.compare_documents(
                question=request.question,
                documents=(
                    request.documents
                ),
                top_k_per_document=(
                    request
                    .top_k_per_document
                ),
            )
        )

    except ValueError as exc:
        logger.warning(
            "compare_request_rejected",
            exc_info=exc,
        )

        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail=(
                "Invalid document "
                "comparison request."
            ),
        ) from exc

    except Exception as exc:
        logger.exception(
            "compare_failed"
        )

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "Document comparison failed."
            ),
        ) from exc