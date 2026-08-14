import shutil
from pathlib import Path
from fastapi import FastAPI, File, HTTPException, UploadFile
from src.core.rag import rag_service
from src.schemas import (AskRequest, AskResponse, Citation, SearchRequest, CompareRequest,
    CompareResponse)

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Research RAG Assistant", version="0.1.0")

@app.get("/health")
def health():
    return {"status": "ok", "indexed_chunks": rag_service.store.size}

@app.post("/documents/upload")
def upload_document(file: UploadFile = File(...)):
    filename = Path(file.filename or "document.pdf").name
    if Path(filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    destination = UPLOAD_DIR / filename
    with destination.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    try:
        return rag_service.ingest_pdf(destination)
    except Exception as exc:
        if destination.exists():
            destination.unlink()
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@app.post("/search", response_model=list[Citation])
def search(request: SearchRequest):
    return rag_service.search(request.query, request.top_k)

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    return rag_service.ask(request.question, request.top_k)

@app.post("/compare", response_model=CompareResponse)
def compare(request: CompareRequest):
    return rag_service.compare_documents(
        question=request.question,
        documents=request.documents,
        top_k_per_document=request.top_k_per_document,
    )
