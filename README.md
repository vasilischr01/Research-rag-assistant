# Research RAG Assistant

A local Retrieval-Augmented Generation (RAG) backend for scientific PDF analysis, semantic search, evidence reranking, citation-aware retrieval, and multi-document comparison.

The project provides an end-to-end pipeline for uploading research papers, extracting and chunking their content, generating semantic embeddings, retrieving relevant evidence, reranking retrieved passages with a cross-encoder, and exposing the workflow through a FastAPI REST API.

It also includes a retrieval evaluation pipeline using **Recall@k** and **Mean Reciprocal Rank (MRR)**.

---

## Overview

Research RAG Assistant is designed to support structured exploration of scientific and technical PDF documents.

Instead of relying only on keyword matching, the system represents text chunks as dense semantic embeddings and retrieves evidence according to similarity with the user's query.

A second-stage **cross-encoder reranker** evaluates the retrieved candidates more precisely and reorders them before they are returned.

The system supports:

- PDF ingestion
- Semantic document search
- Cross-encoder evidence reranking
- Citation-aware retrieval
- Question answering over indexed documents
- Multi-document comparison
- Retrieval quality evaluation
- Local vector indexing
- FastAPI REST API
- Interactive Swagger/OpenAPI documentation

No uploaded research papers are distributed with this repository.

---

## Architecture

The retrieval pipeline is:

```text
PDF Document
     |
     v
PDF Text Extraction
     |
     v
Page-aware Chunking
     |
     v
Sentence Transformer Embeddings
     |
     v
Vector Index
     |
     v
Dense Semantic Retrieval
     |
     v
Cross-Encoder Reranking
     |
     v
Ranked Evidence
     |
     +--------------------+
     |                    |
     v                    v
Question Answering    Multi-document Comparison
     |                    |
     v                    v
Citations            Per-document Evidence
```

At ingestion time:

```text
PDF
 -> extract pages
 -> split pages into overlapping chunks
 -> generate embeddings
 -> store embeddings and metadata
```

At query time:

```text
Question
 -> embed query
 -> retrieve candidate chunks
 -> rerank candidates
 -> return highest-ranked evidence
```

---

## Features

### PDF Ingestion

PDF files can be uploaded through the API.

The ingestion pipeline:

1. Stores the uploaded document locally
2. Extracts text page by page
3. Splits the document into overlapping chunks
4. Generates semantic embeddings for every chunk
5. Adds the chunks and metadata to the local vector index

Stored metadata includes:

- Document name
- Page number
- Chunk identifier
- Extracted text

---

### Semantic Search

The `/search` endpoint performs semantic retrieval over indexed document chunks.

Instead of matching only exact words, the query is embedded into the same vector space as the document chunks.

The system first retrieves a larger candidate set and then reranks those candidates.

Example request:

```json
{
  "query": "What methods were used to forecast migraine?",
  "top_k": 5
}
```

Typical result fields include:

```json
{
  "chunk_id": "paper.pdf:p2:c2",
  "document": "paper.pdf",
  "page": 2,
  "text": "...",
  "score": 0.72,
  "reranker_score": 3.91
}
```

---

## Cross-Encoder Reranking

Initial semantic retrieval is efficient because it compares vector embeddings.

However, embedding similarity does not always produce the optimal ordering of candidate passages.

For this reason, the project includes a second-stage **cross-encoder reranker**.

```text
Query
  |
  v
Dense Retrieval
  |
  v
Candidate Chunks
  |
  v
Cross-Encoder
  |
  v
Final Ranked Evidence
```

The cross-encoder jointly evaluates:

```text
(query, candidate passage)
```

and produces a relevance score for each candidate.

This improves evidence ordering before results are returned.

---

## Question Answering

The `/ask` endpoint combines retrieval with answer generation.

Example:

```json
{
  "question": "What limitations did the authors report?",
  "top_k": 5
}
```

The system returns an answer together with supporting citations.

```json
{
  "answer": "...",
  "citations": [...]
}
```

When no external LLM is configured, the system remains usable as a retrieval-oriented assistant based on evidence extracted from the indexed documents.

This keeps the core retrieval pipeline local and avoids requiring a paid external API.

---

## Citations

Retrieved evidence preserves document provenance.

Citation information includes:

- Document
- Page
- Chunk ID
- Retrieved text
- Retrieval score
- Reranker score where applicable

Human-readable citations can be represented as:

```text
[paper.pdf, p. 2]
```

This makes it possible to trace retrieved evidence back to the original source document.

---

## Multi-Document Comparison

The `/compare` endpoint allows the same question to be evaluated across multiple indexed documents.

Example request:

```json
{
  "question": "Compare the data sources and methods used to study migraine.",
  "documents": [
    "paper_a.pdf",
    "paper_b.pdf"
  ],
  "top_k_per_document": 3
}
```

The system:

1. Performs semantic retrieval
2. Filters candidate chunks by document
3. Reranks evidence independently for each document
4. Returns the strongest evidence for every paper
5. Generates an extractive comparison summary
6. Identifies frequently shared terms across the selected evidence

Example response structure:

```json
{
  "question": "Compare the data sources and methods used to study migraine.",
  "documents": [
    {
      "document": "paper_a.pdf",
      "evidence": [
        {
          "rank": 1,
          "document": "paper_a.pdf",
          "page": 2,
          "chunk_id": "paper_a.pdf:p2:c2",
          "retrieval_score": 0.6248,
          "reranker_score": 3.3921,
          "citation": "[paper_a.pdf, p. 2]",
          "text": "..."
        }
      ]
    }
  ],
  "summary": {
    "papers": [],
    "shared_terms": [],
    "note": "Extractive comparison summary generated from the highest-ranked evidence chunks; no external LLM used."
  }
}
```

This functionality can support:

- Literature review workflows
- Methodology comparison
- Dataset comparison
- Limitation analysis
- Model performance comparison
- Research paper review

---

## Retrieval Evaluation

The project contains a dedicated retrieval evaluation module.

The retrieval system is evaluated using:

### Recall@k

Recall@k measures how many relevant evidence chunks appear in the first `k` retrieved results.

Examples:

```text
Recall@3
Recall@5
```

Higher values indicate that relevant evidence is more consistently retrieved near the top of the ranking.

### Mean Reciprocal Rank

Mean Reciprocal Rank measures how early the first relevant result appears.

For one query:

```text
Reciprocal Rank = 1 / rank_of_first_relevant_result
```

The final MRR is the average reciprocal rank across all evaluation queries.

An MRR closer to `1.0` indicates that relevant evidence tends to appear very high in the ranking.

---

## Example Retrieval Evaluation

A small evaluation set of four research questions was used during development.

Example result after introducing reranking:

```text
Queries:        4
Mean Recall@3:  0.4166
Mean Recall@5:  0.4166
MRR:            0.75
```

During development, MRR increased from approximately:

```text
0.58 -> 0.75
```

after introducing the second-stage reranking pipeline on this evaluation set.

Because the evaluation dataset is intentionally small, these values should be interpreted as development diagnostics rather than as a large-scale benchmark.

Evaluation code:

```text
src/evaluation/retrieval.py
```

Example evaluation output:

```text
evaluation/retrieval_eval.json
```

Run the evaluation with:

```bash
python -m src.evaluation.retrieval
```

---

## API Endpoints

### Health Check

```http
GET /health
```

Returns the service status and number of indexed chunks.

Example:

```json
{
  "status": "ok",
  "indexed_chunks": 144
}
```

---

### Upload Document

```http
POST /documents/upload
```

Uploads and indexes a PDF document.

Example response:

```json
{
  "document": "research-paper.pdf",
  "pages": 13,
  "chunks_added": 83,
  "index_size": 144
}
```

---

### Search

```http
POST /search
```

Performs semantic retrieval and reranking.

Example:

```json
{
  "query": "What data sources were used in the study?",
  "top_k": 5
}
```

---

### Ask

```http
POST /ask
```

Retrieves relevant evidence and produces a citation-aware answer.

Example:

```json
{
  "question": "What limitations were reported?",
  "top_k": 5
}
```

---

### Compare

```http
POST /compare
```

Compares evidence across multiple indexed documents.

Example:

```json
{
  "question": "Compare the data sources and methods used to study migraine.",
  "documents": [
    "paper_a.pdf",
    "paper_b.pdf"
  ],
  "top_k_per_document": 3
}
```

---

## Interactive API Documentation

FastAPI automatically provides Swagger documentation.

After starting the API, open:

```text
http://127.0.0.1:8000/docs
```

The Swagger interface can be used to:

- Upload PDFs
- Run semantic searches
- Ask questions
- Compare documents
- Inspect API schemas
- Inspect JSON responses

OpenAPI JSON is available at:

```text
http://127.0.0.1:8000/openapi.json
```

---

## Tech Stack

### Backend

- Python 3.11
- FastAPI
- Uvicorn
- Pydantic

### Machine Learning and Retrieval

- Sentence Transformers
- Transformer-based embeddings
- Cross-Encoder reranking
- FAISS
- NumPy
- PyTorch

### PDF Processing

- PyMuPDF

### Testing

- pytest
- FastAPI TestClient

### Engineering

- Docker
- GitHub Actions
- Environment-based configuration
- Persistent local vector index

---

## Project Structure

```text
research-rag-assistant/
|
|-- .github/
|   `-- workflows/
|       `-- ci.yml
|
|-- data/
|   |-- indexes/
|   |   `-- .gitkeep
|   |
|   `-- uploads/
|       `-- .gitkeep
|
|-- evaluation/
|   `-- retrieval_eval.json
|
|-- src/
|   |-- api/
|   |   |-- __init__.py
|   |   `-- main.py
|   |
|   |-- core/
|   |   |-- __init__.py
|   |   |-- chunking.py
|   |   |-- config.py
|   |   |-- embeddings.py
|   |   |-- generation.py
|   |   |-- pdf.py
|   |   |-- rag.py
|   |   |-- reranker.py
|   |   `-- vector_store.py
|   |
|   |-- evaluation/
|   |   |-- __init__.py
|   |   `-- retrieval.py
|   |
|   |-- __init__.py
|   `-- schemas.py
|
|-- tests/
|   |-- test_api.py
|   |-- test_chunking.py
|   `-- test_vector_store.py
|
|-- .env.example
|-- .gitignore
|-- Dockerfile
|-- LICENSE
|-- pyproject.toml
|-- README.md
`-- requirements.txt
```

---

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd research-rag-assistant
```

### 2. Create a virtual environment

Windows:

```powershell
py -3.11 -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the API

Start the FastAPI development server:

```bash
uvicorn src.api.main:app --reload
```

API:

```text
http://127.0.0.1:8000
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

---

## Running Tests

Run the complete test suite with:

```bash
pytest -q
```

Current development result:

```text
8 passed
```

The test suite covers core functionality including:

- API health checks
- Upload validation
- Semantic search
- Document comparison
- Comparison response structure
- Evidence ranking fields
- Chunking
- Vector-store behavior

---

## Data and PDFs

Uploaded PDF documents are **not included in this repository**.

Generated vector indexes are also excluded from version control.

The following directories are intentionally ignored:

```text
data/uploads/
data/indexes/
```

Only `.gitkeep` files are tracked so the directory structure remains available after cloning.

To use the project, upload your own PDF documents through:

```http
POST /documents/upload
```

This keeps user documents and generated indexes local and prevents research papers or private documents from being unintentionally published through Git.

---

## Configuration

Environment-specific configuration can be stored in a local `.env` file.

An example configuration file is provided as:

```text
.env.example
```

The real `.env` file is excluded from Git.

Sensitive credentials and API keys should never be committed to the repository.

---

## Local-First Design

The core retrieval pipeline operates locally.

This includes:

- PDF extraction
- Chunking
- Embedding generation
- Vector indexing
- Semantic retrieval
- Cross-encoder reranking
- Multi-document evidence comparison
- Retrieval evaluation

This allows the application to be developed and tested without requiring a paid external LLM API.

A local or external language model can be added later as an optional generation layer without redesigning the retrieval architecture.

---

## Limitations

The current implementation has several intentional limitations:

- The retrieval benchmark currently contains only a small number of manually defined evaluation queries.
- Multi-document summaries are extractive rather than fully generative when no LLM is configured.
- PDF parsing quality depends on the structure and quality of the source PDF.
- Scanned image-only PDFs may require OCR.
- The current vector index is designed primarily for local development rather than distributed large-scale retrieval.
- Evaluation currently focuses primarily on retrieval quality rather than full end-to-end answer correctness.
- No external LLM is required by the core system.

---

## Future Improvements

Potential extensions include:

- Local LLM integration
- Optional external LLM integration
- Answer-generation evaluation
- Faithfulness and groundedness evaluation
- Larger retrieval benchmark
- Hybrid lexical and dense retrieval
- Query rewriting
- Metadata filtering
- Document-level ranking
- More advanced citation verification
- Persistent database-backed metadata storage
- Web frontend
- Streaming answers
- Docker Compose deployment
- GPU acceleration where available

---

## Why This Project

This project explores the engineering components required for a practical RAG system rather than treating RAG as only an LLM prompting problem.

The implementation focuses on:

- Retrieval quality
- Evidence provenance
- Reranking
- Evaluation
- API design
- Reproducibility
- Multi-document research workflows

The architecture can serve as a foundation for applications involving scientific papers, technical documentation, reports, and other document-heavy knowledge bases.

---

## License

This project is licensed under the MIT License.

See the `LICENSE` file for details.