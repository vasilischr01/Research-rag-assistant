# Research RAG Assistant

A local-first Retrieval-Augmented Generation (RAG) backend for scientific PDF analysis, semantic search, evidence reranking, citation-aware retrieval, retrieval evaluation, and multi-document comparison.

The project implements an end-to-end document retrieval pipeline that can ingest research papers, extract and chunk their content, generate semantic embeddings, retrieve relevant evidence with FAISS, rerank candidate passages using a cross-encoder, and expose the workflow through a FastAPI REST API.

The system also includes a dedicated retrieval benchmarking pipeline using **Recall@k**, **Mean Reciprocal Rank (MRR)**, and latency measurements.

---

## Overview

Research RAG Assistant is designed to support structured exploration of scientific and technical PDF documents.

Instead of relying on keyword matching, the system represents document chunks as dense semantic embeddings and retrieves evidence according to semantic similarity with the user's query.

A second-stage **cross-encoder reranker** evaluates the retrieved candidates more precisely and reorders them before they are returned.

The system supports:

- PDF ingestion
- Page-aware text chunking
- Semantic document search
- Dense vector retrieval with FAISS
- Cross-encoder evidence reranking
- Citation-aware retrieval
- Question answering over indexed documents
- Multi-document comparison
- Retrieval quality benchmarking
- Latency evaluation
- Persistent local vector indexing
- FastAPI REST API
- Interactive Swagger/OpenAPI documentation
- Automated testing
- Docker support
- GitHub Actions CI

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
Page-Aware Chunking
     |
     v
Sentence Transformer Embeddings
     |
     v
FAISS Vector Index
     |
     v
Dense Semantic Retrieval
     |
     v
Candidate Passages
     |
     v
Cross-Encoder Reranking
     |
     v
Ranked Evidence
     |
     +--------------------------+
     |                          |
     v                          v
Question Answering      Multi-Document Comparison
     |                          |
     v                          v
Citation-Aware Output   Per-Document Evidence
```

### Ingestion Pipeline

```text
PDF
 -> extract pages
 -> split pages into overlapping chunks
 -> generate embeddings
 -> store embeddings and metadata
```

### Query Pipeline

```text
Question
 -> embed query
 -> retrieve candidate chunks with FAISS
 -> rerank candidates with CrossEncoder
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
5. Adds embeddings and metadata to the persistent FAISS index

Stored metadata includes:

- Document name
- Page number
- Chunk identifier
- Extracted text

---

### Semantic Search

The `/search` endpoint performs semantic retrieval over indexed document chunks.

Instead of matching exact keywords, the query is embedded into the same vector space as the document chunks.

The system retrieves a larger candidate pool and subsequently reranks those candidates using a cross-encoder.

Example request:

```json
{
  "query": "What methods were used to forecast migraine?",
  "top_k": 5
}
```

Typical result:

```json
{
  "chunk_id": "paper.pdf:p4:c2",
  "document": "paper.pdf",
  "page": 4,
  "text": "...",
  "score": 0.72,
  "reranker_score": 3.91
}
```

The `score` field represents the first-stage dense retrieval similarity score.

The `reranker_score` represents the relevance score assigned by the second-stage cross-encoder.

---

## Cross-Encoder Reranking

Dense semantic retrieval is efficient because queries and document chunks are represented as embeddings and compared in vector space.

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

The current reranker uses:

```text
cross-encoder/ms-marco-MiniLM-L6-v2
```

Candidate passages are reordered according to their cross-encoder relevance scores before the final top-k results are returned.

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

The system returns an answer together with supporting evidence.

Example structure:

```json
{
  "answer": "...",
  "citations": [
    {
      "document": "paper.pdf",
      "page": 9,
      "chunk_id": "paper.pdf:p9:c1",
      "text": "...",
      "score": 0.68,
      "reranker_score": 4.12
    }
  ]
}
```

When no external LLM is configured, the application remains usable as a retrieval-oriented research assistant based on evidence extracted from indexed documents.

This allows the core retrieval pipeline to operate locally without requiring a paid external API.

---

## Citations and Evidence Provenance

Retrieved evidence preserves its source metadata.

Citation information includes:

- Document
- Page
- Chunk ID
- Retrieved text
- Dense retrieval score
- Reranker score where applicable

Human-readable citations can be represented as:

```text
[paper.pdf, p. 4]
```

This allows retrieved evidence to be traced back to its original source document.

---

## Multi-Document Comparison

The `/compare` endpoint allows the same research question to be evaluated across multiple indexed documents.

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

The project includes a dedicated retrieval evaluation pipeline.

Evaluation focuses on whether relevant source pages appear near the top of the retrieved ranking.

### Recall@k

Recall@k measures how much of the relevant evidence is retrieved within the first `k` results.

Examples:

```text
Recall@3
Recall@5
```

Higher values indicate that relevant evidence is more consistently present near the top of the ranking.

### Mean Reciprocal Rank

Mean Reciprocal Rank measures how early the first relevant result appears.

For one query:

```text
Reciprocal Rank = 1 / rank_of_first_relevant_result
```

MRR is the average reciprocal rank across all evaluation queries.

An MRR closer to `1.0` indicates that relevant evidence tends to appear near the beginning of the ranking.

---

## Retrieval Benchmark

Retrieval quality was evaluated on **12 manually curated research questions** with page-level relevance annotations.

The benchmark compares:

```text
Dense FAISS Retrieval
```

against:

```text
Dense FAISS Retrieval
        +
Cross-Encoder Reranking
```

The same dense candidate pool is used for reranking, allowing the quality contribution of the second-stage model to be measured directly.

### Benchmark Results

| Retrieval Strategy | Recall@3 | Recall@5 | MRR | Avg. Query Latency |
|---|---:|---:|---:|---:|
| Dense Retrieval | 0.708 | 0.792 | 0.597 | 397 ms |
| Dense + Cross-Encoder Reranking | 0.708 | **0.875** | **0.736** | 2202 ms |

Cross-encoder reranking produced:

- **23.3% improvement in MRR**
- **10.5% improvement in Recall@5**
- No change in Recall@3

The results demonstrate that second-stage reranking substantially improves the ordering of relevant evidence and increases retrieval coverage within the top five results.

The improvement comes with higher inference latency, exposing an explicit **retrieval-quality vs latency trade-off**.

### Candidate Pool Ablation

The effect of the number of dense candidates passed to the cross-encoder
was evaluated using three repeated runs per configuration after model warm-up.

| Candidate Pool | Recall@5 | MRR | Median Total Latency |
|---:|---:|---:|---:|
| 5 | 0.792 | **0.788** | **256 ms** |
| 10 | 0.792 | 0.736 | 502 ms |
| 20 | **0.875** | 0.736 | 1245 ms |
| 40 | 0.833 | 0.715 | 2663 ms |

The ablation reveals a clear quality-latency trade-off:

- A candidate pool of **20** achieved the highest Recall@5.
- A candidate pool of **5** achieved the highest MRR and lowest latency.
- Increasing the pool from 20 to 40 increased latency substantially without improving retrieval quality.
- A candidate pool of 10 was dominated by the smaller pool in this evaluation.

These results suggest two useful operating points: a low-latency configuration
with 5 candidates and a retrieval-quality-oriented configuration with 20 candidates.

### Evaluation Configuration

- Evaluation queries: **12**
- Dense candidate pool: **20 chunks**
- Metrics:
  - Recall@3
  - Recall@5
  - Mean Reciprocal Rank
- Dense retrieval:
  - Sentence-transformer embeddings
  - FAISS inner-product search
- Reranker:
  - `cross-encoder/ms-marco-MiniLM-L6-v2`
- Benchmark includes:
  - Dense retrieval latency
  - Cross-encoder reranking latency
  - End-to-end retrieval latency

Evaluation implementation:

```text
src/evaluation/retrieval.py
```

Evaluation dataset:

```text
evaluation/retrieval_eval.json
```

Generated benchmark:

```text
evaluation/retrieval_benchmark.json
```

Run the benchmark with:

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

Performs dense semantic retrieval followed by cross-encoder reranking.

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

FastAPI automatically provides Swagger/OpenAPI documentation.

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
|   |-- retrieval_eval.json
|   |-- retrieval_benchmark.json
|   `-- candidate_k_ablation.json
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

## Running the Retrieval Benchmark

Make sure the required documents have already been ingested into the vector store.

Run:

```bash
python -m src.evaluation.retrieval
```

The benchmark report is written to:

```text
evaluation/retrieval_benchmark.json
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

Only `.gitkeep` files are tracked so that the directory structure remains available after cloning.

To use the project, upload your own PDF documents through:

```http
POST /documents/upload
```

This keeps user documents and generated indexes local and prevents research papers or private documents from being unintentionally published through Git.

---

## Configuration

Environment-specific configuration can be stored in a local `.env` file.

An example configuration file is provided:

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
- Page-aware chunking
- Embedding generation
- Vector indexing
- Semantic retrieval
- Cross-encoder reranking
- Multi-document evidence comparison
- Retrieval evaluation
- Benchmark generation

This allows the application to be developed and tested without requiring a paid external LLM API.

A local or external language model can be added as an optional generation layer without redesigning the retrieval architecture.

---

## Engineering Decisions

### Two-Stage Retrieval

The system deliberately separates efficient first-stage dense retrieval from more computationally expensive second-stage reranking.

This enables the retrieval pipeline to balance:

- Retrieval coverage
- Ranking quality
- Computational cost
- Query latency

The benchmark quantifies this trade-off instead of assuming that reranking is always beneficial without cost.

### Persistent Vector Index

FAISS embeddings and chunk metadata are persisted locally, allowing indexed documents to remain available between application restarts.

### Page-Level Provenance

Every chunk retains its source document and page information so that retrieved evidence can be traced back to the original PDF.

### Local-First Execution

Core retrieval functionality does not depend on external hosted APIs, making the system easier to reproduce and suitable for private or sensitive document workflows.

---

## Limitations

The current implementation has several limitations:

- The retrieval benchmark currently contains **12 manually curated evaluation queries**, which provides useful development-level comparison but is not intended to represent a large-scale information retrieval benchmark.
- Evaluation currently focuses primarily on retrieval quality rather than full end-to-end answer correctness.
- Multi-document summaries are extractive when no LLM is configured.
- PDF parsing quality depends on the structure and quality of the source PDF.
- Scanned image-only PDFs may require OCR.
- The current vector index is designed primarily for local development rather than distributed large-scale retrieval.
- Cross-encoder reranking improves ranking quality but introduces significant additional inference latency.
- The current benchmark uses page-level relevance annotations rather than fine-grained chunk-level human relevance judgments.

---

## Future Improvements

Potential extensions include:

- Larger and more diverse retrieval benchmark
- Candidate-pool size ablation experiments
- Hybrid lexical and dense retrieval
- BM25 baseline
- Query rewriting
- Metadata filtering
- Document-level ranking
- Answer-generation evaluation
- Faithfulness evaluation
- Groundedness evaluation
- Automated citation verification
- Local LLM integration
- Optional external LLM integration
- Persistent database-backed metadata storage
- Web frontend
- Streaming answers
- Docker Compose deployment
- GPU acceleration where available
- Prometheus metrics
- Retrieval latency monitoring
- Automated benchmark regression testing

---

## Why This Project

This project explores the engineering components required for a practical Retrieval-Augmented Generation system rather than treating RAG as only an LLM prompting problem.

The implementation focuses on:

- Semantic retrieval
- Retrieval quality
- Evidence provenance
- Neural reranking
- Quantitative evaluation
- Quality-latency trade-offs
- API design
- Reproducibility
- Local-first execution
- Multi-document research workflows

The architecture can serve as a foundation for applications involving:

- Scientific papers
- Technical documentation
- Internal knowledge bases
- Engineering reports
- Research literature
- Other document-heavy information systems

---

## License

This project is licensed under the MIT License.

See the `LICENSE` file for details. 