# Research RAG Assistant

A local-first Retrieval-Augmented Generation (RAG) backend for scientific PDF analysis, semantic retrieval, neural reranking, citation-aware question answering, multi-document comparison, and quantitative retrieval evaluation.

The project implements an end-to-end document retrieval pipeline that can ingest research papers, extract and chunk their content, generate normalized semantic embeddings, retrieve relevant evidence with FAISS, rerank candidate passages using a cross-encoder, and expose the complete workflow through a FastAPI REST API.

Beyond basic RAG functionality, the project includes retrieval benchmarking, candidate-pool ablation experiments, latency measurements, configurable retrieval modes, automated tests, Docker support, and GitHub Actions CI.

---

## Overview

Research RAG Assistant is designed to support structured exploration of scientific and technical PDF documents.

Instead of relying only on keyword matching, the system represents document chunks as dense semantic embeddings and retrieves evidence according to semantic similarity with the user's query.

A second-stage **cross-encoder reranker** then jointly evaluates each query-passage pair and reorders the retrieved candidates before the final results are returned.

The system supports:

- PDF ingestion
- Page-aware text extraction
- Overlapping text chunking
- Semantic document search
- Dense vector retrieval with FAISS
- Cross-encoder neural reranking
- Citation-aware retrieval
- Question answering over indexed documents
- Multi-document comparison
- Page-level evidence provenance
- Retrieval quality benchmarking
- Candidate-pool ablation experiments
- Latency evaluation
- Configurable retrieval modes
- Persistent local vector indexing
- FastAPI REST API
- Interactive Swagger/OpenAPI documentation
- Automated testing
- Docker support
- GitHub Actions CI

Uploaded research papers and generated vector indexes are not distributed with the repository.

---

## Architecture

The main retrieval pipeline is:

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
Normalized Sentence-Transformer Embeddings
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
 -> generate normalized embeddings
 -> store embeddings and metadata
```

### Query Pipeline

```text
Question
 -> generate normalized query embedding
 -> retrieve candidate chunks with FAISS
 -> rerank candidates with CrossEncoder
 -> return highest-ranked evidence
```

---

## Retrieval Design

### Dense Retrieval

Document chunks and queries are encoded with a Sentence Transformer using:

```python
normalize_embeddings=True
```

The vector index uses:

```text
FAISS IndexFlatIP
```

Because both document and query embeddings are L2-normalized, inner-product search is equivalent to **cosine-similarity ranking**.

This provides efficient first-stage semantic retrieval.

### Cross-Encoder Reranking

Dense embedding similarity is efficient, but it does not always provide the best ranking of retrieved passages.

For this reason, the system performs second-stage neural reranking.

```text
Query
  |
  v
Dense Retrieval
  |
  v
Candidate Pool
  |
  v
Cross-Encoder
  |
  v
Final Top-K Evidence
```

The cross-encoder jointly evaluates:

```text
(query, candidate passage)
```

and assigns an independent relevance score to every candidate.

The current reranker is:

```text
cross-encoder/ms-marco-MiniLM-L6-v2
```

The candidates are then reordered according to their reranker scores.

---

## Retrieval Modes

The API exposes three retrieval modes that control the number of dense candidates passed to the cross-encoder.

| Mode | Candidate Pool | Intended Use |
|---|---:|---|
| `fast` | 5 | Lowest latency |
| `balanced` | 10 | Intermediate configuration |
| `quality` | 20 | Higher retrieval coverage |

The default mode is:

```text
quality
```

These modes were selected based on an empirical candidate-pool ablation experiment.

Example:

```json
{
  "query": "What methods were used to forecast migraine?",
  "top_k": 5,
  "retrieval_mode": "quality"
}
```

The same retrieval modes are available through the `/ask` endpoint.

---

## Features

### PDF Ingestion

PDF files can be uploaded through the API.

The ingestion pipeline:

1. Stores the uploaded document locally
2. Extracts text page by page
3. Splits pages into overlapping chunks
4. Generates normalized semantic embeddings
5. Adds embeddings and metadata to the persistent FAISS index

Stored metadata includes:

- Document name
- Page number
- Chunk identifier
- Extracted text

---

## Semantic Search

The `/search` endpoint performs dense semantic retrieval followed by cross-encoder reranking.

Example request:

```json
{
  "query": "What methods were used to forecast migraine?",
  "top_k": 5,
  "retrieval_mode": "quality"
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

The fields represent:

- `score`: first-stage dense retrieval similarity
- `reranker_score`: second-stage cross-encoder relevance score

---

## Question Answering

The `/ask` endpoint combines retrieval with answer generation.

Example:

```json
{
  "question": "What limitations did the authors report?",
  "top_k": 5,
  "retrieval_mode": "quality"
}
```

Example response:

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

The retrieval mode controls the candidate pool used before reranking.

When no external LLM is configured, the application remains usable as a retrieval-oriented research assistant based on evidence extracted from indexed documents.

This keeps the core retrieval pipeline local and avoids requiring a paid hosted API.

---

## Citations and Evidence Provenance

Every retrieved chunk preserves its source metadata.

Evidence information includes:

- Document
- Page
- Chunk ID
- Retrieved text
- Dense retrieval score
- Cross-encoder reranker score

Human-readable citations can be represented as:

```text
[paper.pdf, p. 4]
```

This allows retrieved evidence to be traced back to its original source document.

---

## Multi-Document Comparison

The `/compare` endpoint evaluates the same research question across multiple indexed documents.

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
4. Returns the strongest evidence from every requested paper
5. Generates an extractive comparison summary
6. Identifies frequently shared terms across selected evidence

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

Potential use cases include:

- Literature review workflows
- Methodology comparison
- Dataset comparison
- Limitation analysis
- Model performance comparison
- Research paper review

---

# Evaluation

## Retrieval Metrics

The project includes a dedicated retrieval evaluation pipeline.

Evaluation focuses on whether manually annotated relevant source pages appear near the top of the retrieved ranking.

### Recall@k

Recall@k measures the fraction of relevant evidence retrieved within the first `k` results.

The benchmark reports:

```text
Recall@3
Recall@5
```

Higher values indicate better retrieval coverage near the top of the ranking.

### Mean Reciprocal Rank

Mean Reciprocal Rank measures how early the first relevant result appears.

For an individual query:

```text
Reciprocal Rank = 1 / rank_of_first_relevant_result
```

MRR is the mean reciprocal rank across all evaluation queries.

An MRR closer to `1.0` indicates that relevant evidence tends to appear earlier in the ranking.

---

## Dense vs Reranked Retrieval Benchmark

Retrieval quality was evaluated using **12 manually curated research questions** with page-level relevance annotations.

The experiment compares:

```text
Dense FAISS Retrieval
```

against:

```text
Dense FAISS Retrieval
        +
Cross-Encoder Reranking
```

The same initial dense candidate pool is used for the reranking experiment so that the contribution of the second-stage model can be measured directly.

### Results

| Retrieval Strategy | Recall@3 | Recall@5 | MRR | Avg. Query Latency |
|---|---:|---:|---:|---:|
| Dense Retrieval | 0.708 | 0.792 | 0.597 | 397 ms |
| Dense + Cross-Encoder Reranking | 0.708 | **0.875** | **0.736** | 2202 ms |

Cross-encoder reranking produced:

- **23.3% relative improvement in MRR**
- **10.5% relative improvement in Recall@5**
- No observed improvement in Recall@3

The experiment demonstrates that second-stage reranking improves evidence ordering and top-five retrieval coverage, while introducing additional inference latency.

This exposes a measurable **retrieval quality vs latency trade-off**.

---

## Candidate Pool Ablation

A separate ablation experiment evaluated how many dense candidates should be passed to the cross-encoder.

Each candidate-pool configuration was evaluated over the same **12 queries** using **three repeated runs after model warm-up**.

| Candidate Pool | Recall@5 | MRR | Median Total Latency |
|---:|---:|---:|---:|
| 5 | 0.792 | **0.788** | **256 ms** |
| 10 | 0.792 | 0.736 | 502 ms |
| 20 | **0.875** | 0.736 | 1245 ms |
| 40 | 0.833 | 0.715 | 2663 ms |

### Findings

- `candidate_k = 5` achieved the **lowest latency** and highest measured MRR.
- `candidate_k = 20` achieved the **highest Recall@5**.
- Increasing the pool from 20 to 40 substantially increased latency without improving retrieval quality.
- `candidate_k = 10` did not provide a better operating point than the configurations at 5 or 20 in this evaluation.

The experiment therefore identifies two useful operating points:

```text
fast
candidate_k = 5
```

for latency-sensitive retrieval, and:

```text
quality
candidate_k = 20
```

for higher retrieval coverage.

The API exposes both configurations directly through retrieval modes.

---

## Evaluation Configuration

### Main Retrieval Benchmark

- Evaluation queries: **12**
- Candidate pool: **20 chunks**
- Metrics:
  - Recall@3
  - Recall@5
  - Mean Reciprocal Rank
- Dense retrieval:
  - Sentence Transformer embeddings
  - L2-normalized vectors
  - FAISS inner-product search
  - Cosine-equivalent ranking
- Reranker:
  - `cross-encoder/ms-marco-MiniLM-L6-v2`
- Measurements:
  - Dense retrieval latency
  - Cross-encoder reranking latency
  - Total retrieval latency

### Candidate-Pool Ablation

Candidate values:

```text
5
10
20
40
```

Runs per configuration:

```text
3
```

Model warm-up is performed before timing to reduce one-time initialization effects.

---

## Evaluation Files

Evaluation queries:

```text
evaluation/retrieval_eval.json
```

Main benchmark output:

```text
evaluation/retrieval_benchmark.json
```

Candidate-pool ablation output:

```text
evaluation/candidate_k_ablation.json
```

Main benchmark implementation:

```text
src/evaluation/retrieval.py
```

Ablation implementation:

```text
src/evaluation/ablation.py
```

Run the retrieval benchmark:

```bash
python -m src.evaluation.retrieval
```

Run the candidate-pool ablation:

```bash
python -m src.evaluation.ablation
```

---

# API

## Health Check

```http
GET /health
```

Returns service status and current number of indexed chunks.

Example:

```json
{
  "status": "ok",
  "indexed_chunks": 144
}
```

---

## Upload Document

```http
POST /documents/upload
```

Uploads and indexes a PDF.

Example response:

```json
{
  "document": "research-paper.pdf",
  "pages": 13,
  "chunks_added": 83,
  "index_size": 144
}
```

Non-PDF uploads are rejected.

---

## Search

```http
POST /search
```

Performs dense semantic retrieval followed by cross-encoder reranking.

Example:

```json
{
  "query": "What data sources were used in the study?",
  "top_k": 5,
  "retrieval_mode": "quality"
}
```

Supported retrieval modes:

```text
fast
balanced
quality
```

---

## Ask

```http
POST /ask
```

Retrieves relevant evidence and produces a citation-aware answer.

Example:

```json
{
  "question": "What limitations were reported?",
  "top_k": 5,
  "retrieval_mode": "quality"
}
```

---

## Compare

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

FastAPI automatically exposes Swagger/OpenAPI documentation.

Start the API and open:

```text
http://127.0.0.1:8000/docs
```

The Swagger interface can be used to:

- Upload PDF documents
- Run semantic searches
- Select retrieval modes
- Ask document-grounded questions
- Compare documents
- Inspect request schemas
- Inspect response schemas
- Execute API requests interactively

OpenAPI JSON is available at:

```text
http://127.0.0.1:8000/openapi.json
```

---

# Tech Stack

## Backend

- Python 3.11
- FastAPI
- Uvicorn
- Pydantic

## Machine Learning and Retrieval

- Sentence Transformers
- Transformer-based embeddings
- Cross-Encoder reranking
- FAISS
- NumPy
- PyTorch

## PDF Processing

- PyMuPDF

## Testing

- pytest
- FastAPI TestClient

## Engineering

- Docker
- GitHub Actions
- Environment-based configuration
- Persistent local vector index
- Automated CI testing

---

# Project Structure

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
|   |   |-- ablation.py
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

# Installation

## 1. Clone the Repository

```bash
git clone <repository-url>
cd research-rag-assistant
```

## 2. Create a Virtual Environment

### Windows

```powershell
py -3.11 -m venv .venv
```

Activate:

```powershell
.venv\Scripts\Activate.ps1
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the API

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

# Running Tests

Run the complete test suite:

```bash
pytest -q
```

Current test result:

```text
11 passed
```

The test suite covers functionality including:

- API health checks
- Upload validation
- Semantic search
- Retrieval-mode routing
- Fast retrieval configuration
- Quality retrieval configuration
- Invalid retrieval-mode validation
- Document comparison
- Comparison response structure
- Evidence ranking fields
- Chunking
- Vector-store behavior

---

# Continuous Integration

GitHub Actions runs the test suite automatically on:

```text
push
pull_request
```

The CI workflow:

1. Checks out the repository
2. Configures Python 3.11
3. Uses pip dependency caching
4. Upgrades pip
5. Installs project dependencies
6. Runs the pytest suite

Workflow:

```text
.github/workflows/ci.yml
```

---

# Docker

A Dockerfile is included for containerized execution.

Build the image:

```bash
docker build -t research-rag-assistant .
```

Run it according to the environment and port configuration used by the application.

---

# Data and PDFs

Uploaded PDF documents are **not included in this repository**.

Generated vector indexes are also excluded from version control.

The following directories are intentionally ignored:

```text
data/uploads/
data/indexes/
```

Only `.gitkeep` files are tracked so that the required directory structure remains present after cloning.

To use the project, upload your own PDF documents through:

```http
POST /documents/upload
```

This keeps user documents and generated retrieval artifacts local and prevents private or copyrighted documents from being unintentionally committed.

---

# Configuration

Environment-specific configuration can be stored in a local:

```text
.env
```

An example configuration file is provided:

```text
.env.example
```

Local environment files are excluded from Git, while `.env.example` remains tracked.

Sensitive credentials and API keys should never be committed to the repository.

---

# Local-First Design

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
- Candidate-pool ablation

This allows the application to be developed and evaluated without requiring a paid hosted LLM API.

A local or external language model can be added as an optional generation layer without redesigning the retrieval architecture.

---

# Engineering Decisions

## Two-Stage Retrieval

The system deliberately separates:

```text
efficient first-stage dense retrieval
```

from:

```text
more computationally expensive second-stage neural reranking
```

This allows the pipeline to balance:

- Retrieval coverage
- Ranking quality
- Computational cost
- Query latency

The included benchmark quantifies this trade-off instead of assuming that reranking is beneficial without measuring its cost.

---

## Configurable Retrieval Modes

Candidate-pool size is exposed as a user-facing retrieval mode rather than being permanently hardcoded.

This allows the same retrieval system to support different operating requirements:

```text
fast
balanced
quality
```

The modes are grounded in measured ablation results rather than arbitrary configuration values.

---

## Persistent Vector Index

FAISS embeddings and chunk metadata are persisted locally, allowing indexed documents to remain available between application restarts.

---

## Cosine-Equivalent Dense Search

Embeddings are generated with L2 normalization.

FAISS `IndexFlatIP` therefore ranks normalized query and document vectors equivalently to cosine similarity.

---

## Page-Level Provenance

Every chunk retains source document and page metadata.

This allows retrieved evidence to be traced back to the original PDF.

---

## Local-First Execution

Core retrieval functionality does not depend on external hosted APIs.

This improves reproducibility and supports workflows involving private or sensitive documents.

---

# Limitations

The current implementation has several limitations:

- The benchmark currently contains **12 manually curated evaluation queries** and should be interpreted as a development-level evaluation rather than a large-scale information retrieval benchmark.
- Evaluation primarily measures retrieval performance rather than complete end-to-end answer correctness.
- Page-level relevance annotations are used instead of fine-grained chunk-level human judgments.
- The candidate-pool findings are based on the current evaluation corpus and may not generalize to substantially different document collections.
- Multi-document summaries are extractive when no LLM is configured.
- PDF parsing quality depends on source document structure.
- Scanned image-only PDFs may require OCR.
- The current FAISS index is designed primarily for local development rather than distributed large-scale retrieval.
- Cross-encoder reranking improves retrieval quality but adds inference latency.
- Latency measurements depend on local hardware and should not be interpreted as universal production performance.

---

# Future Improvements

Potential extensions include:

- Larger and more diverse retrieval benchmark
- Multi-document retrieval evaluation
- Chunk-level human relevance annotations
- BM25 baseline
- Hybrid lexical + dense retrieval
- Reciprocal Rank Fusion
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
- GPU acceleration
- Prometheus metrics
- Retrieval latency monitoring
- Automated benchmark regression testing
- Load testing
- Larger-scale vector-store evaluation

---

# Why This Project

This project explores the engineering components required for a practical Retrieval-Augmented Generation system rather than treating RAG as only an LLM prompting problem.

The implementation focuses on:

- Semantic retrieval
- Neural reranking
- Retrieval quality
- Evidence provenance
- Quantitative evaluation
- Ablation experiments
- Quality-latency trade-offs
- Configurable inference behavior
- API design
- Testing
- Continuous integration
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

# License

This project is licensed under the MIT License.

See the `LICENSE` file for details.