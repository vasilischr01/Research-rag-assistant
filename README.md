# Research RAG Assistant

[![CI](https://github.com/vasilischr01/Research-rag-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/vasilischr01/Research-rag-assistant/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

A local-first Retrieval-Augmented Generation (RAG) backend for scientific PDF analysis with **dense retrieval, BM25 lexical retrieval, hybrid Reciprocal Rank Fusion (RRF), optional cross-encoder reranking, citation-aware question answering, multi-document comparison, quantitative retrieval evaluation, and API security hardening**.

The system ingests research PDFs, extracts and chunks their content, generates normalized semantic embeddings, persists them in FAISS, builds a BM25 lexical index, combines lexical and dense rankings through RRF, optionally reranks candidates with a cross-encoder, and exposes the workflow through FastAPI.

The default retrieval strategy is **Hybrid RRF**, selected from measured benchmark results rather than by assumption.

---

## Benchmark Highlights

- **0.8055 MRR** with Hybrid Dense + BM25 + RRF — highest MRR in the current evaluation
- **0.9583 Recall@5** with BM25 — highest top-five retrieval coverage
- **+34.9% MRR** for Hybrid RRF relative to the dense FAISS baseline
- **+10.5% Recall@5** for Hybrid RRF relative to the dense baseline
- Hybrid RRF outperformed Hybrid + Cross-Encoder reranking on MRR
- **5 selectable retrieval strategies**
- **3 configurable candidate-pool modes**
- **27 automated tests**
- GitHub Actions CI
- Evaluation uses **document + page relevance annotations** to avoid page-number collisions across PDFs

## Engineering Validation

```text
27 automated tests passed
Ruff: All checks passed
Five-way retrieval benchmark included
Candidate-pool ablation included
Security/upload controls covered by tests
```

---

## Overview

Research RAG Assistant is designed for structured exploration of scientific and technical PDF documents.

It combines:

- **Dense semantic retrieval** for semantic similarity and paraphrases
- **BM25 lexical retrieval** for exact and rare-term matches
- **Reciprocal Rank Fusion (RRF)** to combine rankings without score calibration
- **Optional cross-encoder reranking** for second-stage relevance scoring
- **Citation-aware outputs** with document/page provenance
- **Multi-document comparison**
- **Retrieval benchmarking and ablation experiments**
- **FastAPI REST endpoints**
- **Local-first execution**
- **Security-hardened file uploads and API responses**

Uploaded papers and generated indexes are excluded from Git.

---

## Architecture

### Document Ingestion

```text
PDF Document
    |
    v
Page-Aware Text Extraction
    |
    v
Overlapping Text Chunks
    |
    +-----------------------------+
    |                             |
    v                             v
Sentence Transformer         BM25 Index
Embeddings
    |
    v
FAISS Vector Index
```

Both retrieval indexes operate over the same chunk metadata:

```text
chunk_id
document
page
text
```

After ingestion, the BM25 index is rebuilt from the complete locally persisted chunk collection.

### Default Query Pipeline

```text
                    Query
                      |
            +---------+---------+
            |                   |
            v                   v
      Dense Retrieval      BM25 Retrieval
          (FAISS)            (Lexical)
            |                   |
            +---------+---------+
                      |
                      v
            Reciprocal Rank Fusion
                      |
                      v
                  Final Top-K
                      |
            +---------+---------+
            |                   |
            v                   v
   Question Answering    Citation-Aware Output
```

The default `hybrid` strategy does **not** apply a cross-encoder after RRF because the benchmark showed higher MRR without the extra reranking stage.

---

## Retrieval Strategies

| Strategy | Pipeline | Reranker |
|---|---|---|
| `dense` | FAISS dense retrieval | No |
| `bm25` | BM25 lexical retrieval | No |
| `hybrid` | Dense + BM25 + RRF | No |
| `dense_reranked` | Dense → Cross-Encoder | Yes |
| `hybrid_reranked` | Dense + BM25 → RRF → Cross-Encoder | Yes |

Default:

```text
hybrid
```

---

## Retrieval Modes

| Mode | Candidate Pool | Intended Use |
|---|---:|---|
| `fast` | 5 | Lower candidate-processing cost |
| `balanced` | 10 | Intermediate configuration |
| `quality` | 20 | Larger retrieval candidate pool |

Default:

```text
quality
```

---

## Retrieval Components

### Dense Semantic Retrieval

- Sentence Transformer embeddings
- `normalize_embeddings=True`
- FAISS `IndexFlatIP`
- L2-normalized vectors, so inner-product ranking is cosine-equivalent

### BM25 Lexical Retrieval

Default parameters:

```text
k1 = 1.5
b = 0.75
```

BM25 performed especially well on the current scientific corpus, producing the highest measured Recall@5.

### Reciprocal Rank Fusion

For rank position `r`:

```text
RRF contribution = 1 / (k + r)
```

with:

```text
k = 60
```

Hybrid results preserve:

```text
rrf_score
dense_score
bm25_score
```

### Cross-Encoder Reranking

Current model:

```text
cross-encoder/ms-marco-MiniLM-L6-v2
```

Reranking is optional because additional neural inference did not improve every retrieval configuration.

---

## PDF Ingestion

The ingestion pipeline:

1. Stores the uploaded PDF locally
2. Extracts text page by page
3. Splits pages into overlapping chunks
4. Generates normalized semantic embeddings
5. Adds embeddings and metadata to FAISS
6. Rebuilds BM25 from persisted chunks

Stored metadata includes:

- Document name
- Page number
- Chunk ID
- Extracted text

### Upload Hardening

The upload endpoint validates:

```text
.pdf extension
MIME/content type
PDF signature (%PDF-)
20 MB file-size limit
duplicate filenames
safe basename extraction
partial-file cleanup on failure
```

Behavior:

- Invalid PDF → `400 Bad Request`
- Duplicate filename → `409 Conflict`
- Oversized PDF → `413 Request Entity Too Large`
- Unexpected processing failure → sanitized `500 Internal Server Error`

The API does not trust only the filename or MIME type; it also validates the `%PDF-` file signature.

---

## Search

Endpoint:

```http
POST /search
```

Example:

```json
{
  "query": "What methods were used to forecast migraine?",
  "top_k": 5,
  "retrieval_mode": "quality",
  "retrieval_strategy": "hybrid"
}
```

Hybrid results preserve score provenance:

```json
{
  "chunk_id": "paper.pdf:p4:c2",
  "document": "paper.pdf",
  "page": 4,
  "text": "...",
  "score": null,
  "rrf_score": 0.0325,
  "dense_score": 0.712,
  "bm25_score": 3.84,
  "reranker_score": null
}
```

---

## Question Answering

Endpoint:

```http
POST /ask
```

Example:

```json
{
  "question": "What limitations did the authors report?",
  "top_k": 5,
  "retrieval_mode": "quality",
  "retrieval_strategy": "hybrid"
}
```

Each citation preserves document, page, chunk ID, evidence text, and available retrieval scores.

When no external LLM is configured, the system remains usable as a retrieval-oriented research assistant.

---

## Multi-Document Comparison

Endpoint:

```http
POST /compare
```

The comparison pipeline:

1. Runs Hybrid Dense + BM25 retrieval
2. Fuses rankings with RRF
3. Separates candidates by requested document
4. Applies per-document cross-encoder reranking
5. Returns strongest evidence per paper
6. Generates an extractive comparison summary
7. Identifies shared terms

Use cases include literature review, methodology comparison, dataset comparison, limitation analysis, and technical-document comparison.

---

## Evaluation

### Ground Truth

The benchmark contains:

```text
12 manually curated research queries
```

Relevance is represented as:

```text
(document, page)
```

rather than page number alone.

### Metrics

The benchmark reports:

- Recall@3
- Recall@5
- Mean Reciprocal Rank (MRR)
- Average latency

### Five-Way Retrieval Benchmark

| Retrieval Strategy | Recall@3 | Recall@5 | MRR | Avg. Latency |
|---|---:|---:|---:|---:|
| **BM25** | **0.8333** | **0.9583** | 0.7500 | **~2.9 ms** |
| Dense FAISS | 0.7083 | 0.7917 | 0.5972 | ~21.6 ms |
| **Hybrid RRF** | 0.7917 | 0.8750 | **0.8055** | ~15.9 ms |
| Dense + Cross-Encoder | 0.7083 | 0.8750 | 0.7361 | ~1210 ms |
| Hybrid + Cross-Encoder | 0.7917 | 0.8750 | 0.7361 | ~1 s |

Latency values are local measurements and are hardware-dependent.

### Benchmark Findings

Hybrid RRF improved MRR from:

```text
0.5972 -> 0.8055
```

approximately:

```text
+34.9%
```

Recall@5 improved from:

```text
0.7917 -> 0.8750
```

approximately:

```text
+10.5%
```

BM25 achieved the highest measured Recall@5, while Hybrid RRF achieved the highest MRR.

Cross-encoder reranking improved the dense baseline but reduced MRR when applied after Hybrid RRF in the current evaluation.

Therefore the default remains:

```text
hybrid
```

---

## Candidate-Pool Ablation

Each configuration was run three times after model warm-up.

| Candidate Pool | Recall@5 | MRR | Median Total Latency |
|---:|---:|---:|---:|
| 5 | 0.792 | **0.788** | **256 ms** |
| 10 | 0.792 | 0.736 | 502 ms |
| 20 | **0.875** | 0.736 | 1245 ms |
| 40 | 0.833 | 0.715 | 2663 ms |

Findings:

- Small candidate pools substantially reduce reranking latency
- `candidate_k = 20` produced the highest Recall@5
- Increasing from 20 to 40 increased latency without improving quality
- `candidate_k = 5` is a strong low-latency reranking operating point

---

## API Security and Reliability

### Security Headers

Normal responses include:

```text
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
Referrer-Policy: no-referrer
Permissions-Policy: camera=(), microphone=(), geolocation=()
Cache-Control: no-store
```

### Request Size Limits

- JSON/API requests: **64 KiB**
- PDF payload: **20 MiB**
- Separate multipart upload request limit

The file-size limit is enforced while streaming the upload.

### Rate Limiting

```text
60 requests / 60 seconds / client IP
```

Exceeded limits return:

```text
429 Too Many Requests
Retry-After: 60
```

The current limiter is process-local and intended for single-instance/local deployment. A distributed deployment should use an API gateway, ingress controller, or shared store.

### Error Sanitization

The API returns controlled client-facing errors and avoids exposing raw internal exception strings.

---

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Health and indexed-chunk status |
| `POST` | `/documents/upload` | Upload and index a PDF |
| `POST` | `/search` | Search indexed evidence |
| `POST` | `/ask` | Retrieve and answer with citations |
| `POST` | `/compare` | Compare evidence across documents |

Swagger:

```text
http://127.0.0.1:8000/docs
```

OpenAPI:

```text
http://127.0.0.1:8000/openapi.json
```

---

## Automated Tests

Current result:

```text
27 passed
```

The suite covers:

- Health endpoint
- Upload validation
- Search behavior
- Retrieval modes
- BM25 behavior
- Reciprocal Rank Fusion
- Document comparison
- Chunking
- Vector-store behavior
- Security headers
- Fake-PDF rejection
- Duplicate filename protection
- Oversized PDF rejection
- Rate-limit enforcement
- Sanitized 400/500 error responses

Run:

```bash
pytest -q
```

---

## Linting

```bash
ruff check .
```

Current status:

```text
All checks passed!
```

---

## Continuous Integration

GitHub Actions runs on:

```text
push
pull_request
```

The CI pipeline configures Python 3.11, installs dependencies, and executes automated checks.

Workflow:

```text
.github/workflows/ci.yml
```

---

## Tech Stack

### Backend

- Python 3.11
- FastAPI
- Uvicorn
- Pydantic

### Retrieval / ML

- Sentence Transformers
- FAISS
- BM25
- Reciprocal Rank Fusion
- Cross-Encoder reranking
- NumPy
- PyTorch

### PDF Processing

- PyMuPDF

### Quality / Infrastructure

- pytest
- Ruff
- FastAPI TestClient
- Docker
- GitHub Actions
- Environment-based configuration

---

## Project Structure

```text
research-rag-assistant/
├── .github/
│   └── workflows/
│       └── ci.yml
├── data/
│   ├── indexes/
│   │   └── .gitkeep
│   └── uploads/
│       └── .gitkeep
├── evaluation/
│   ├── retrieval_eval.json
│   ├── retrieval_benchmark.json
│   └── candidate_k_ablation.json
├── src/
│   ├── api/
│   │   └── main.py
│   ├── core/
│   │   ├── bm25.py
│   │   ├── chunking.py
│   │   ├── config.py
│   │   ├── embeddings.py
│   │   ├── generation.py
│   │   ├── hybrid.py
│   │   ├── pdf.py
│   │   ├── rag.py
│   │   ├── reranker.py
│   │   └── vector_store.py
│   ├── evaluation/
│   │   ├── ablation.py
│   │   └── retrieval.py
│   └── schemas.py
├── tests/
│   ├── test_api.py
│   ├── test_bm25.py
│   ├── test_chunking.py
│   ├── test_hybrid.py
│   └── test_vector_store.py
├── .env.example
├── .gitignore
├── Dockerfile
├── LICENSE
├── pyproject.toml
├── README.md
└── requirements.txt
```

---

## Installation

### 1. Clone

```bash
git clone https://github.com/vasilischr01/Research-rag-assistant.git
cd Research-rag-assistant
```

### 2. Create a virtual environment

Windows:

```powershell
py -3.11 -m venv .venv
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

## Docker

Build:

```bash
docker build -t research-rag-assistant .
```

The image provides a reproducible environment for the API and retrieval stack.

---

## Data, Privacy, and Secrets

The repository excludes:

```text
data/uploads/
data/indexes/
.env
```

Only `.gitkeep` files preserve local data-directory structure.

The optional language-model key is read at runtime from:

```text
LLM_API_KEY
```

`.env.example` contains an empty placeholder. Real credentials belong in the ignored `.env` file or an external secret-management system.

---

## Evaluation Files

```text
evaluation/retrieval_eval.json
evaluation/retrieval_benchmark.json
evaluation/candidate_k_ablation.json
```

Run the main benchmark:

```bash
python -m src.evaluation.retrieval
```

Run the candidate-pool ablation:

```bash
python -m src.evaluation.ablation
```

---

## Limitations

- The benchmark contains only **12 manually curated queries**
- The evaluation corpus remains limited
- Relevance annotations are page-level rather than chunk-level
- Retrieval evaluation does not yet measure full answer correctness
- Multi-document summaries are extractive without a configured language model
- PDF parsing quality depends on document structure
- Image-only PDFs may require OCR
- FAISS/BM25 are designed for local rather than distributed retrieval
- Cross-encoder and benchmark latency depend on hardware
- The rate limiter is process-local
- No authentication/RBAC layer is currently included
- PDF signature validation is intentionally lightweight and is not malware scanning

---

## Future Improvements

- Larger multi-document evaluation set
- 30–50+ manually annotated queries
- Chunk-level relevance judgments
- More diverse scientific/technical corpora
- Confidence intervals for retrieval metrics
- Answer-quality, faithfulness, and groundedness evaluation
- Automated citation verification
- Query rewriting
- Metadata filtering
- Persistent database-backed metadata
- Web frontend
- Streaming responses
- Prometheus metrics
- Retrieval latency monitoring
- Load testing
- Benchmark regression testing
- Distributed rate limiting
- Authentication / RBAC
- Larger-scale vector-store evaluation

---

## Why This Project

This project treats RAG as an **information-retrieval engineering system**, not only an LLM prompting exercise.

It demonstrates:

```text
dense retrieval
lexical retrieval
hybrid retrieval
Reciprocal Rank Fusion
neural reranking
evidence provenance
quantitative evaluation
ablation experiments
benchmark-driven architecture decisions
quality-latency trade-offs
API engineering
security hardening
automated testing
continuous integration
local-first execution
multi-document research workflows
```

The architecture can serve as a foundation for scientific papers, technical documentation, internal knowledge bases, engineering reports, and other document-heavy information systems.

---

## License

MIT License.
