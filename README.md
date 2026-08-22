# Research RAG Assistant

[![CI](https://github.com/vasilischr01/Research-rag-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/vasilischr01/Research-rag-assistant/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

A local-first Retrieval-Augmented Generation (RAG) backend for scientific PDF analysis with **dense retrieval, BM25 lexical retrieval, hybrid Reciprocal Rank Fusion (RRF), optional cross-encoder reranking, citation-aware question answering, multi-document comparison, and quantitative retrieval evaluation**.

The system implements an end-to-end document retrieval pipeline that can ingest research papers, extract and chunk their content, generate normalized semantic embeddings, persist them in FAISS, build a BM25 lexical index, combine dense and lexical rankings through RRF, optionally rerank candidate passages with a cross-encoder, and expose the complete workflow through a FastAPI REST API.

The default production retrieval strategy is **Hybrid RRF**, selected empirically from a five-way retrieval benchmark rather than chosen by assumption.

## Benchmark Highlights

- **0.8055 MRR** with Hybrid Dense + BM25 + RRF — highest MRR in the current evaluation
- **0.9583 Recall@5** with BM25 — highest top-five retrieval coverage
- **+34.9% MRR** for Hybrid RRF relative to the dense FAISS baseline
- **+10.5% Recall@5** for Hybrid RRF relative to the dense baseline
- Hybrid RRF achieved higher MRR than Hybrid + Cross-Encoder reranking
- **5 selectable retrieval strategies**
- **3 configurable candidate-pool modes**
- **20 automated tests** with GitHub Actions CI
- Evaluation uses **document + page relevance annotations** to avoid page-number collisions across PDFs

---

## Overview

Research RAG Assistant is designed for structured exploration of scientific and technical PDF documents.

The system combines complementary information-retrieval techniques:

- **Dense semantic retrieval** captures semantic similarity even when query and document wording differ.
- **BM25 lexical retrieval** rewards exact and rare term matches.
- **Reciprocal Rank Fusion** combines both rankings without requiring their raw scores to be directly comparable.
- **Cross-encoder reranking** remains available as an optional second-stage ranking strategy.

The benchmark showed that a more complex neural pipeline was not automatically better: on the current manually annotated evaluation set, **Hybrid RRF without cross-encoder reranking produced the highest MRR**, while BM25 produced the highest Recall@5.

This makes the default retrieval behavior **benchmark-driven rather than complexity-driven**.

The system supports:

- PDF ingestion
- Page-aware text extraction
- Overlapping text chunking
- L2-normalized Sentence Transformer embeddings
- Dense semantic retrieval with FAISS
- BM25 lexical retrieval
- Hybrid dense + lexical retrieval
- Reciprocal Rank Fusion
- Optional cross-encoder reranking
- Five retrieval strategies
- Three candidate-pool modes
- Citation-aware retrieval
- Question answering over indexed documents
- Multi-document comparison
- Document + page evidence provenance
- Retrieval benchmarking
- Candidate-pool ablation experiments
- Latency evaluation
- Persistent local indexing
- FastAPI REST API
- Swagger/OpenAPI documentation
- Automated testing
- Docker support
- GitHub Actions CI

Uploaded research papers and generated vector indexes are not distributed with the repository.

---

# Architecture

## Document Ingestion

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
Sentence Transformer           BM25 Index
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

After every document ingestion, the BM25 index is rebuilt from the complete locally persisted chunk collection.

---

## Default Query Pipeline

The benchmark-selected default retrieval strategy is:

```text
                         Query
                           |
                +----------+----------+
                |                     |
                v                     v
          Dense Retrieval        BM25 Retrieval
             (FAISS)               (Lexical)
                |                     |
                +----------+----------+
                           |
                           v
                Reciprocal Rank Fusion
                           |
                           v
                     Final Top-K
                           |
             +-------------+-------------+
             |                           |
             v                           v
      Question Answering       Citation-Aware Output
```

The default `hybrid` strategy does **not** apply a cross-encoder after RRF.

This design follows the measured benchmark result in which Hybrid RRF achieved higher MRR than Hybrid + Cross-Encoder reranking while requiring substantially less inference time.

---

## Optional Reranking Pipelines

Cross-encoder reranking remains available through dedicated strategies.

### Dense + Cross-Encoder

```text
Query
  |
  v
Dense FAISS Retrieval
  |
  v
Candidate Pool
  |
  v
Cross-Encoder
  |
  v
Final Top-K
```

### Hybrid + Cross-Encoder

```text
                 Query
                   |
          +--------+--------+
          |                 |
          v                 v
       Dense              BM25
          |                 |
          +--------+--------+
                   |
                   v
                  RRF
                   |
                   v
          Cross-Encoder
                   |
                   v
              Final Top-K
```

The current cross-encoder is:

```text
cross-encoder/ms-marco-MiniLM-L6-v2
```

---

# Retrieval Strategies

The API exposes five retrieval strategies.

| Strategy | Pipeline | Reranker |
|---|---|---|
| `dense` | FAISS dense retrieval | No |
| `bm25` | BM25 lexical retrieval | No |
| `hybrid` | Dense + BM25 + RRF | No |
| `dense_reranked` | Dense → Cross-Encoder | Yes |
| `hybrid_reranked` | Dense + BM25 → RRF → Cross-Encoder | Yes |

The default strategy is:

```text
hybrid
```

This strategy was selected because it achieved the **highest MRR** in the current five-way benchmark.

---

# Retrieval Modes

Retrieval modes control candidate-pool size independently from retrieval strategy.

| Mode | Candidate Pool | Intended Use |
|---|---:|---|
| `fast` | 5 | Lower candidate-processing cost |
| `balanced` | 10 | Intermediate configuration |
| `quality` | 20 | Larger retrieval candidate pool |

The default mode is:

```text
quality
```

For non-reranked strategies, candidate-pool size controls how many initial retrieval candidates are generated before final top-k truncation.

For reranked strategies, it also controls how many candidates are passed to the cross-encoder.

Example:

```json
{
  "query": "What methods were used to forecast migraine?",
  "top_k": 5,
  "retrieval_mode": "quality",
  "retrieval_strategy": "hybrid"
}
```

---

# Retrieval Components

## Dense Semantic Retrieval

Document chunks and queries are encoded using a Sentence Transformer.

Embeddings are generated with:

```python
normalize_embeddings=True
```

The vector index uses:

```text
FAISS IndexFlatIP
```

Since both document and query embeddings are L2-normalized, inner-product ranking is equivalent to cosine-similarity ranking.

This provides the semantic retrieval component of the system.

---

## BM25 Lexical Retrieval

The project includes a local BM25 implementation operating over the same document chunks stored by the dense vector index.

BM25 scores each chunk using:

- Query-term frequency
- Document frequency
- Inverse document frequency
- Chunk length normalization
- Configurable `k1`
- Configurable `b`

Default parameters:

```text
k1 = 1.5
b = 0.75
```

BM25 was particularly effective on the current scientific-paper evaluation set, achieving the highest measured Recall@5.

---

## Reciprocal Rank Fusion

Dense and BM25 scores exist on incompatible numeric scales.

Rather than directly adding or normalizing those scores, Hybrid retrieval combines their **rank positions** using Reciprocal Rank Fusion.

For a document ranked at position `r`:

```text
RRF contribution = 1 / (k + r)
```

The implementation uses:

```text
k = 60
```

Evidence appearing highly in both dense and lexical rankings therefore receives a stronger combined score.

The hybrid result preserves:

```text
rrf_score
dense_score
bm25_score
```

so the contribution of each retrieval component remains inspectable.

---

## Cross-Encoder Reranking

Cross-encoder reranking jointly evaluates each:

```text
(query, candidate passage)
```

pair.

Unlike dense retrieval, the query and candidate are processed together, enabling more detailed relevance scoring.

The project keeps reranking optional because benchmark results showed that additional neural inference did not improve every retrieval configuration.

---

# PDF Ingestion

PDF files can be uploaded through the API.

The ingestion pipeline:

1. Stores the uploaded PDF locally
2. Extracts text page by page
3. Splits each page into overlapping chunks
4. Generates L2-normalized semantic embeddings
5. Adds embeddings and metadata to the persistent FAISS index
6. Rebuilds the BM25 lexical index from the persisted chunk collection

Stored metadata includes:

- Document name
- Page number
- Chunk ID
- Extracted text

---

# Search

The `/search` endpoint supports all five retrieval strategies.

Example:

```json
{
  "query": "What methods were used to forecast migraine?",
  "top_k": 5,
  "retrieval_mode": "quality",
  "retrieval_strategy": "hybrid"
}
```

Example Hybrid result:

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

For a `dense_reranked` result, the response may instead contain:

```json
{
  "chunk_id": "paper.pdf:p4:c2",
  "document": "paper.pdf",
  "page": 4,
  "text": "...",
  "score": 0.712,
  "rrf_score": null,
  "dense_score": null,
  "bm25_score": null,
  "reranker_score": 4.18
}
```

### Score Semantics

- `score`
  - raw first-stage score for pure dense or BM25 retrieval
- `rrf_score`
  - ranking score produced by Reciprocal Rank Fusion
- `dense_score`
  - original dense component score preserved by Hybrid retrieval
- `bm25_score`
  - original lexical component score preserved by Hybrid retrieval
- `reranker_score`
  - cross-encoder relevance score for reranked strategies

This avoids treating scores from different retrieval algorithms as though they were directly interchangeable.

---

# Question Answering

The `/ask` endpoint combines retrieval with answer generation.

Example:

```json
{
  "question": "What limitations did the authors report?",
  "top_k": 5,
  "retrieval_mode": "quality",
  "retrieval_strategy": "hybrid"
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
      "score": null,
      "rrf_score": 0.0321,
      "dense_score": 0.68,
      "bm25_score": 4.07,
      "reranker_score": null
    }
  ]
}
```

When no external LLM is configured, the system remains usable as a retrieval-oriented research assistant based on locally indexed evidence.

---

# Citations and Evidence Provenance

Every retrieved chunk preserves source metadata.

Evidence may include:

- Document
- Page
- Chunk ID
- Retrieved text
- Dense score
- BM25 score
- RRF score
- Cross-encoder score where applicable

Human-readable provenance can be represented as:

```text
[paper.pdf, p. 4]
```

This allows evidence to be traced back to its original PDF.

---

# Multi-Document Comparison

The `/compare` endpoint evaluates the same research question across multiple indexed documents.

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

The comparison pipeline:

1. Performs Hybrid Dense + BM25 retrieval
2. Combines results with RRF
3. Separates candidates by requested document
4. Applies cross-encoder reranking independently within each document
5. Returns the strongest evidence for each paper
6. Generates an extractive comparison summary
7. Identifies frequently shared terms

Cross-encoder reranking remains useful here as a dedicated per-document evidence-ranking stage.

Potential use cases include:

- Literature review
- Methodology comparison
- Dataset comparison
- Limitation analysis
- Model performance comparison
- Technical-document comparison

---

# Evaluation

## Evaluation Ground Truth

The benchmark contains:

```text
12 manually curated research queries
```

Relevance is represented using:

```text
(document, page)
```

rather than page number alone.

Example:

```json
{
  "query": "What methods were used to forecast migraine?",
  "relevant_locations": [
    {
      "document": "Forecasting migraine with time-series.pdf",
      "page": 4
    },
    {
      "document": "Forecasting migraine with time-series.pdf",
      "page": 5
    }
  ]
}
```

This prevents a page with the same numeric page number from a different indexed PDF from being counted as relevant accidentally.

---

## Metrics

### Recall@k

Recall@k measures the fraction of annotated relevant locations retrieved within the first `k` results.

The benchmark reports:

```text
Recall@3
Recall@5
```

Higher values indicate greater evidence coverage near the top of the ranking.

### Mean Reciprocal Rank

Reciprocal Rank for one query is:

```text
RR = 1 / rank_of_first_relevant_result
```

MRR averages reciprocal rank across all evaluation queries.

Higher MRR means that the first relevant result tends to appear earlier.

---

# Five-Way Retrieval Benchmark

The main benchmark compares:

1. BM25
2. Dense FAISS retrieval
3. Hybrid Dense + BM25 + RRF
4. Dense + Cross-Encoder
5. Hybrid RRF + Cross-Encoder

Evaluation configuration:

```text
Queries: 12
Candidate pool: 20
Evaluation unit: document + page
Metrics: Recall@3, Recall@5, MRR
```

## Results

| Retrieval Strategy | Recall@3 | Recall@5 | MRR | Avg. Latency |
|---|---:|---:|---:|---:|
| **BM25** | **0.8333** | **0.9583** | 0.7500 | **~2.9 ms** |
| Dense FAISS | 0.7083 | 0.7917 | 0.5972 | ~21.6 ms |
| **Hybrid RRF** | 0.7917 | 0.8750 | **0.8055** | ~15.9 ms |
| Dense + Cross-Encoder | 0.7083 | 0.8750 | 0.7361 | ~1210 ms |
| Hybrid + Cross-Encoder | 0.7917 | 0.8750 | 0.7361 | ~1 s |

Latency values are local measurements and may vary between benchmark runs and hardware environments.

---

## Benchmark Findings

### Hybrid vs Dense

Hybrid RRF improved MRR from:

```text
0.5972 -> 0.8055
```

which corresponds to approximately:

```text
+34.9%
```

relative MRR improvement.

Recall@5 improved from:

```text
0.7917 -> 0.8750
```

or approximately:

```text
+10.5%
```

relative improvement.

### BM25

BM25 achieved:

```text
Recall@5 = 0.9583
```

the highest measured Recall@5 in the current evaluation.

It also produced the lowest retrieval latency.

This result is plausible for the current corpus because many evaluation questions contain scientific terminology that overlaps directly with the source paper.

### Hybrid RRF

Hybrid achieved:

```text
MRR = 0.8055
```

the highest MRR in the benchmark.

The result suggests that combining lexical and semantic rankings improved the placement of the first relevant result.

### Cross-Encoder Reranking

Cross-encoder reranking improved the original dense baseline:

```text
Dense MRR
0.5972

Dense + CrossEncoder MRR
0.7361
```

However, applying the same cross-encoder after Hybrid RRF reduced MRR:

```text
Hybrid RRF
0.8055

Hybrid + CrossEncoder
0.7361
```

without improving Recall@3 or Recall@5 in the current evaluation.

For this reason, the system does **not** assume that additional neural reranking always improves retrieval.

The default strategy is therefore:

```text
hybrid
```

rather than:

```text
hybrid_reranked
```

This is an intentional benchmark-driven engineering decision.

---

# Candidate-Pool Ablation

A separate experiment evaluated the number of candidates used by the reranking pipeline.

Each configuration was run three times after model warm-up.

| Candidate Pool | Recall@5 | MRR | Median Total Latency |
|---:|---:|---:|---:|
| 5 | 0.792 | **0.788** | **256 ms** |
| 10 | 0.792 | 0.736 | 502 ms |
| 20 | **0.875** | 0.736 | 1245 ms |
| 40 | 0.833 | 0.715 | 2663 ms |

The ablation showed:

- Smaller candidate pools can substantially reduce reranking latency.
- `candidate_k = 20` produced the highest Recall@5 for the evaluated reranking configuration.
- Increasing the pool from 20 to 40 increased latency without improving retrieval quality.
- `candidate_k = 5` represented a strong low-latency reranking operating point.

These findings motivated the API retrieval modes.

---

# Evaluation Files

Ground-truth evaluation set:

```text
evaluation/retrieval_eval.json
```

Main five-way benchmark:

```text
evaluation/retrieval_benchmark.json
```

Candidate-pool ablation:

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

Run the main benchmark:

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

Example response:

```json
{
  "status": "ok",
  "indexed_chunks": 144
}
```

---

## Upload PDF

```http
POST /documents/upload
```

Uploads and indexes a PDF.

Example:

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

Example:

```json
{
  "query": "What data sources were used in the study?",
  "top_k": 5,
  "retrieval_mode": "quality",
  "retrieval_strategy": "hybrid"
}
```

Supported modes:

```text
fast
balanced
quality
```

Supported strategies:

```text
dense
bm25
hybrid
dense_reranked
hybrid_reranked
```

---

## Ask

```http
POST /ask
```

Example:

```json
{
  "question": "What limitations were reported?",
  "top_k": 5,
  "retrieval_mode": "quality",
  "retrieval_strategy": "hybrid"
}
```

---

## Compare

```http
POST /compare
```

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

# Interactive API Documentation

FastAPI exposes interactive Swagger documentation at:

```text
http://127.0.0.1:8000/docs
```

The interface allows users to:

- Upload PDFs
- Run searches
- Select retrieval mode
- Select retrieval strategy
- Ask document-grounded questions
- Compare documents
- Inspect request schemas
- Inspect response schemas
- Execute requests interactively

OpenAPI JSON:

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

## Retrieval and Machine Learning

- Sentence Transformers
- FAISS
- BM25
- Reciprocal Rank Fusion
- Cross-Encoder reranking
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
- Persistent local indexes
- Environment-based configuration
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
|   |   |-- bm25.py
|   |   |-- chunking.py
|   |   |-- config.py
|   |   |-- embeddings.py
|   |   |-- generation.py
|   |   |-- hybrid.py
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
|   |-- test_bm25.py
|   |-- test_chunking.py
|   |-- test_hybrid.py
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

## 1. Clone

```bash
git clone https://github.com/vasilischr01/Research-rag-assistant.git
cd Research-rag-assistant
```

## 2. Create Virtual Environment

### Windows

```powershell
py -3.11 -m venv .venv
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

```bash
pytest -q
```

Current development result:

```text
20 passed
```

The test suite covers:

- API health checks
- Upload validation
- Search endpoint behavior
- Retrieval-mode routing
- Fast candidate-pool configuration
- Quality candidate-pool configuration
- Invalid retrieval-mode validation
- BM25 tokenization
- BM25 relevance ranking
- BM25 top-k behavior
- Empty BM25 behavior
- Reciprocal Rank Fusion
- RRF duplicate handling
- RRF top-k behavior
- Empty RRF behavior
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
5. Installs dependencies
6. Executes the pytest suite

Workflow:

```text
.github/workflows/ci.yml
```

---

# Docker

Build the container:

```bash
docker build -t research-rag-assistant .
```

The Docker image provides a reproducible environment for running the API and retrieval stack.

---

# Data and Privacy

Uploaded PDF documents are not included in the repository.

Generated vector indexes are also excluded from Git.

Ignored directories include:

```text
data/uploads/
data/indexes/
```

Only `.gitkeep` files are tracked to preserve the directory structure.

The project can therefore be used with private documents without publishing those PDFs or generated indexes through the repository.

---

# Configuration

Environment-specific settings can be stored in:

```text
.env
```

An example is provided:

```text
.env.example
```

`.env` files and credentials are excluded from Git.

---

# Local-First Design

Core retrieval functionality runs locally:

- PDF extraction
- Chunking
- Embedding generation
- FAISS indexing
- BM25 indexing
- Dense retrieval
- Lexical retrieval
- RRF fusion
- Optional cross-encoder reranking
- Multi-document comparison
- Evaluation
- Benchmark generation

The core retrieval architecture therefore does not require a paid hosted LLM service.

A local or external language model can be integrated as an optional answer-generation layer.

---

# Engineering Decisions

## Benchmark-Selected Default

The default retrieval pipeline was selected from measured results rather than from architectural complexity.

The benchmark showed:

```text
Hybrid RRF MRR = 0.8055

Hybrid + CrossEncoder MRR = 0.7361
```

Therefore:

```text
hybrid
```

is the default strategy.

The cross-encoder remains available when explicitly requested.

---

## Hybrid Retrieval

Dense and lexical retrieval capture different relevance signals.

Dense retrieval helps with:

- Semantic similarity
- Paraphrases
- Conceptual matching

BM25 helps with:

- Exact terminology
- Technical vocabulary
- Rare terms
- Acronyms
- Literal phrase overlap

RRF combines those rankings without requiring score calibration between BM25 and vector similarity.

---

## Explicit Score Provenance

Hybrid retrieval does not expose an ambiguous generic ranking score.

Instead it preserves:

```text
rrf_score
dense_score
bm25_score
```

This makes retrieval decisions easier to inspect and avoids conflating scores from different ranking algorithms.

---

## Persistent Vector Index

FAISS embeddings and chunk metadata are persisted locally between application restarts.

The BM25 index is reconstructed from the persisted chunk collection.

---

## Cosine-Equivalent Dense Search

Embedding generation uses L2 normalization.

FAISS `IndexFlatIP` therefore performs ranking equivalent to cosine similarity for the stored vectors.

---

## Document + Page Evaluation

Ground-truth relevance is represented as:

```text
(document, page)
```

rather than page number alone.

This prevents false-positive evaluation matches when several PDFs contain the same page number.

---

# Limitations

The current implementation has several limitations:

- The benchmark contains **12 manually curated queries**, so results should be interpreted as development-level evidence rather than a large-scale IR benchmark.
- The evaluation corpus is currently limited and should be expanded across more documents and domains.
- Findings such as BM25's strong performance may depend on the terminology and structure of the current scientific corpus.
- Relevance annotations are page-level rather than chunk-level.
- Retrieval evaluation does not yet measure full answer correctness.
- Multi-document summaries are extractive when no language model is configured.
- PDF parsing quality depends on source-document structure.
- Image-only PDFs may require OCR.
- The local BM25 implementation rebuilds its index after document ingestion rather than using a distributed persistent search engine.
- The FAISS configuration targets local development rather than distributed retrieval at production scale.
- Cross-encoder latency depends heavily on hardware.
- Benchmark latency values are local measurements and should not be interpreted as universal production latency.

---

# Future Improvements

Potential extensions include:

- Larger multi-document evaluation set
- 30–50+ manually annotated benchmark queries
- Chunk-level relevance judgments
- More diverse scientific and technical corpora
- Statistical confidence intervals for benchmark metrics
- Repeated latency measurements for the five-way benchmark
- Query rewriting
- Metadata filtering
- Document-level ranking
- Answer-quality evaluation
- Faithfulness evaluation
- Groundedness evaluation
- Automated citation verification
- Local LLM integration
- Optional external LLM integration
- Persistent database-backed metadata
- Web frontend
- Streaming responses
- Docker Compose
- GPU acceleration
- Prometheus metrics
- Retrieval latency monitoring
- Load testing
- Automated benchmark regression testing
- Larger-scale vector-store evaluation

---

# Why This Project

This project explores practical RAG engineering as an **information-retrieval system**, rather than treating RAG as only an LLM prompting task.

The implementation focuses on:

- Dense retrieval
- Lexical retrieval
- Hybrid retrieval
- Reciprocal Rank Fusion
- Neural reranking
- Evidence provenance
- Quantitative evaluation
- Ablation experiments
- Benchmark-driven architecture decisions
- Quality-latency trade-offs
- Explainable score provenance
- Configurable retrieval behavior
- API design
- Automated testing
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