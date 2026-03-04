# LegacyLens

RAG-powered CLI and API for querying legacy Fortran codebases using natural language. Built to make LAPACK (Linear Algebra PACKage) searchable and explainable without reading thousands of lines of Fortran.

## How It Works

```
Fortran source → Chunking (one routine per chunk) → Embeddings (Voyage Code 3)
  → Vector storage (Pinecone) → Semantic search + Reranking → Answer generation (Claude)
```

Source files are parsed for subroutine/function boundaries, parameters, call graphs, and purpose comments. Chunks are enriched with natural language summaries before embedding.

## Setup

```bash
pip install .
cp .env.example .env   # fill in API keys
```

### Required Environment Variables

| Variable | Purpose |
|---|---|
| `VOYAGE_API_KEY` | Voyage Code 3 embeddings and reranking |
| `PINECONE_API_KEY` | Pinecone vector database |
| `OPENROUTER_API_KEY` | Claude LLM via OpenRouter |

### Optional

| Variable | Default | Purpose |
|---|---|---|
| `PINECONE_INDEX_NAME` | `legacylens` | Pinecone index name |
| `LAPACK_DATA_DIR` | `data/lapack` | Base directory for source file lookups |

## CLI Commands

### Ingestion

```bash
# Real-time ingestion (rate-limited, ~2.5h for full LAPACK)
legacylens ingest data/lapack/

# Batch ingestion via Voyage Batch API (~10-15 min for full LAPACK)
legacylens batch-ingest data/lapack/
legacylens batch-ingest data/lapack/ --no-reset   # keep existing index
```

### Querying

```bash
legacylens query "How does DGESV solve linear systems?" [--top-k 5] [--mode explain] [--no-answer] [--show-code]
legacylens explain "What does DGETRF do?"           # shortcut for --mode explain
legacylens deps DGEMM                                # dependency graph for a routine
legacylens document DGESV                            # generate modern documentation
legacylens logic "How does pivoting work in LU?"     # extract mathematical logic
```

| Flag | Default | Description |
|---|---|---|
| `--top-k` | 5 (10 for deps) | Number of chunks to retrieve |
| `--mode` | `explain` | Response mode: `explain`, `deps`, `docs`, `business_logic` |
| `--no-answer` | off | Show retrieved chunks only, skip LLM |
| `--show-code` | off | Display source code snippets |

### Operations

```bash
legacylens view SRC/dgesv.f              # view source with syntax highlighting
legacylens stats                          # Pinecone index statistics
legacylens serve [--host 0.0.0.0] [--port 8000]  # start FastAPI server (Swagger at /docs)
```

## Ingested Codebase

**Target:** LAPACK — the standard Fortran library for numerical linear algebra.

**Supported file types:** `.f`, `.f90`, `.f95`, `.f03`, `.for`

The Fortran chunker extracts routine boundaries, parameters, CALL/EXTERNAL/USE statements, purpose comments, and precision classification (single, double, complex, double-complex). Large files (>15K tokens) are split at comment-delimited sections. Produces ~1,850 chunks for the full LAPACK + BLAS source tree.

## Tech Stack

| Layer | Tool |
|---|---|
| Embeddings | Voyage Code 3 (1024-dim) |
| Reranking | Voyage `rerank-2` |
| Vector DB | Pinecone (serverless, cosine, AWS us-east-1) |
| LLM | Claude Haiku 4.5 via OpenRouter (Claude Opus 4.6 optional) |
| CLI | Typer |
| API | FastAPI + Uvicorn |
| Terminal UI | Rich (panels, tables, syntax highlighting, progress bars) |
| Testing | pytest + pytest-asyncio |

## API Server

```bash
legacylens serve
```

Endpoints: `GET /health`, `POST /query`, `POST /search`, `GET /stats`. Swagger UI at `/docs`.

## Deployment

Deployed on **Fly.io** (1 shared CPU, 1GB RAM, IAD region). Auto-scales from 0.

```bash
fly deploy
fly secrets set VOYAGE_API_KEY=... PINECONE_API_KEY=... OPENROUTER_API_KEY=...
```

## Testing

```bash
pytest tests/
```

Tests cover chunking against real LAPACK files, API endpoints, retrieval pipeline metrics (precision@k, recall@k, MRR, hit rate), source reading, and LLM generation. Tests that require LAPACK data files skip gracefully if not present.
