# RAG Architecture — LegacyLens

## Overview

LegacyLens is a RAG-powered system for querying the LAPACK Fortran codebase (~1,850 files) using natural language. The pipeline ingests Fortran source files, chunks them by subroutine boundary, embeds them with a code-specialized model, stores vectors in Pinecone, and retrieves relevant chunks at query time to ground LLM-generated answers in actual source code.

**Stack:** Custom Python pipeline (no LangChain/LlamaIndex) — Voyage Code 3 embeddings, Pinecone vector DB, Claude Haiku 4.5 via OpenRouter, FastAPI + Typer CLI.

The custom pipeline was chosen over framework alternatives because LAPACK requires a custom Fortran chunker, the three SDKs compose cleanly, and the total RAG pipeline is ~200 lines — comparable to framework setup code alone. This also keeps the MCP upgrade path unconstrained.

## Vector Database Selection

**Pinecone** (managed, serverless) was selected for:

- **Zero ops:** No provisioning, scaling, or backup management — critical for a 1-week sprint.
- **Metadata filtering:** Native support for filtering on structured fields (unit_name, precision, routine_role, calls, parameters) during vector search. This powers the entity-filtered retrieval tier.
- **Free tier:** Sufficient for LAPACK's ~1,850 vectors at 1024 dimensions.

Each vector stores the full chunk metadata alongside its embedding, enabling filtered queries like "vectors where `calls` contains XERBLA and `precision` is double" without a separate metadata store.

## Embedding Strategy

**Voyage Code 3** (1024 dimensions, 16K token context window) embeds both documents and queries.

- **Code-optimized:** 10-20% higher retrieval quality on code search benchmarks vs. general-purpose models (OpenAI text-embedding-3-small/large).
- **16K context:** Accommodates 97% of LAPACK routines as single chunks without truncation. OpenAI's 8K limit and Cohere's 512-token limit would force aggressive splitting.
- **Asymmetric search:** Voyage uses `input_type="document"` at ingestion and `input_type="query"` at retrieval, adjusting the embedding space for query-document similarity.
- **Reranking:** Voyage's `rerank-2` cross-encoder model is used post-retrieval to reorder candidates. Cross-encoders process query and document jointly, catching relevance signals that cosine similarity misses (e.g., distinguishing "eigenvalues" from "singular values").

### Chunk Enrichment

Raw Fortran code uses opaque names (DGESV, DGETRF) with no lexical overlap to natural language queries. Every chunk is prepended with a natural-language header before embedding:

```
# DGESV
- Double Precision Fortran Subroutine
# Purpose: DGESV computes the solution to a real system of linear equations A * X = B.
# File: dgesv.f | Lines: 1-178 | Calls: DGETRF, DGETRS, XERBLA

      SUBROUTINE DGESV( N, NRHS, A, LDA, IPIV, B, LDB, INFO )
      ...
```

This bridges the vocabulary gap at ingestion time rather than at query time, avoiding the latency cost of query expansion or HyDE approaches.

## Chunking Approach

LAPACK follows a one-subroutine-per-file convention, making each file a natural chunk boundary.

### Extraction Method

Regex-based extraction (not AST parsing) processes the entire codebase in under 2 seconds. LAPACK's 30+ year formatting consistency makes regex reliable. Tree-sitter-fortran would add native dependencies for marginal benefit.

### Metadata Extracted Per Chunk

| Field | Source | Purpose |
|-------|--------|---------|
| `unit_name` | SUBROUTINE/FUNCTION declaration | Filtered lookups, display |
| `unit_type` | Which regex matched | Distinguishes subroutines from functions |
| `parameters` | Parenthesized list after name | Interface display, parameter-based retrieval |
| `calls` | All `CALL X` statements | Dependency mapping |
| `purpose` | LAPACK comment header | Natural-language anchor for embedding |
| `precision` | First letter: S/D/C/Z | Precision-variant deduplication |
| `category` | 2nd-3rd letters (GE, SY, PO...) | Matrix type classification |
| `routine_role` | Rule-based (driver/computational/auxiliary/blas) | Driver boosting in retrieval |
| `file_path`, `start_line`, `end_line` | File system | Source code resolution |

### Splitting Long Files

~3% of files exceed the 15K token embedding limit. These are split using preamble preservation: the comment header and declarations are prepended to each sub-chunk, with 15-line overlap at boundaries. Split points target LAPACK's section comment delimiters (`* ==========`).

## Retrieval Pipeline

The retrieval pipeline uses a multi-tier architecture to address three LAPACK-specific failure modes:

1. **Vocabulary gap** — "solve linear equations" has no lexical overlap with `SUBROUTINE DGESV`
2. **Precision-variant flooding** — SGESV/DGESV/CGESV/ZGESV consume all top-5 slots
3. **Driver burial** — internal helpers score higher than user-facing routines on raw cosine similarity

### Three Retrieval Paths

| Path | Trigger | Pinecone Queries |
|------|---------|-----------------|
| **Pin-unit** | CLI `deps`/`document` commands | 2 (filtered + semantic) |
| **Entity-aware** | Uppercase Fortran identifiers in query | 4-7 entity + 2 semantic |
| **Semantic-only** | No identifiers detected | 2 (driver-filtered + unfiltered) |

### Entity Detection

A regex scanner identifies Fortran identifiers (DGETRF, IPIV, XERBLA) in queries. These are classified as routines or parameters using LAPACK naming rules, then used to issue metadata-filtered Pinecone queries — bypassing the vocabulary gap entirely.

### Two-Pass Semantic Search

1. **Driver-filtered pass:** Ensures user-facing routines appear in candidates despite scoring lower on cosine similarity.
2. **Unfiltered pass:** Over-fetches 3x for diversification headroom.

### Precision-Variant Diversification

Results are deduplicated by base routine name (strip S/D/C/Z prefix). When variants collide, the D-prefix (double precision) form is preferred as the canonical reference. A final cross-tier diversification pass catches duplicates spanning the entity and semantic tiers.

### Processing Pipeline

```
Entity Results → Dedup → Sort by role → Rerank (Voyage rerank-2) → Diversify
                                                                        ↓
Non-Entity Results → Dedup vs entity IDs → Rerank                      ↓
                                              ↓                         ↓
                                    Final Diversification (cross-tier, top_k)
                                              ↓
                                    Score Threshold Filter (0.45, entity bypass)
```

## Source Code Integration

Retrieved chunks carry `file_path`, `start_line`, and `end_line` metadata. A shared `source_reader` module resolves these to actual source files on disk and reads the relevant line ranges.

Source code is integrated at three levels:

1. **LLM Context:** `build_context()` appends the actual Fortran source to each chunk's metadata block, giving the LLM real code to reason about (not just metadata summaries).
2. **CLI Output:** `format_results()` renders syntax-highlighted Fortran snippets with Rich when `--show-code` is enabled (default).
3. **API Response:** The `/query` endpoint returns `start_line`, `end_line`, and `snippet` fields on each `ChunkResult`.

Path resolution tries `base/file_path` then `base/SRC/file_path` (matching LAPACK's directory structure). All source reading degrades gracefully — returning empty strings on failure, never raising exceptions.

## Failure Modes

| Failure | Impact | Mitigation |
|---------|--------|------------|
| Voyage API down | No embedding/reranking | Reranking falls back to cosine-similarity ordering; embedding failure is fatal (no local fallback) |
| Pinecone unreachable | No retrieval | Fatal — no local vector store fallback |
| OpenRouter/LLM down | No answer generation | CLI `--no-answer` mode still shows retrieval results |
| Source files missing | No code snippets | Graceful degradation: LLM context omits source, CLI hides snippet, API returns empty string |
| Oversized source | Token budget exceeded | Snippets truncated at 8K chars with `[truncated]` marker |
| Low-confidence results | Noise in output | Score threshold (0.45) filters weak results; entity matches bypass threshold |
| Precision flooding | Redundant results | Diversification collapses S/D/C/Z variants to one representative |

## Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| Ingestion speed | ~1,850 files in < 2 min | Regex chunking + batch embedding via Voyage |
| Query latency (semantic) | ~400-600ms | 1 embed + 2 Pinecone queries + 1-2 rerank calls |
| Query latency (entity-aware) | ~600-900ms | Additional 4-5 Pinecone queries for entity tier |
| Answer generation | ~2-4s | Claude Haiku 4.5 via OpenRouter |
| Pinecone queries per request | 2-7 | Varies by path (see retrieval pipeline) |
| Vector dimensions | 1024 | Voyage Code 3 |
| Index size | ~1,850 vectors | One per LAPACK routine (+ split chunks) |
| Cost per query | ~$0.004 | Embed ($0.0001) + Rerank ($0.002) + LLM ($0.002) |
