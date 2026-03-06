# RAG Architecture — LegacyLens

## Overview

LegacyLens is a RAG-powered system for querying legacy codebases using natural language. The pipeline supports multiple languages through a pluggable chunker architecture — currently Fortran (LAPACK, ~1,850 files) and Python (any project). It ingests source files, chunks them at language-appropriate boundaries, embeds them with a code-specialized model, stores vectors in Pinecone with per-source namespace isolation, and retrieves relevant chunks at query time using language-aware strategies to ground LLM-generated answers in actual source code.

**Stack:** Custom Python pipeline (no LangChain/LlamaIndex) — Voyage Code 3 embeddings, Pinecone vector DB, Claude Haiku 4.5 via OpenRouter, FastAPI + Typer CLI.

The custom pipeline was chosen over framework alternatives because legacy codebases require custom chunkers per language, the three SDKs compose cleanly, and the total RAG pipeline is ~200 lines — comparable to framework setup code alone. This also keeps the MCP upgrade path unconstrained.

## Source Management

Each ingested codebase is tracked as a **source** with its own Pinecone namespace, ensuring vector isolation between projects. A local JSON registry at `~/.legacylens/sources.json` stores:

| Field | Purpose |
|-------|---------|
| `name` | User-provided or derived from folder basename |
| `path` | Absolute path to ingested directory |
| `namespace` | Pinecone namespace (slugified name) |
| `languages` | Detected languages (e.g., `["python"]` or `["fortran"]`) |
| `extensions` | Extension → file count (e.g., `{".py": 35}`) |
| `chunk_count` | Total chunks ingested |
| `ingested_at` | ISO timestamp |

**Pre-scan at ingest time:** Before chunking, the directory is walked to count files by extension and match against registered chunkers. This determines languages up front, so at query time the retrieval layer already knows which strategy to use without re-scanning.

**Namespace isolation:** Each source's vectors live in a separate Pinecone namespace. Querying source A returns zero results from source B, even though they share the same index.

## Vector Database Selection

**Pinecone** (managed, serverless) was selected for:

- **Zero ops:** No provisioning, scaling, or backup management — critical for a 2-day sprint.
- **Metadata filtering:** Native support for filtering on structured fields (unit_name, precision, routine_role, calls, parameters, uses) during vector search. This powers the entity-filtered retrieval tier.
- **Namespace support:** Built-in namespace isolation enables multi-source management without separate indexes.
- **Free tier:** Sufficient for multiple codebases at 1024 dimensions.

Each vector stores the full chunk metadata alongside its embedding, enabling filtered queries like "vectors where `calls` contains XERBLA and `precision` is double" without a separate metadata store.

## Embedding Strategy

**Voyage Code 3** (1024 dimensions, 16K token context window) embeds both documents and queries.

- **Code-optimized:** 10-20% higher retrieval quality on code search benchmarks vs. general-purpose models (OpenAI text-embedding-3-small/large).
- **16K context:** Accommodates 97% of LAPACK routines and most Python units as single chunks without truncation. OpenAI's 8K limit and Cohere's 512-token limit would force aggressive splitting.
- **Asymmetric search:** Voyage uses `input_type="document"` at ingestion and `input_type="query"` at retrieval, adjusting the embedding space for query-document similarity.
- **Reranking:** Voyage's `rerank-2` cross-encoder model is used post-retrieval to reorder candidates. Cross-encoders process query and document jointly, catching relevance signals that cosine similarity misses (e.g., distinguishing "eigenvalues" from "singular values").

### Chunk Enrichment

Every chunk is prepended with a natural-language header before embedding. The format is language-aware:

**Fortran example** (opaque names like DGESV have no lexical overlap with natural language queries):
```
# DGESV
- Double Precision Fortran Subroutine
# Purpose: DGESV computes the solution to a real system of linear equations A * X = B.
# File: dgesv.f | Lines: 1-178 | Calls: DGETRF, DGETRS, XERBLA

      SUBROUTINE DGESV( N, NRHS, A, LDA, IPIV, B, LDB, INFO )
      ...
```

**Python example** (names are more descriptive but docstrings bridge the intent gap):
```
# BaseModel
- Python Class
# Purpose: Base class for all Pydantic models. Manages validation and serialization.
# File: main.py | Lines: 45-320 | Calls: model_validate, __init_subclass__

class BaseModel(metaclass=ModelMetaclass):
    ...
```

This bridges the vocabulary gap at ingestion time rather than at query time, avoiding the latency cost of query expansion or HyDE approaches.

## Chunking Approach

LegacyLens uses a pluggable chunker architecture (`BaseChunker` ABC → `ChunkerRegistry`) that dispatches files to the correct chunker by extension. The registry's `chunk_directory()` method walks a directory and routes each file automatically.

### Fortran Chunker

LAPACK follows a one-subroutine-per-file convention, making each file a natural chunk boundary.

**Extraction method:** Regex-based (not AST parsing) — processes ~1,850 files in under 2 seconds. LAPACK's 30+ year formatting consistency makes regex reliable.

**Metadata extracted:**

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

**Splitting long files:** ~3% of files exceed the 15K token embedding limit. These are split using preamble preservation: the comment header and declarations are prepended to each sub-chunk, with 15-line overlap at boundaries. Split points target LAPACK's section comment delimiters (`* ==========`).

### Python Chunker

Uses the `ast` module (stdlib) for reliable parsing — handles indentation, decorators, async, type hints, and all modern Python syntax.

**Chunking strategy:**

| Unit | Chunk behavior |
|------|---------------|
| Top-level `def`/`async def` | One chunk each (decorators included) |
| Top-level `class` | One chunk (nested classes/inner functions stay with parent) |
| Module preamble | Imports + module docstring + constants → separate chunk |
| Large classes (>60K chars) | Split at method boundaries; preamble chunk + method group chunks |
| Script-style code | Lines outside any function/class → single `module_code` chunk |
| SyntaxError files | Whole file as one `module` chunk with regex-based name extraction |

**Metadata extracted:**

| Field | Source | Purpose |
|-------|--------|---------|
| `unit_name` | `node.name` or filename stem | Filtered lookups, display |
| `unit_type` | `function`, `async_function`, `class`, `method`, `async_method`, `module`, `module_code` | Type-aware retrieval |
| `parameters` | `ast.arguments` — names with type annotations | Interface display |
| `calls` | `ast.Call` nodes → function/method names | Dependency mapping |
| `uses` | `import X` / `from X import Y` → module names | Import tracking |
| `purpose` | `ast.get_docstring()` — first 1-2 sentences | Natural-language anchor |

**Performance:** Pydantic's 103 Python files (35 `.py` in the main package, 103 total including subpackages) produce 1,052 chunks in 0.23 seconds.

### Shared Metadata Schema

Both chunkers produce `Chunk` objects with `ChunkMetadata` that shares a common schema. Fortran-only fields (`precision`, `category`, `routine_role`) are `None` for Python chunks. The `language` field (`"fortran"` or `"python"`) drives downstream behavior in retrieval and rendering.

## Retrieval Pipeline

The retrieval pipeline uses a multi-tier architecture with **language-aware strategy selection**. The source's pre-scanned language metadata (stored at ingest time) determines which entity extraction and diversification strategy to use — no guessing at query time.

### Language-Specific Failure Modes

**Fortran (LAPACK):**
1. **Vocabulary gap** — "solve linear equations" has no lexical overlap with `SUBROUTINE DGESV`
2. **Precision-variant flooding** — SGESV/DGESV/CGESV/ZGESV consume all top-5 slots
3. **Driver burial** — internal helpers score higher than user-facing routines on raw cosine similarity

**Python:**
1. **Vocabulary gap** (milder) — function/class names are more descriptive, but abbreviations and domain jargon still exist
2. **No precision-variant flooding** — Python has no S/D/C/Z naming convention; diversification is skipped

### Three Retrieval Paths

| Path | Trigger | Pinecone Queries |
|------|---------|-----------------|
| **Pin-unit** | CLI `deps`/`document` commands | 2 (filtered + semantic) |
| **Entity-aware** | Language-specific identifiers in query | 2-7 entity + 1-2 semantic |
| **Semantic-only** | No identifiers detected | 1-2 (driver-filtered + unfiltered) |

### Entity Detection

Entity detection is dispatched by language:

**Fortran:** A regex scanner identifies uppercase Fortran identifiers (DGETRF, IPIV, XERBLA). These are classified as routines or parameters using LAPACK naming rules, then used to issue metadata-filtered Pinecone queries with dual precision passes (D-prefix + all-precision).

**Python:** A regex scanner identifies:
- `snake_case` identifiers → likely functions/variables → `unit_name` and `calls` filters
- `CamelCase` identifiers → likely classes → `unit_name` filter
- `dotted.module.paths` → likely imports → `uses` filter

Common English words are filtered out in both languages.

### Two-Pass Semantic Search

For Fortran sources:
1. **Driver-filtered pass:** Ensures user-facing routines appear in candidates despite scoring lower on cosine similarity.
2. **Unfiltered pass:** Over-fetches 3x for diversification headroom.

For Python sources:
1. **Unfiltered pass only:** No driver/role hierarchy in Python. Over-fetches 3x for reranking headroom.

### Precision-Variant Diversification

Results are deduplicated by base routine name — but **only for Fortran chunks**. The `_base_routine_name()` function strips S/D/C/Z prefixes for Fortran and returns names as-is for Python. The `_diversify_results()` function reads the `language` field from each chunk's metadata and only applies D-prefix preference for Fortran results. Python chunks pass through undeduplicated.

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

Diversification is language-aware at every stage. Fortran chunks are deduplicated by SDCZ prefix; Python chunks are never collapsed.

## Source Code Integration

Retrieved chunks carry `file_path`, `start_line`, and `end_line` metadata. A shared `source_reader` module resolves these to actual source files on disk and reads the relevant line ranges.

Source code is integrated at three levels:

1. **LLM Context:** `build_context()` appends actual source code to each chunk's metadata block in a language-tagged fenced code block (`` ```fortran `` or `` ```python ``), giving the LLM real code to reason about.
2. **CLI Output:** `format_results()` renders syntax-highlighted snippets with Rich using the chunk's `language` metadata for lexer selection.
3. **API Response:** The `/query` endpoint returns `start_line`, `end_line`, and `snippet` fields on each `ChunkResult`.

Path resolution tries `base/file_path` then `base/SRC/file_path` (the SRC fallback matches LAPACK's directory structure; it's harmless for Python projects). All source reading degrades gracefully — returning empty strings on failure, never raising exceptions.

## Failure Modes

| Failure | Impact | Mitigation |
|---------|--------|------------|
| Voyage API down | No embedding/reranking | Reranking falls back to cosine-similarity ordering; embedding failure is fatal (no local fallback) |
| Pinecone unreachable | No retrieval | Fatal — no local vector store fallback |
| OpenRouter/LLM down | No answer generation | CLI `--no-answer` mode still shows retrieval results |
| Source files missing | No code snippets | Graceful degradation: LLM context omits source, CLI hides snippet, API returns empty string |
| Oversized source | Token budget exceeded | Snippets truncated at 8K chars with `[truncated]` marker |
| Low-confidence results | Noise in output | Score threshold (0.45) filters weak results; entity matches bypass threshold |
| Precision flooding | Redundant Fortran results | Diversification collapses S/D/C/Z variants to one representative (Fortran only) |
| Python SyntaxError | Unparseable file | Fallback: whole file as one chunk with regex-based name extraction |
| Unknown source | `--source` flag with unregistered name | CLI errors with helpful message before any API calls |

## Performance Characteristics

| Metric | Fortran (LAPACK) | Python (Pydantic) | Notes |
|--------|-----------------|-------------------|-------|
| Chunking speed | 1,847 chunks in 1.8s | 1,052 chunks in 0.23s | Regex vs. AST parsing |
| Batch ingest (end-to-end) | ~10-15 min | ~9 min (520s) | Dominated by Voyage Batch API |
| Embedding time | ~8-12 min (batch) | ~8.4 min (batch) | 753K tokens for Pydantic |
| Pinecone upsert | ~12s | ~7.6s | Batches of 100 vectors |
| Query latency (semantic) | ~400-600ms | ~400-600ms | 1 embed + 1-2 Pinecone queries + rerank |
| Query latency (entity-aware) | ~600-900ms | ~500-700ms | Fewer entity queries for Python (no dual-precision) |
| Answer generation | ~2-4s | ~2-4s | Claude Haiku 4.5 via OpenRouter |
| Pinecone queries per request | 2-7 | 2-5 | Python skips driver-filtered pass |
| Vector dimensions | 1024 | 1024 | Voyage Code 3 |
| Index size | ~1,850 vectors | ~1,052 vectors | Per-source namespace |
| Cost per query | ~$0.004 | ~$0.004 | Embed ($0.0001) + Rerank ($0.002) + LLM ($0.002) |
