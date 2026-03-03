# Chunking Strategy

## The Core Insight

LAPACK follows a one-subroutine-per-file convention. Every `.f` file contains exactly one `SUBROUTINE` or `FUNCTION`, complete with a structured comment header documenting its purpose, parameters, and algorithm. This means our natural chunk boundary is the file itself — no need for sliding windows, recursive splitting, or overlap between chunks.

85% of LAPACK routines fit within the Voyage Code 3 embedding model's 16K token context window as a single chunk. The remaining ~3% of oversized files are split with a preamble-preservation strategy described below.

## Architecture

```
BaseChunker (ABC)                # Pluggable interface for future COBOL support
├── supported_extensions()       # [".f", ".f90", ".f95", ".f03", ".for"]
├── chunk_file(path, content)    # → list[Chunk]
└── chunk_directory(dir)         # Walk + filter + chunk all files

FortranChunker(BaseChunker)      # LAPACK-specific implementation

ChunkerRegistry                  # Auto-selects chunker by file extension
```

Each `Chunk` has three components:
- **`content`** — raw Fortran source (stored for display)
- **`enriched_content`** — natural-language header + source (this is what gets embedded)
- **`metadata`** — structured fields stored alongside the vector in Pinecone

## Why Regex Instead of a Parser

We use regex-based extraction rather than tree-sitter-fortran or fparser2. The rationale:

1. **LAPACK is extremely consistent.** Every file follows the same formatting conventions established over 30+ years. Fixed-form Fortran uses columns 1-6 for labels/continuation, column 7+ for statements. The patterns don't vary.
2. **No external dependencies.** tree-sitter-fortran requires native binaries; fparser2 has weak fixed-form support. Regex needs only the stdlib `re` module.
3. **Speed.** Regex chunking processes the entire LAPACK codebase (~1,850 files) in under 2 seconds.

The regex patterns handle both fixed-form and free-form Fortran:

| Pattern | What it matches |
|---------|----------------|
| `^\s{0,6}\s*SUBROUTINE\s+(\w+)\s*\(([^)]*)\)` | Subroutine name + parameters |
| `^\s{0,6}\s*.*FUNCTION\s+(\w+)\s*\(([^)]*)\)` | Function name + parameters (with optional type prefix) |
| `\bCALL\s+(\w+)` | All CALL targets (dependency graph) |
| `^\s+EXTERNAL\s+(.+)` | External declarations |
| `^\s*USE\s+(\w+)` | Module imports |
| `(?:Purpose\|PURPOSE)\s*[:\n=]` | Start of LAPACK purpose documentation |

The subroutine/function patterns also handle Fortran qualifiers (`RECURSIVE`, `PURE`, `ELEMENTAL`) via optional prefix groups.

## What We Extract Per Chunk

Every chunk carries structured metadata that Pinecone stores alongside the vector, enabling filtered queries at retrieval time:

| Field | How it's extracted | Why it matters |
|-------|-------------------|---------------|
| `unit_name` | Regex on SUBROUTINE/FUNCTION declaration | Vector ID, filtered lookups, display |
| `unit_type` | Which regex matched | Distinguishes subroutines from functions |
| `parameters` | Parenthesized list after name | Shows routine interface in results |
| `calls` | All `CALL X` statements in body | Powers dependency mapping (`ll deps`) |
| `external_deps` | `EXTERNAL` declarations | Identifies non-intrinsic function dependencies |
| `uses` | `USE module` statements | Tracks module dependencies |
| `purpose` | Extracted from LAPACK comment header | Displayed in results; also embedded |
| `precision` | First letter of name: S/D/C/Z | Enables precision-aware filtering |
| `category` | 2nd-3rd letters of name (GE, SY, PO...) | Matrix type classification |
| `routine_role` | Rule-based classification | Two-pass retrieval boosting (see below) |
| `file_path`, `start_line`, `end_line` | File system + line counter | Exact source location in results |
| `chunk_index`, `chunk_total` | Only for split files | Tracks sub-chunk position |

### Precision Classification

LAPACK routine names encode the numeric precision in the first character:

| Prefix | Precision | Example |
|--------|-----------|---------|
| S | single (float32) | SGESV |
| D | double (float64) | DGESV |
| C | complex | CGESV |
| Z | double complex | ZGESV |

This means DGESV, SGESV, CGESV, and ZGESV are all the same algorithm with different numeric types. We extract this to power precision-variant deduplication at retrieval time.

### Category Classification

Characters 2-3 of the name encode the matrix type:

| Prefix | Category | Example |
|--------|----------|---------|
| GE | general | DGESV (general solver) |
| SY | symmetric | DSYEV (symmetric eigenvalue) |
| PO | positive definite | DPOSV (positive definite solver) |
| LA | auxiliary | DLAQZ0 (internal helper) |
| TR | triangular | DTRSM (triangular solve) |

18 categories total, covering all LAPACK matrix types.

### Routine Role Classification

Every routine is classified as one of four roles:

| Role | How classified | Count | Purpose |
|------|---------------|-------|---------|
| `driver` | Matched against curated list of ~88 base names | 262 | User-facing entry points (DGESV, DSYEV, DGEEV) |
| `computational` | Default for LAPACK SRC/ routines | 1,389 | Internal workhorses (DGETRF, DGETRS) |
| `auxiliary` | Name has `xLA` prefix (2nd-3rd chars = "la") | 666 | Helpers (DLAQZ0, DLASCL) |
| `blas` | File path contains `BLAS/` | ~150 | Basic Linear Algebra Subprograms |

The driver list is derived from the LAPACK Users' Guide and includes all documented entry points: linear solvers, eigenvalue drivers, SVD drivers, least-squares drivers, and their expert variants.

## Chunk Enrichment

Raw Fortran code uses opaque names (DGESV, DGETRF) that no embedding model will associate with natural-language queries like "solve linear equations." Our solution: prepend a natural-language header to every chunk before embedding.

Example of what gets embedded for DGESV:

```
# DGESV
- Double Precision Fortran Subroutine
# Purpose: DGESV computes the solution to a real system of linear equations A * X = B.
# File: dgesv.f | Lines: 1-178 | Calls: DGETRF, DGETRS, XERBLA

      SUBROUTINE DGESV( N, NRHS, A, LDA, IPIV, B, LDB, INFO )
      ...
```

This bridges the vocabulary gap between user queries (English) and code (Fortran naming conventions). The enrichment is constructed from metadata we already extracted, so it costs nothing extra.

Key design decisions:
- **Include the full source code** after the header — the embedding model sees both the NL description and the actual implementation, so it can match on algorithm details too.
- **Include the call graph** in the header — queries like "what calls XERBLA" benefit from having `Calls: DGETRF, DGETRS, XERBLA` in the embedded text.
- **Limit purpose to 2 sentences** — keeps the NL header focused; full docs are in the source comments below.

## Splitting Long Files

~3% of LAPACK files exceed the 15K token limit (we use 15K, not 16K, to leave headroom). These are typically driver routines like DGESVD (1000+ lines) or STRSYL (1001 lines).

### Strategy: Preamble Preservation + Section-Boundary Splitting

1. **Identify the preamble**: everything from the start of the file through the last declaration statement (`INTEGER`, `REAL`, `DOUBLE PRECISION`, `EXTERNAL`, `PARAMETER`, etc.) before the first executable statement. This includes the full comment header and all variable declarations.

2. **Truncate preamble if needed**: cap at ~4K tokens. Most preambles are well under this.

3. **Find split points in executable code**: look for LAPACK's major comment delimiters — lines matching `*  ==========` that separate algorithmic sections (e.g., "Quick return" vs "Main loop" vs "Error handling").

4. **Fallback splitting**: if no comment delimiters found, split at fixed character intervals.

5. **Build sub-chunks**: each sub-chunk = preamble + one section of executable code. Adjacent sub-chunks get 15 lines of overlap to preserve context at boundaries.

6. **Track position**: each sub-chunk gets `chunk_index` (1-indexed) and `chunk_total` metadata, plus the same `unit_name` as the parent. The vector ID includes the chunk index: `fortran:path:name:1`, `fortran:path:name:2`, etc.

Why preamble preservation matters: the comment header contains the purpose description and parameter documentation. Without it, a sub-chunk of executable code is just cryptic Fortran with no semantic anchor for the embedding model.

## Retrieval-Time Strategies

The metadata fields extracted during chunking (`precision`, `routine_role`, `parameters`, `calls`, `unit_name`) directly enable the retrieval pipeline's entity-aware filtering, precision-variant diversification, and driver boosting. See **retrieval_strategy.md** for full documentation of the multi-tier retrieval architecture.

## What We Deliberately Don't Do

- **No AST parsing.** Regex is sufficient for LAPACK's consistent formatting. tree-sitter would add a native dependency for marginal benefit on this codebase.
- **No cross-chunk overlap for normal files.** LAPACK routines are self-contained. Cross-routine relationships are captured via the `calls` metadata field, not text overlap.
- **No comment stripping.** LAPACK comments are the highest-value text for retrieval. They contain natural-language descriptions that directly match user queries.
- **No hierarchical chunking.** We don't create file-level + function-level + block-level chunks. One subroutine per file means there's only one natural level. The metadata fields (calls, parameters, purpose) provide the relational structure.
- **No semantic splitting via LLM.** Regex section-boundary detection works because LAPACK's comment delimiters are perfectly consistent (`* ====...`). LLM-based splitting would be slower and more expensive for no quality gain.

## Key Files

| File | Role |
|------|------|
| `src/legacylens/chunkers/base.py` | `BaseChunker` ABC, `Chunk`, `ChunkMetadata`, `ChunkerRegistry` |
| `src/legacylens/chunkers/fortran.py` | `FortranChunker`, all regex patterns, classification logic |
| `src/legacylens/rag/retrieve.py` | Diversification (`_diversify_results`), two-pass driver boost |
| `src/legacylens/rag/ingest.py` | Pipeline: chunk → embed → upsert |
| `src/legacylens/rag/embeddings.py` | Voyage Code 3 integration (1024-dim vectors) |
| `tests/test_chunker.py` | Tests on DGESV, DGETRF, DGEMM, precision variants, edge cases |
| `tests/test_retrieve.py` | Tests for diversification, two-pass retrieval, pin_unit |
