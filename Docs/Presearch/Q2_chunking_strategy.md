# Technical Research: Chunking Strategy (Fortran/LAPACK)

## Decision
How to chunk LAPACK Fortran source files for embedding and retrieval.

## Constraints
- LAPACK follows one-subroutine-per-file convention (each .f file = one SUBROUTINE/FUNCTION)
- Mix of fixed-form (.f, Fortran 77) and free-form (.f90) files
- Embedding model (Voyage Code 2) has 16,000 token context window
- 85%+ of subroutines fit in one chunk (< 16K tokens)
- ~3% of files exceed 16K tokens (driver routines like DGESVD, DGEEV)
- Rich comment headers with natural-language descriptions in every file
- Need pluggable interface for future COBOL support

## LAPACK Structure

### File Organization
- `SRC/` — Core routines (~1,700 files) — primary ingestion target
- `BLAS/SRC/` — BLAS routines (~150 files)
- `TESTING/` — Test drivers
- Naming: first letter = precision (s=single, d=double, c=complex, z=double-complex)

### Subroutine Size Distribution
| Category | Lines | Proportion | Tokens (approx) |
|----------|-------|-----------|-----------------|
| Small utility | 30-100 | ~40% | 120-800 |
| Medium solvers | 100-500 | ~45% | 400-4,000 |
| Large drivers | 500-2000 | ~12% | 2,000-16,000 |
| Very large | 2000-4000+ | ~3% | 8,000-32,000 |

## Parsing Approach

### Candidates
| Approach | Effort | Reliability on LAPACK | Dependencies |
|----------|--------|----------------------|-------------|
| Regex-based | 0.5 day | Excellent (consistent formatting) | None |
| tree-sitter-fortran | 1-2 days | Very good | py-tree-sitter |
| fparser2 | 1-2 days | Fair (weak fixed-form) | fparser |

### Recommendation: Regex-based
LAPACK's extreme consistency makes regex optimal. Key patterns:
- Subroutine: `^\s{6}SUBROUTINE\s+(\w+)`
- Function: `^\s{6}.*FUNCTION\s+(\w+)`
- End: `^\s{6}END\s*$`
- Calls: `CALL\s+(\w+)`
- Externals: `EXTERNAL\s+(.+)`
- Purpose: text between `Purpose` and next `=====` delimiter

Fallback to tree-sitter-fortran if edge cases emerge.

## Chunking Strategy

### Primary: One chunk per file (= one subroutine)
Since LAPACK = one subroutine per file, each file becomes one chunk containing:
1. Full comment header (purpose, parameter docs, algorithm notes)
2. Subroutine declaration and body
3. END statement

**Never strip comments** — they contain the natural-language descriptions that drive retrieval quality.

### Splitting Long Subroutines (> 16K tokens)
For the ~3% of files exceeding the embedding model's limit:
1. Keep comment header + declaration as a "preamble" (up to ~4K tokens) included in every sub-chunk
2. Split executable code at major comment-delimited section boundaries
3. Include 10-20 lines overlap between consecutive code sections
4. Add metadata: `chunk_index: 1/N`, `parent_function: DGESVD`

### Chunk Enrichment
Prepend a 1-3 line natural-language summary before raw source:
```
# DGESV - Double-precision General Solver
# Purpose: Computes the solution to a real system of linear equations A * X = B
# Calls: DGETRF, DGETRS, XERBLA | File: SRC/dgesv.f | Lines: 1-120
```
This bridges LAPACK's opaque naming (DGESV) to natural-language queries ("solve linear equations").

### Overlap Strategy
**Not needed for function-level chunks** — each subroutine is self-contained. Cross-function relationships captured via CALL metadata. Exception: split sub-chunks get preamble inclusion + 10-20 line overlap.

## Metadata Per Chunk
| Field | Extraction |
|-------|-----------|
| `file_path` | File system |
| `start_line` / `end_line` | Line counter |
| `subroutine_name` | Regex on declaration |
| `unit_type` | "subroutine" / "function" |
| `precision` | First letter: s/d/c/z |
| `parameters` | Parenthesized list in declaration |
| `calls` | All `CALL X` statements |
| `external_deps` | `EXTERNAL` declarations |
| `category` | Prefix mapping (GE=general, SY=symmetric, PO=positive-definite) |
| `purpose` | First sentence of Purpose section |
| `chunk_index` / `chunk_total` | For split routines only |
| `token_count` | Counted during embedding |

## Pluggable Interface Design

```
BaseChunker (ABC)
├── supported_extensions() → [".f", ".f90"]
├── chunk_file(path, content) → list[Chunk]
└── chunk_directory(dir) → list[Chunk]

Chunk
├── content: str          (raw source)
├── enriched_content: str (with NL summary prefix)
└── metadata: ChunkMetadata

ChunkerRegistry
├── register(chunker)
└── get_chunker(file_path) → BaseChunker
```

COBOL chunker later implements the same interface, parsing DIVISION/SECTION/PARAGRAPH boundaries, extracting PERFORM instead of CALL, COPY instead of USE.

## Recommendation Summary
| Decision | Choice | Rationale |
|----------|--------|-----------|
| Parser | Regex-based | LAPACK's consistency; saves 1+ day vs tree-sitter |
| Chunk boundary | One chunk per file | One-subroutine-per-file convention |
| Comment headers | Always include | Critical for embedding quality |
| Long file splitting | Preamble + comment sections | Only ~3% of files; preserves context |
| Enrichment | Prepend NL summary | Bridges opaque naming to queries |
| Overlap | None (preamble for splits) | Functions are self-contained |
| Interface | BaseChunker ABC | Clean COBOL extension point |
| Sprint effort | ~1.5 days | Parser + metadata + interface + tests |

## Decision Confirmation
- [x] User confirmed on 2026-03-02
