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

## Performance

Retrieval quality evaluated against 15 golden queries across 5 categories (top-k=5):

| Category | Queries | Precision@5 | Recall@5 | MRR | Hit Rate | Retrieval (ms) | E2E (ms) |
|---|---|---|---|---|---|---|---|
| Entity (direct) | 4 | 0.20 | 1.00 | 1.00 | 1.00 | 1138 | 8795 |
| Entity (callers) | 3 | 0.33 | 1.00 | 0.78 | 1.00 | 560 | 6991 |
| Parameter-based | 2 | 0.20 | 0.75 | 1.00 | 1.00 | 515 | 8165 |
| Semantic | 4 | 0.20 | 1.00 | 0.83 | 1.00 | 565 | 13300 |
| Utility | 2 | 0.20 | 1.00 | 1.00 | 1.00 | 682 | 10332 |
| **Overall** | **15** | **0.23** | **0.97** | **0.91** | **1.00** | **726** | **9756** |

**Note: Precision is low (0.20–0.33) as expected since most queries have only 1–2 expected units in top-5**

## Examples

#### CLI query for top result with code snippets shown (`--show-code` defaults to false)
```bash
legacylens query "What does DGESV do?" --show-code --top-k 1
```

```
Found 1 relevant chunks:

1. SUBROUTINE DGESV  score: 0.758  (driver)
  File          dgesv.f:1-178
  Language      fortran
  Purpose       DGESV computes the solution to a real system of linear equations A * X = B, where A is an
                N-by-N matrix and X and B are N-by-NRHS matrices. The LU decomposition with partial
                pivoting and row interchan...
  Parameters    N, NRHS, A, LDA, IPIV, B, LDB, INFO
  Calls         DGETRF, DGETRS, XERBLA
    1 *> \addtogroup gesv
    2 *>
    3 *> \brief <b> DGESV computes the solution to system of linear equations A * X = B for GE matrices</b>
    ...
   21 *       SUBROUTINE DGESV( N, NRHS, A, LDA, IPIV, B, LDB, INFO )
    ...
  121       SUBROUTINE DGESV( N, NRHS, A, LDA, IPIV, B, LDB, INFO )
  122       IMPLICIT NONE
    ...
  165       CALL DGETRF( N, N, A, LDA, IPIV, INFO )
  166       IF( INFO.EQ.0 ) THEN
  170          CALL DGETRS( 'No transpose', N, NRHS, A, LDA, IPIV, B, LDB,
  171      $                INFO )
  172       END IF
  173       RETURN
  177       END

╭───────────────────────────────── LegacyLens Answer ──────────────────────────────────╮
│                              DGESV: Solving Linear Systems                            │
│                                                                                       │
│ DGESV solves a system of linear equations of the form:                                │
│                                                                                       │
│  A * X = B                                                                            │
│                                                                                       │
│ DGESV uses a two-step approach:                                                       │
│                                                                                       │
│ Step 1: LU Factorization (DGETRF)                                                     │
│ Step 2: Triangular Solve (DGETRS)                                                     │
│                                                                                       │
│  • INFO = 0: Success                                                                  │
│  • INFO < 0: Invalid input parameter                                                  │
│  • INFO > 0: The matrix is singular                                                   │
╰───────────────────────────────────────────────────────────────────────────────────────╯
```

#### Default query with explain mode, returning top 5 results
```bash
legacylens query "How does pivoting work in LU factorization?"
```

```
Query: How does pivoting work in LU factorization?

Found 5 relevant chunks:

1. SUBROUTINE DGETC2  score: 0.574  (computational)
  File          dgetc2.f:1-230
  Language      fortran
  Purpose       DGETC2 computes an LU factorization with complete pivoting of
                the n-by-n matrix A. The factorization has the form A = P * L
                * U * Q, where P and Q are permutation matrices, L is lower
                triangular with ...
  Parameters    N, A, LDA, IPIV, JPIV, INFO
  Calls         DGER, DSWAP

2. SUBROUTINE DGETRF  score: 0.576  (computational)
  File          dgetrf.f:1-226
  Language      fortran
  Purpose       DGETRF computes an LU factorization of a general M-by-N matrix
                A using partial pivoting with row interchanges. The
                factorization has the form A = P * L * U where P is a
                permutation matrix, L is lower ...
  Parameters    M, N, A, LDA, IPIV, INFO
  Calls         DGEMM, DGETRF2, DLASWP, DTRSM, XERBLA

3. SUBROUTINE DGETF2  score: 0.584  (computational)
  File          dgetf2.f:1-211
  ...

4. SUBROUTINE DGETRF2  score: 0.587  (computational)
  File          dgetrf2.f:1-271
  ...

5. SUBROUTINE DGESV  score: 0.488  (driver)
  File          dgesv.f:1-178
  ...

╭───────────────────────────── LegacyLens Answer ──────────────────────────────╮
│                         Pivoting in LU Factorization                         │
│                                                                              │
│ Pivoting is a crucial technique used in LU factorization to maintain         │
│ numerical stability and handle singularities.                                │
│                                                                              │
│ Two Pivoting Strategies in LAPACK                                            │
│                                                                              │
│ 1. Partial Pivoting (Used in DGETF2, DGETRF, DGETRF2)                        │
│                                                                              │
│  • IDAMAX finds the row with the maximum absolute value in column J          │
│  • IPIV(J) records which row was swapped into position J                     │
│  • DSWAP exchanges the entire row with the current row                       │
│                                                                              │
│ 2. Complete Pivoting (Used in DGETC2)                                        │
│                                                                              │
│  • Searches all remaining rows and columns for the largest element           │
│  • Tracks both row swaps (IPIV) and column swaps (JPIV)                      │
│  • More expensive computationally (O(n^3) search vs O(n^2))                  │
│  • Provides better numerical stability for nearly-singular matrices          │
│                                                                              │
│ Summary                                                                      │
│                                                                              │
│  Feature       Partial Pivoting              Complete Pivoting (DGETC2)      │
│  Search scope  One column                    Entire submatrix                │
│  Cost          O(n^2)                        O(n^3)                          │
│  Stability     Good for most cases           Better for ill-conditioned      │
│  Pivot arrays  IPIV only                     IPIV + JPIV                     │
╰──────────────────────────────────────────────────────────────────────────────╯
```

#### Limit results with `--top-k`
```bash
legacylens query "What routines handle eigenvalues?" --top-k 3
```

```
Query: What routines handle eigenvalues?

Found 3 relevant chunks:

1. SUBROUTINE DHGEQZ  score: 0.612  (computational)
  File          dhgeqz.f:1-1382
  Language      fortran
  Purpose       DHGEQZ computes the eigenvalues of a real matrix pair (H,T),
                where H is an upper Hessenberg matrix and T is upper
                triangular, using the double-shift QZ method.
  Parameters    JOB, COMPQ, COMPZ, N, ILO, IHI, H, LDH, T, LDT
  Calls         DLAG2, DLARFG, DLARTG, DLASET, DLASV2, DROT, XERBLA

2. SUBROUTINE DGEEV  score: 0.571  (driver)
  File          dgeev.f:1-541
  Language      fortran
  Purpose       DGEEV computes for an N-by-N real nonsymmetric matrix A, the
                eigenvalues and, optionally, the left and/or right
                eigenvectors.
  Parameters    JOBVL, JOBVR, N, A, LDA, WR, WI, VL, LDVL, VR
  Calls         DGEBAK, DGEBAL, DGEHRD, DHSEQR, DLACPY, DLARTG, DLASCL,
                DORGHR, DROT, DSCAL

3. SUBROUTINE DGGEVX  score: 0.587  (driver)
  File          dggevx.f:1-876
  Language      fortran
  Purpose       DGGEVX computes for a pair of N-by-N real nonsymmetric
                matrices (A,B) the generalized eigenvalues, and optionally,
                the left and/or right generalized eigenvectors.
  Parameters    BALANC, JOBVL, JOBVR, SENSE, N, A, LDA, B, LDB, ALPHAR
  Calls         DGEQRF, DGGBAK, DGGBAL, DGGHRD, DHGEQZ, DLACPY, DLASCL,
                DLASET, DORGQR, DORMQR

╭───────────────────────────── LegacyLens Answer ──────────────────────────────╮
│                  Eigenvalue Computation Routines in LAPACK                   │
│                                                                              │
│ High-Level Driver Routines                                                   │
│                                                                              │
│ DGEEV - General Eigenvalue Problem (Standard)                                │
│  • Input: Matrix A                                                           │
│  • Output: Real and imaginary parts of eigenvalues (WR, WI)                  │
│  • Call chain: DGEEV -> DGEBAL -> DGEHRD -> DHSEQR -> DTREVC3 -> DGEBAK      │
│                                                                              │
│ DGGEVX - Generalized Eigenvalue Problem (with balancing & conditioning)      │
│  • Input: Matrices A and B                                                   │
│  • Output: Generalized eigenvalues as (alpha, beta) pairs                    │
│  • Call chain: DGGEVX -> DGGBAL -> DGGHRD -> DHGEQZ -> DTGEVC -> DGGBAK     │
│                                                                              │
│ Low-Level Computational Routine                                              │
│                                                                              │
│ DHGEQZ - QZ Algorithm for Hessenberg-Triangular Pairs                        │
│  • The core computational engine using the double-shift QZ method            │
│  • Input: H (upper Hessenberg), T (upper triangular)                         │
│  • Output: Eigenvalues as (alpha, beta) pairs                                │
╰──────────────────────────────────────────────────────────────────────────────╯
```

#### Retrieval only with `--no-answer` (skip LLM generation)
```bash
legacylens query "matrix transpose" --no-answer
```

```
Query: matrix transpose

Found 5 relevant chunks:

1. SUBROUTINE DTFSM  score: 0.491  (computational)
  File          dtfsm.f:1-1031
  Language      fortran
  Purpose       Level 3 BLAS like routine for A in RFP Format. DTFSM solves
                the matrix equation op( A )*X = alpha*B or X*op( A ) =
                alpha*B where alpha is a scalar, X and B are m by n matrices,
                A is a unit, or non-...
  Parameters    TRANSR, SIDE, UPLO, TRANS, DIAG, M, N, ALPHA, A, B
  Calls         DGEMM, DTRSM, XERBLA

2. SUBROUTINE DTRSYL3  score: 0.497  (computational)
  File          dtrsyl3.f:1-1264
  Language      fortran
  Purpose       DTRSYL3 solves the real Sylvester matrix equation: op(A)*X +
                X*op(B) = scale*C or op(A)*X - X*op(B) = scale*C, where op(A)
                = A or A**T, and A and B are both upper quasi-triangular.
  Parameters    TRANA, TRANB, ISGN, M, N, A, LDA, B, LDB, C
  Calls         DGEMM, DLASCL, DSCAL, DTRSYL, XERBLA

3. SUBROUTINE DGESVD  score: 0.455  (driver)
  File          dgesvd.f:209-3553
  Language      fortran
  Purpose       DGESVD computes the singular value decomposition (SVD) of a
                real M-by-N matrix A, optionally computing the left and/or
                right singular vectors. The SVD is written A = U * SIGMA *
                transpose(V)...
  Parameters    JOBU, JOBVT, M, N, A, LDA, S, U, LDU, VT
  Calls         DBDSQR, DGEBRD, DGELQF, DGEMM, DGEQRF, DLACPY, DLASCL,
                DLASET, DORGBR, DORGLQ

4. SUBROUTINE DTRSYL  score: 0.496  (computational)
  File          dtrsyl.f:1-1001
  Language      fortran
  Purpose       DTRSYL solves the real Sylvester matrix equation: op(A)*X +
                X*op(B) = scale*C or op(A)*X - X*op(B) = scale*C, where op(A)
                = A or A**T, and A and B are both upper quasi-triangular.
  Parameters    TRANA, TRANB, ISGN, M, N, A, LDA, B, LDB, C
  Calls         DLALN2, DLASY2, DSCAL, XERBLA

5. SUBROUTINE DGESVJ  score: 0.483  (computational)
  File          dgesvj.f:340-1662
  Language      fortran
  Purpose       DGESVJ computes the singular value decomposition (SVD) of a
                real M-by-N matrix A, where M >= N.
  Parameters    JOBA, JOBU, JOBV, M, N, A, LDA, SVA, MV, V
  Calls         DAXPY, DCOPY, DGSVJ0, DGSVJ1, DLASCL, DLASET, DLASSQ,
                DROTM, DSCAL, DSWAP
```

#### Dependency graph mode with `--mode deps`
```bash
legacylens query "What does DGETRF call?" --mode deps
```

```
Query: What does DGETRF call?

Found 5 relevant chunks:

1. SUBROUTINE DGETRF  score: 0.681  (computational)
  File          VARIANTS/lu/REC/dgetrf.f:1-278
  Language      fortran
  Parameters    M, N, A, LDA, IPIV, INFO
  Calls         DGEMM, DLASWP, DSCAL, DTRSM, TO, XERBLA

2. SUBROUTINE DGESV  score: 0.531  (driver)
  File          dgesv.f:1-178
  ...

3. SUBROUTINE DGESVXX  score: 0.502  (driver)
  File          dgesvxx.f:1-768
  ...

4. SUBROUTINE DGESVX  score: 0.481  (driver)
  File          dgesvx.f:1-605
  ...

5. SUBROUTINE DSGESV  score: 0.461  (computational)
  File          dsgesv.f:1-434
  ...

╭───────────────────────────── LegacyLens Answer ──────────────────────────────╮
│                             DGETRF Dependencies                              │
│                                                                              │
│ External Subroutines Called:                                                 │
│                                                                              │
│  DTRSM   - Solves triangular systems (used to solve for U block)             │
│  DSCAL   - Scales a vector (used to normalize the pivot column)              │
│  DLASWP  - Applies row permutations (used for pivot swaps)                   │
│  DGEMM   - General matrix multiplication (Schur complement update)           │
│  XERBLA  - Error handler (called for invalid input parameters)               │
│                                                                              │
│ External Functions Called:                                                   │
│                                                                              │
│  DLAMCH  - Returns machine constants (specifically 'S' for safe minimum)     │
│  IDAMAX  - Finds the index of maximum absolute value (pivot selection)       │
│  DISNAN  - Tests for NaN values                                              │
│                                                                              │
│ Call Chain in the Algorithm                                                  │
│                                                                              │
│  1 IDAMAX - Find the pivot (maximum element in current column)               │
│  2 DLASWP - Apply pending row permutations to L and U blocks                 │
│  3 DSCAL  - Scale the pivot column to normalize the L factor                 │
│  4 DTRSM  - Solve for the U block using the L triangular factor              │
│  5 DGEMM  - Update the remaining Schur complement                            │
│                                                                              │
│ Usage Context                                                                │
│                                                                              │
│  • DGESV   - Simple LU solver (calls DGETRF, then DGETRS)                    │
│  • DGESVX  - Expert solver with error bounds                                 │
│  • DGESVXX - Extended expert solver                                          │
│  • DSGESV  - Mixed-precision solver                                          │
╰──────────────────────────────────────────────────────────────────────────────╯
```

#### `explain` shortcut
```bash
legacylens explain "How does DGEMM perform matrix multiplication?"
```

```
Explaining: How does DGEMM perform matrix multiplication?

╭──────────────────────────────── Explanation ─────────────────────────────────╮
│                   How DGEMM Performs Matrix Multiplication                   │
│                                                                              │
│ DGEMM (Double precision GEneral Matrix Multiply) is a Level 3 BLAS routine   │
│ that computes:                                                               │
│                                                                              │
│  C := alpha*op(A)*op(B) + beta*C                                             │
│                                                                              │
│ where:                                                                       │
│  • op(X) is either X or X^T (transpose), controlled by TRANSA and TRANSB     │
│  • alpha and beta are scalar multipliers                                     │
│  • A is an m x k matrix (after applying op)                                  │
│  • B is a k x n matrix (after applying op)                                   │
│  • C is an m x n matrix (the accumulator)                                    │
│                                                                              │
│ From the signature:                                                          │
│                                                                              │
│  SUBROUTINE DGEMM(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB,            │
│                    BETA, C, LDC)                                             │
│                                                                              │
│ Why This Design?                                                             │
│                                                                              │
│ DGEMM is a Level 3 BLAS operation -- the computational cost is O(n^3) while  │
│ memory access is only O(n^2), allowing modern CPUs to achieve high           │
│ performance through cache optimization and parallelization. This makes it    │
│ the workhorse for most eigenvalue and factorization algorithms.              │
╰──────────────────────────────────────────────────────────────────────────────╯
```

#### `deps` shortcut (pins the routine, uses `--top-k 10`)
```bash
legacylens deps DGESV
```

```
Dependencies for: DGESV

╭──────────────────────────── Dependencies: DGESV ─────────────────────────────╮
│                  DGESV Dependencies and Call Relationships                   │
│                                                                              │
│ What DGESV Calls                                                             │
│                                                                              │
│  CALL DGETRF( N, N, A, LDA, IPIV, INFO )                                     │
│  IF( INFO.EQ.0 ) THEN                                                        │
│     CALL DGETRS( 'No transpose', N, NRHS, A, LDA, IPIV, B, LDB, INFO )       │
│  END IF                                                                      │
│                                                                              │
│ Core Dependencies:                                                           │
│                                                                              │
│  1 DGETRF - LU factorization with partial pivoting                           │
│  2 DGETRS - Solve using LU factorization                                     │
│  3 XERBLA - Error handler                                                    │
│                                                                              │
│ Algorithm Flow                                                               │
│                                                                              │
│  Input: A (NxN matrix), B (NxNRHS right-hand sides)                          │
│           |                                                                  │
│     [Validate inputs via XERBLA if needed]                                   │
│           |                                                                  │
│     DGETRF: A = P*L*U factorization                                          │
│           |                                                                  │
│     [Check if factorization succeeded (INFO==0)]                             │
│           |                                                                  │
│     DGETRS: Solve P*L*U*X = B -> X                                           │
│           |                                                                  │
│  Output: X (solution), IPIV (pivot info), INFO (status)                      │
│                                                                              │
│ What Calls DGESV                                                             │
│                                                                              │
│  • SGESV   - Single precision equivalent                                     │
│  • DSGESV  - Mixed precision variant with iterative refinement               │
│  • DGESVX  - Expert driver with equilibration and error bounds               │
│  • DGESVXX - Extra-precise expert driver with detailed error analysis        │
╰──────────────────────────────────────────────────────────────────────────────╯
```

#### Generate modernized documentation with `document`
```bash
legacylens document DGETRF
```

```
Generating docs for: DGETRF

╭─────────────────────────── Documentation: DGETRF ────────────────────────────╮
│               DGETRF -- LU Factorization with Partial Pivoting               │
│                                                                              │
│ Function Signature                                                           │
│                                                                              │
│  SUBROUTINE DGETRF( M, N, A, LDA, IPIV, INFO )                               │
│     INTEGER :: M, N, LDA, INFO                                               │
│     DOUBLE PRECISION :: A( LDA, * )                                          │
│     INTEGER :: IPIV( * )                                                     │
│  END SUBROUTINE                                                              │
│                                                                              │
│ Parameters                                                                   │
│                                                                              │
│  Parameter  Type              Intent  Description                            │
│  M          INTEGER           IN      Number of rows in matrix A             │
│  N          INTEGER           IN      Number of columns in matrix A          │
│  A          DOUBLE PRECISION  INOUT   MxN matrix to factor. On exit,         │
│                                       contains L and U.                      │
│  LDA        INTEGER           IN      Leading dimension of A                 │
│  IPIV       INTEGER           OUT     Pivot indices array of length min(M,N) │
│  INFO       INTEGER           OUT     Status code: 0 = success               │
│                                                                              │
│ Algorithm                                                                    │
│                                                                              │
│ The routine uses a blocked right-looking Level 3 BLAS algorithm:             │
│  1 Determines an optimal block size via ILAENV                               │
│  2 Processes the matrix in column panels of size NB                          │
│  3 For each panel:                                                           │
│     • Calls DGETRF2 to factor the diagonal block                             │
│     • Applies row swaps to previous columns (DLASWP)                         │
│     • Computes the block row of U (DTRSM)                                    │
│     • Updates the remaining trailing submatrix (DGEMM)                       │
│                                                                              │
│ Call Chain                                                                   │
│                                                                              │
│  DGETRF                                                                      │
│  +-- ILAENV       (determine optimal block size)                             │
│  +-- DGETRF2      (unblocked LU or factorize current panel)                  │
│  +-- DLASWP       (apply row permutations)                                   │
│  +-- DTRSM        (solve triangular systems: compute U block)                │
│  +-- DGEMM        (matrix multiply: Schur complement update)                 │
│  +-- XERBLA       (error handling)                                           │
│                                                                              │
│ Related Routines                                                             │
│                                                                              │
│  DGETF2   Unblocked LU factorization (called by DGETRF2)                     │
│  DGETRS   Solve Ax=b using LU factors from DGETRF                            │
│  DGETRI   Compute matrix inverse from LU factors                             │
│  DGESV    Solve Ax=b directly (combines DGETRF + DGETRS)                     │
│  ZGETRF   Complex variant (same algorithm, complex*16)                       │
╰──────────────────────────────────────────────────────────────────────────────╯
```

#### Extract mathematical logic with `logic`
```bash
legacylens logic "How does partial pivoting improve numerical stability?"
```

```
Extracting logic: How does partial pivoting improve numerical stability?

╭─────────────────────────────── Business Logic ───────────────────────────────╮
│              How Partial Pivoting Improves Numerical Stability               │
│                                                                              │
│ The Problem It Solves                                                        │
│                                                                              │
│ When solving systems like A*X = B using Gaussian elimination, if the pivot   │
│ element is very small, you divide by that small number:                      │
│                                                                              │
│  factor = A(i,j) / A(j,j)   # If A(j,j) is tiny, factor becomes huge        │
│                                                                              │
│ This amplifies rounding errors and can make the computed solution wildly     │
│ inaccurate.                                                                  │
│                                                                              │
│ The Partial Pivoting Solution                                                │
│                                                                              │
│ From the algorithm in DGETF2 (the unblocked LU factorization):               │
│                                                                              │
│  DO 10 J = 1, MIN( M, N )                                                    │
│           JP = J - 1 + IDAMAX( M-J+1, A( J, J ), 1 )                         │
│           IPIV( J ) = JP                                                     │
│           IF( A( JP, J ).NE.ZERO ) THEN                                      │
│              IF( JP.NE.J )                                                   │
│       $         CALL DSWAP( N, A( J, 1 ), LDA, A( JP, 1 ), LDA )             │
│                                                                              │
│ Key steps:                                                                   │
│  1 IDAMAX - Find the element with the largest absolute value in column J     │
│  2 IPIV(J) = JP - Record the swap                                            │
│  3 DSWAP - Exchange rows so the largest element becomes the pivot            │
│                                                                              │
│  Issue                Without Pivoting           With Partial Pivoting       │
│  Pivot selection      Uses A(j,j) regardless     Uses max(|A(j,j)|, ...)    │
│  Division factor      Can be ~0.0001             Guaranteed largest avail.   │
│  Error amplification  factor ~ 10,000x           factor ~ 1 (controlled)    │
│  Singular detection   Might miss singularity     Detects when ALL pivots=0  │
│                                                                              │
│ Error Bounds                                                                 │
│                                                                              │
│  error ~ machine_epsilon x condition_number(A) x growth_factor               │
│                                                                              │
│ Partial pivoting keeps growth_factor <= 2^(N-1) in worst case, whereas       │
│ without pivoting it can be unbounded.                                        │
╰──────────────────────────────────────────────────────────────────────────────╯
```

#### View source file with syntax highlighting
```bash
legacylens view SRC/dgesv.f
```

```
SRC/dgesv.f (177 lines)

    1 *> \addtogroup gesv
    2 *>
    3 *> \brief <b> DGESV computes the solution to system of linear equations A
    4 *
    5 *  =========== DOCUMENTATION ===========
    ...
   21 *       SUBROUTINE DGESV( N, NRHS, A, LDA, IPIV, B, LDB, INFO )
   22 *
   23 *       .. Scalar Arguments ..
   24 *       INTEGER            INFO, LDA, LDB, N, NRHS
   25 *       ..
   26 *       .. Array Arguments ..
   27 *       INTEGER            IPIV( * )
   28 *       DOUBLE PRECISION   A( LDA, * ), B( LDB, * )
   29 *       ..
    ...
  121       SUBROUTINE DGESV( N, NRHS, A, LDA, IPIV, B, LDB, INFO )
  122       IMPLICIT NONE
    ...
  165       CALL DGETRF( N, N, A, LDA, IPIV, INFO )
  166       IF( INFO.EQ.0 ) THEN
  170          CALL DGETRS( 'No transpose', N, NRHS, A, LDA, IPIV, B, LDB,
  171      $                INFO )
  172       END IF
  173       RETURN
  176 *
  177       END
```

#### Show index statistics
```bash
legacylens stats
```

```
Index Stats:
  Total vectors: 2320
  Dimension: 1024
```
