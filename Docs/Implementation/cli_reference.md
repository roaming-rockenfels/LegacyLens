# CLI Reference

## Overview

LegacyLens ships a single CLI binary `legacylens` (aliased to `ll` for convenience) built on Typer. Every command follows the same pattern: accept a natural-language question or routine name, run the retrieval pipeline against Pinecone, optionally pass retrieved chunks to an LLM for answer generation, and render output to the terminal via Rich.

There are 10 commands organized into three groups:

| Group | Commands | What they do |
|-------|----------|--------------|
| **Ingestion** | `ingest`, `batch-ingest` | Load Fortran source into the vector database |
| **Query** | `query`, `explain`, `deps`, `document`, `logic` | Ask questions, get answers |
| **Operations** | `view`, `stats`, `serve` | Inspect source, check index health, start API server |

## Prerequisites

Three environment variables must be set (via `.env` or shell):

| Variable | Required by | Purpose |
|----------|------------|---------|
| `VOYAGE_API_KEY` | All commands except `view`, `stats` | Voyage Code 3 embeddings + reranking |
| `PINECONE_API_KEY` | All commands except `view` | Vector storage and retrieval |
| `OPENROUTER_API_KEY` | All query commands (not `--no-answer`) | LLM answer generation via Claude |

Optional:

| Variable | Default | Purpose |
|----------|---------|---------|
| `PINECONE_INDEX_NAME` | `"legacylens"` | Name of the Pinecone index |
| `LAPACK_DATA_DIR` | `"data/lapack"` | Base path for `view` command file lookups |

## Ingestion Commands

These load Fortran source code into Pinecone. You run them once (or when the codebase changes), not per-query.

### `ll ingest <path>`

Synchronous ingestion using the real-time Voyage API. Chunks all Fortran files in `<path>`, embeds them in small batches, and upserts to Pinecone.

```
ll ingest data/lapack
```

**Pipeline:** chunk files → embed in batches of 3 → upsert to Pinecone in batches of 100.

**Rate limiting:** The free Voyage tier allows 3 requests/minute and 10K tokens/minute. The default settings (`EMBED_BATCH_SIZE=3`, `EMBED_DELAY=25s`) stay within these limits. For the full LAPACK codebase (~1,850 chunks), this takes approximately 2.5 hours.

**When to use this:** Small codebases, incremental updates, or when you have a paid Voyage plan (adjust `EMBED_BATCH_SIZE` and `EMBED_DELAY` in `ingest.py`).

**Output:**
```
Step 1/3: Chunking files in data/lapack...
  Found 1847 chunks in 1.8s
Step 2/3: Generating embeddings with Voyage Code 3...
  Generated 1847 embeddings in 9240.3s
Step 3/3: Upserting to Pinecone...
  Upserted 1847 vectors in 12.4s
Ingestion complete! 9254.5s total
```

### `ll batch-ingest <path>`

Asynchronous ingestion using the Voyage Batch API. Writes all chunks to a JSONL file, uploads it, creates a batch job, polls for completion, then downloads results and upserts to Pinecone.

```
ll batch-ingest data/lapack
ll batch-ingest data/lapack --no-reset  # keep existing index
```

| Option | Default | Purpose |
|--------|---------|---------|
| `--no-reset` | `False` | Skip deleting and recreating the Pinecone index |

**Pipeline:** chunk files → write JSONL → upload to Voyage Files API → create batch job → poll every 10-60s → download embeddings → upsert to Pinecone.

**Why this exists:** The Batch API processes all embeddings server-side with no per-request rate limits. The full LAPACK codebase completes in ~10-15 minutes instead of 2.5 hours. The tradeoff is that it's all-or-nothing — you can't incrementally add files.

**Default behavior deletes the Pinecone index** before upserting, ensuring a clean slate. Use `--no-reset` to append to an existing index (useful if you've already ingested and want to add more files, though duplicate vectors aren't deduplicated automatically).

**Why batch JSONL instead of individual API calls:** The Voyage Batch API accepts a single JSONL file where each line is an embedding request containing up to 20 chunks (~60K tokens). This reduces 1,850 individual API calls to ~93 batch lines in a single file upload, eliminating rate limit concerns entirely.

## Query Commands

All query commands follow the same internal flow:

```
question → retrieve(question, top_k) → [optional] generate_answer(question, results, mode) → display
```

The retrieval pipeline is documented in `retrieval_strategy.md`. The LLM generation step sends retrieved chunks + the question to Claude Haiku 4.5 via OpenRouter with a mode-specific system prompt.

### `ll query <question>`

The primary command. Retrieves relevant chunks, displays them, then generates an LLM answer.

```
ll query "How does LAPACK solve linear equations?"
ll query "routines that call XERBLA" --no-answer
ll query "what does DGETRF do?" -k 10 -m deps
```

| Option | Default | Purpose |
|--------|---------|---------|
| `--top-k`, `-k` | `5` | Number of chunks to retrieve |
| `--mode`, `-m` | `"explain"` | LLM response mode (see below) |
| `--no-answer` | `False` | Show retrieved chunks only, skip the LLM call |

**Response modes** control the system prompt sent to Claude:

| Mode | Instruction given to LLM | Max tokens |
|------|-------------------------|------------|
| `explain` | "Provide a clear explanation of what this code does, how it works, and why." | 4096 |
| `deps` | "Focus on the dependency relationships: what calls what, what external libraries are used, and the data flow between routines." | 4096 |
| `docs` | "Generate concise modern documentation for the code, including function signature, parameter table, computation description, 1-2 usage examples, and related routines." | 8192 |
| `business_logic` | "Extract and explain the core business/mathematical logic, the algorithm being implemented, and its practical applications." | 4096 |

**`--no-answer` is the debugging workhorse.** It shows you exactly what the retrieval pipeline returns — chunk names, scores, metadata, call graphs — without waiting for or paying for an LLM call. Every retrieval quality investigation starts here.

**Why modes instead of separate commands:** The retrieval step is identical for all modes. Only the LLM prompt changes. Modes keep the CLI surface small while letting you tune the output style. The `explain`, `deps`, `document`, and `logic` convenience commands are just shortcuts that hardcode a mode.

### `ll explain <question>`

Shortcut for `ll query <question> --mode explain`. Retrieves chunks and generates an explanation.

```
ll explain "How does partial pivoting work in DGETRF?"
```

| Option | Default | Purpose |
|--------|---------|---------|
| `--top-k`, `-k` | `5` | Number of chunks to retrieve |

**Difference from `ll query`:** No `--mode` or `--no-answer` options. Always generates an LLM answer in explain mode.

### `ll deps <unit_name>`

Shows dependency relationships for a specific routine. Uses the **pin-unit retrieval path** — the named routine is guaranteed to appear first in results, with semantically related routines filling remaining slots.

```
ll deps DGESV
ll deps DGETRF -k 15
```

| Option | Default | Purpose |
|--------|---------|---------|
| `--top-k`, `-k` | `10` | Number of chunks to retrieve (default is higher than other commands) |

**How it works internally:**
1. Constructs the question: `"What are the dependencies and call relationships of {unit_name}? What does it call and what calls it?"`
2. Calls `retrieve(question, top_k=10, pin_unit=unit_name)`
3. The pin-unit path queries `filter={"unit_name": "DGESV"}` first (gets the routine + any split chunks), then fills remaining slots with semantic search.
4. Passes results to the LLM in `deps` mode.

**Why pin-unit instead of just querying:** Without pinning, the named routine might not appear in results at all. A query about "dependencies of DGESV" might return DGETRF (which DGESV calls) ranked higher than DGESV itself, because DGETRF's enriched content has more overlap with "dependencies" and "call relationships". Pinning guarantees the target routine is present so the LLM can describe its call graph from the source.

**Why top_k defaults to 10:** Dependency analysis benefits from more context. DGESV calls DGETRF, DGETRS, and XERBLA. DGETRF calls DGETRF2, DLASWP, DTRSM, DGEMM, and XERBLA. With top_k=5 you'd miss half the dependency chain. 10 captures two hops in most cases.

### `ll document <unit_name>`

Generates modern documentation for a specific routine. Uses pin-unit retrieval, same as `deps`.

```
ll document DGESV
ll document DSYEV -k 8
```

| Option | Default | Purpose |
|--------|---------|---------|
| `--top-k`, `-k` | `5` | Number of chunks to retrieve |

**What the LLM generates:** Function signature, parameter table, computation description, 1-2 usage examples, and related routines. The `docs` mode gets 8192 max tokens (double the other modes) because parameter tables and examples need space.

**Why this exists as a separate command:** The docs mode prompt is tuned for structured output (tables, code blocks, cross-references). Running `ll query "document DGESV" --mode docs` would work but requires typing the mode flag and doesn't pin the unit. This command is the ergonomic shortcut for the common workflow of generating reference documentation for a single routine.

### `ll logic <question>`

Extracts mathematical or algorithmic logic from the codebase.

```
ll logic "How does LU factorization work?"
ll logic "What algorithm does DSYEV use for eigenvalue computation?"
```

| Option | Default | Purpose |
|--------|---------|---------|
| `--top-k`, `-k` | `5` | Number of chunks to retrieve |

**Difference from `explain`:** The LLM prompt focuses on the mathematical algorithm rather than the code structure. For "How does LU factorization work?", `explain` would describe the Fortran implementation (subroutine calls, parameter passing, error handling), while `logic` would describe the mathematical decomposition (row reduction, pivot selection, triangular systems).

**No pin-unit:** Unlike `deps` and `document`, `logic` takes a free-form question and uses the standard retrieval path (entity-aware or semantic-only). This is intentional — algorithmic questions often span multiple routines ("How does eigenvalue decomposition work?") rather than targeting one.

## Operations Commands

### `ll view <file_path>`

Displays the full source of a Fortran file with syntax highlighting and line numbers. The file path comes from query results (the "File" field in chunk metadata).

```
ll view dgesv.f
ll view VARIANTS/lu/LL/sgetrf.f
ll view dgesv.f --data-dir /path/to/lapack
```

| Option | Default | Purpose |
|--------|---------|---------|
| `--data-dir`, `-d` | `LAPACK_DATA_DIR` env var or `data/lapack` | Base directory for file lookups |

**File resolution:** Tries `<data-dir>/<file_path>` first, then `<data-dir>/SRC/<file_path>` as a fallback. Most LAPACK source files are in the `SRC/` subdirectory, so you can use the bare filename from query results without the `SRC/` prefix.

**Why this exists:** Query results show purpose, parameters, and call graphs — but sometimes you need to read the actual source. This command renders the full file with Fortran syntax highlighting (via Rich's Pygments integration) so you can inspect implementation details, read the full comment header, or trace algorithm logic line by line.

**No API calls required.** This command reads from the local filesystem only. It doesn't need Voyage, Pinecone, or OpenRouter credentials.

### `ll stats`

Shows Pinecone index statistics.

```
ll stats
```

**Output:**
```
Index Stats:
  Total vectors: 1847
  Dimension: 1024
```

**When to use:** After ingestion to verify all chunks were stored. After `batch-ingest --no-reset` to check the cumulative count. As a health check to confirm Pinecone connectivity.

### `ll serve`

Starts the FastAPI REST API server.

```
ll serve
ll serve --port 3000 --host 127.0.0.1
```

| Option | Default | Purpose |
|--------|---------|---------|
| `--port`, `-p` | `8000` | Port to bind to |
| `--host` | `0.0.0.0` | Host to bind to |

**Endpoints exposed:**

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check (`{"status": "ok"}`) |
| `/stats` | GET | Index statistics |
| `/search?question=...&top_k=5` | GET | Retrieval only (no LLM) |
| `/query` | POST | Retrieval + LLM answer generation |
| `/docs` | GET | Swagger UI (auto-generated by FastAPI) |

**The server is the same pipeline as the CLI** — it calls `retrieve()` and `generate_answer()` with the same code paths. The only difference is HTTP transport instead of terminal I/O. See `retrieval_strategy.md` for pipeline details.

**Swagger UI** at `http://localhost:8000/docs` provides an interactive API explorer with request/response schemas, try-it-out forms, and curl examples. This is built into FastAPI — no custom frontend code.

## Common Workflows

### First-time setup

```bash
# 1. Clone and install
pip install -e .

# 2. Set credentials in .env
echo "VOYAGE_API_KEY=..." >> .env
echo "PINECONE_API_KEY=..." >> .env
echo "OPENROUTER_API_KEY=..." >> .env

# 3. Ingest the LAPACK codebase
ll batch-ingest data/lapack

# 4. Verify
ll stats
```

### Investigating a routine

```bash
# What does it do?
ll query "what does DGETRF do?"

# What calls it and what does it call?
ll deps DGETRF

# Generate reference docs
ll document DGETRF

# Read the actual source
ll view dgetrf.f
```

### Exploring a concept

```bash
# Start broad
ll query "How does LAPACK compute eigenvalues?" --no-answer

# Check retrieval quality, then get the full answer
ll query "How does LAPACK compute eigenvalues?"

# Dive into the math
ll logic "What algorithm does DSYEV use for eigenvalue computation?"
```

### Debugging retrieval quality

```bash
# See raw retrieval results without LLM
ll query "routines that call XERBLA" --no-answer

# Increase candidate pool
ll query "routines that call XERBLA" --no-answer -k 10

# Check what's in the index
ll stats
```

## Architecture: How Commands Map to Code

```
CLI Command          → Retrieval Path    → LLM Mode         → Key File
─────────────────────────────────────────────────────────────────────────
ll ingest            → (no retrieval)    → (no LLM)         → rag/ingest.py
ll batch-ingest      → (no retrieval)    → (no LLM)         → rag/batch_ingest.py
ll query             → entity or semantic → configurable     → rag/retrieve.py + rag/generate.py
ll query --no-answer → entity or semantic → (no LLM)        → rag/retrieve.py
ll explain           → entity or semantic → explain          → rag/retrieve.py + rag/generate.py
ll deps              → pin-unit          → deps             → rag/retrieve.py + rag/generate.py
ll document          → pin-unit          → docs             → rag/retrieve.py + rag/generate.py
ll logic             → entity or semantic → business_logic   → rag/retrieve.py + rag/generate.py
ll view              → (no retrieval)    → (no LLM)         → (local filesystem)
ll stats             → (no retrieval)    → (no LLM)         → rag/storage.py
ll serve             → (all of above)    → (all of above)   → api/server.py
```

## Key Files

| File | Role |
|------|------|
| `src/legacylens/cli/main.py` | All 10 CLI commands, Typer app definition |
| `src/legacylens/api/server.py` | FastAPI server, REST endpoints |
| `src/legacylens/rag/retrieve.py` | Retrieval pipeline (entity detection, tiered merge, reranking, diversification) |
| `src/legacylens/rag/generate.py` | LLM answer generation (OpenRouter → Claude), system prompts, mode instructions |
| `src/legacylens/rag/ingest.py` | Synchronous ingestion pipeline (chunk → embed → upsert) |
| `src/legacylens/rag/batch_ingest.py` | Batch ingestion pipeline (Voyage Batch API) |
| `src/legacylens/rag/embeddings.py` | Voyage Code 3 client (embedding + reranking) |
| `src/legacylens/rag/storage.py` | Pinecone client (query, upsert, stats, delete) |
| `src/legacylens/config.py` | Environment variable loading |
