# CLI Reference

## Overview

LegacyLens ships a single CLI binary `legacylens` (aliased to `ll` for convenience) built on Typer. Every command follows the same pattern: accept a natural-language question or routine name, run the retrieval pipeline against Pinecone, optionally pass retrieved chunks to an LLM for answer generation, and render output to the terminal via Rich.

There are 13 commands organized into four groups:

| Group | Commands | What they do |
|-------|----------|--------------|
| **Ingestion** | `ingest`, `batch-ingest` | Load source code into the vector database |
| **Query** | `query`, `explain`, `deps`, `document`, `logic`, `chat` | Ask questions, get answers |
| **Source Management** | `sources list`, `sources info`, `sources rm` | Track and manage ingested codebases |
| **Operations** | `view`, `stats`, `serve` | Inspect source, check index health, start API server |

## Supported Languages

LegacyLens supports multiple languages through its pluggable chunker architecture:

| Language | Extensions | Chunker Strategy |
|----------|-----------|------------------|
| **Fortran** | `.f`, `.f90`, `.f95`, `.f03`, `.for` | Regex-based parsing, LAPACK-aware metadata (precision, category, routine role) |
| **Python** | `.py`, `.pyi` | AST-based parsing, one chunk per top-level function/class, large class splitting |

When ingesting a directory, file extensions are scanned to auto-detect languages. The correct chunker is dispatched per-file, and language metadata is stored with each source for retrieval strategy selection at query time.

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
| `CODEBASE_DATA_DIR` | Falls back to `LAPACK_DATA_DIR`, then `"data/lapack"` | Base path for `view` command file lookups |
| `LAPACK_DATA_DIR` | `"data/lapack"` | Legacy base path (still supported) |

## Ingestion Commands

These load source code into Pinecone. You run them once (or when the codebase changes), not per-query. Each ingest command now **registers a source** — tracking the codebase name, detected languages, file extensions, and chunk count in `~/.legacylens/sources.json`.

### `ll ingest <path>`

Synchronous ingestion using the real-time Voyage API. Scans `<path>` for supported files, chunks them with the appropriate language chunker, embeds in small batches, and upserts to Pinecone in a named namespace.

```
ll ingest data/lapack --name lapack
ll ingest data/pydantic/pydantic --name pydantic
ll ingest src/ --name myproject
```

| Option | Default | Purpose |
|--------|---------|---------|
| `--name`, `-n` | Folder basename | Source name for the registry and Pinecone namespace |

**Pipeline:** scan directory → detect languages → chunk files → embed in batches of 3 → upsert to Pinecone namespace → register source.

**Pre-scan:** Before chunking, the directory is scanned for file extensions. If no supported files are found, the command errors out early with a helpful message listing supported extensions.

**Namespace isolation:** Each source gets its own Pinecone namespace (derived from the source name). This means multiple codebases can be ingested into the same index without interference.

**Rate limiting:** The free Voyage tier allows 3 requests/minute and 10K tokens/minute. The default settings (`EMBED_BATCH_SIZE=3`, `EMBED_DELAY=25s`) stay within these limits. For large codebases, use `batch-ingest` instead.

**Output:**
```
  Detected languages: python
  File extensions: {'.py': 35}
Step 1/3: Chunking files in data/pydantic/pydantic...
  Found 1052 chunks in 0.2s
Step 2/3: Generating embeddings with Voyage Code 3...
  Generated 1052 embeddings in ...
Step 3/3: Upserting to Pinecone...
  Upserted 1052 vectors in 7.6s
Ingestion complete!
  Registered source: pydantic (namespace: pydantic)
```

### `ll batch-ingest <path>`

Asynchronous ingestion using the Voyage Batch API. Writes all chunks to a JSONL file, uploads it, creates a batch job, polls for completion, then downloads results and upserts to Pinecone.

```
ll batch-ingest data/lapack --name lapack
ll batch-ingest data/pydantic/pydantic --name pydantic --no-reset
```

| Option | Default | Purpose |
|--------|---------|---------|
| `--no-reset` | `False` | Skip deleting and recreating the Pinecone index |
| `--name`, `-n` | Folder basename | Source name for the registry and Pinecone namespace |

**Pipeline:** scan directory → detect languages → chunk files → write JSONL → upload to Voyage Files API → create batch job → poll every 10-60s → download embeddings → upsert to Pinecone namespace → register source.

**Why this exists:** The Batch API processes all embeddings server-side with no per-request rate limits. A 1,052-chunk Python codebase (Pydantic) completes in ~9 minutes. The full LAPACK codebase (~1,850 chunks) completes in ~10-15 minutes instead of 2.5 hours. The tradeoff is that it's all-or-nothing — you can't incrementally add files.

**Default behavior deletes the Pinecone index** before upserting, ensuring a clean slate. Use `--no-reset` to append to an existing index.

## Source Management Commands

Sources track ingested codebases. Each source has a name, path, detected languages, extension counts, chunk count, and timestamp. The registry lives at `~/.legacylens/sources.json`.

### `ll sources list`

Lists all registered sources as a table.

```
ll sources list
```

**Output:**
```
         Ingested Sources
┏━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┓
┃ Name     ┃ Path      ┃ Languages ┃ Chunks ┃ Ingested At         ┃
┡━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━┩
│ lapack   │ /path/... │ fortran   │   1847 │ 2026-03-05T20:00:00 │
│ pydantic │ /path/... │ python    │   1052 │ 2026-03-05T21:44:54 │
└──────────┴───────────┴───────────┴────────┴─────────────────────┘
```

### `ll sources info <name>`

Shows detailed information about a specific source, including per-extension file counts.

```
ll sources info pydantic
```

**Output:**
```
Source: pydantic
  Path:       /Users/.../data/pydantic/pydantic
  Namespace:  pydantic
  Languages:  python
  Chunks:     1052
  Ingested:   2026-03-05T21:44:54.123456+00:00
  Extensions:
    .py: 35 files
```

### `ll sources rm <name>`

Removes a source: deletes all vectors in its Pinecone namespace and removes the registry entry.

```
ll sources rm pydantic
```

**Output:**
```
  Deleted Pinecone namespace: pydantic
  Removed source: pydantic
```

## Query Commands

All query commands follow the same internal flow:

```
question → retrieve(question, top_k, namespace, languages) → [optional] generate_answer(question, results, mode) → display
```

The `--source` flag selects which namespace to query and which retrieval strategy to use (based on the source's stored language metadata). Without `--source`, the default Pinecone namespace is queried using the Fortran retrieval strategy (backward-compatible).

### `ll query <question>`

The primary command. Retrieves relevant chunks, displays them, then generates an LLM answer.

```
ll query "How does LAPACK solve linear equations?"
ll query "how does field validation work?" --source pydantic
ll query "routines that call XERBLA" --no-answer
ll query "what does DGETRF do?" -k 10 -m deps --source lapack
```

| Option | Default | Purpose |
|--------|---------|---------|
| `--top-k`, `-k` | `5` | Number of chunks to retrieve |
| `--mode`, `-m` | `"explain"` | LLM response mode (see below) |
| `--no-answer` | `False` | Show retrieved chunks only, skip the LLM call |
| `--show-code` / `--no-code` | `False` | Show source code snippets with syntax highlighting and line numbers |
| `--source`, `-s` | `None` | Source name to query (uses its namespace + language-aware retrieval) |

**Response modes** control the system prompt sent to Claude:

| Mode | Instruction given to LLM | Max tokens |
|------|-------------------------|------------|
| `explain` | "Provide a clear explanation of what this code does, how it works, and why." | 4096 |
| `deps` | "Focus on the dependency relationships: what calls what, what external libraries are used, and the data flow between routines." | 4096 |
| `docs` | "Generate concise modern documentation for the code, including function signature, parameter table, computation description, 1-2 usage examples, and related routines." | 8192 |
| `business_logic` | "Extract and explain the core business/mathematical logic, the algorithm being implemented, and its practical applications." | 4096 |

**Language-aware retrieval:** When `--source` points to a Python source, the retrieval pipeline uses Python entity extraction (snake_case functions, CamelCase classes, dotted module paths) instead of Fortran entity extraction (SDCZ precision prefixes). Diversification skips LAPACK precision-variant deduplication for Python chunks. Code snippets in results use the correct syntax highlighting (Python or Fortran).

### `ll explain <question>`

Shortcut for `ll query <question> --mode explain`. Retrieves chunks and generates an explanation.

```
ll explain "How does partial pivoting work in DGETRF?"
ll explain "how does BaseModel validation work?" --source pydantic
```

| Option | Default | Purpose |
|--------|---------|---------|
| `--top-k`, `-k` | `5` | Number of chunks to retrieve |
| `--source`, `-s` | `None` | Source name to query |

### `ll deps <unit_name>`

Shows dependency relationships for a specific routine or function. Uses the **pin-unit retrieval path** — the named unit is guaranteed to appear first in results, with semantically related chunks filling remaining slots.

```
ll deps DGESV
ll deps DGETRF -k 15 --source lapack
ll deps BaseModel --source pydantic
```

| Option | Default | Purpose |
|--------|---------|---------|
| `--top-k`, `-k` | `10` | Number of chunks to retrieve (default is higher than other commands) |
| `--source`, `-s` | `None` | Source name to query |

**How it works internally:**
1. Constructs the question: `"What are the dependencies and call relationships of {unit_name}? What does it call and what calls it?"`
2. Calls `retrieve(question, top_k=10, pin_unit=unit_name, namespace=..., languages=...)`
3. The pin-unit path queries `filter={"unit_name": "..."}` first (gets the unit + any split chunks), then fills remaining slots with semantic search.
4. Passes results to the LLM in `deps` mode.

### `ll document <unit_name>`

Generates modern documentation for a specific routine or class. Uses pin-unit retrieval.

```
ll document DGESV
ll document BaseModel --source pydantic
```

| Option | Default | Purpose |
|--------|---------|---------|
| `--top-k`, `-k` | `5` | Number of chunks to retrieve |
| `--source`, `-s` | `None` | Source name to query |

### `ll logic <question>`

Extracts mathematical or algorithmic logic from the codebase.

```
ll logic "How does LU factorization work?"
ll logic "how does JSON schema generation work?" --source pydantic
```

| Option | Default | Purpose |
|--------|---------|---------|
| `--top-k`, `-k` | `5` | Number of chunks to retrieve |
| `--source`, `-s` | `None` | Source name to query |

### `ll chat`

Interactive multi-turn chat about the codebase. First query always retrieves; subsequent queries let the LLM decide whether to search again or answer from conversation history.

```
ll chat
ll chat --source pydantic
ll chat -k 10
```

| Option | Default | Purpose |
|--------|---------|---------|
| `--top-k`, `-k` | `5` | Number of chunks to retrieve |
| `--source`, `-s` | `None` | Source name to query |

**Session commands:** Type `reset` to clear conversation history, `exit` or `quit` to end.

## Operations Commands

### `ll view <file_path>`

Displays the full source of a file with syntax highlighting and line numbers. The file path comes from query results (the "File" field in chunk metadata).

```
ll view dgesv.f
ll view connection.py --data-dir /path/to/project
```

| Option | Default | Purpose |
|--------|---------|---------|
| `--data-dir`, `-d` | `CODEBASE_DATA_DIR` env var | Base directory for file lookups |

**File resolution:** Tries `<data-dir>/<file_path>` first, then `<data-dir>/SRC/<file_path>` as a fallback. Syntax highlighting is auto-detected from the file extension (Fortran for `.f`/`.f90`, Python for `.py`, plain text otherwise).

### `ll stats`

Shows Pinecone index statistics.

```
ll stats
```

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
| `/chat` | POST | Multi-turn chat (session-based) |
| `/chat/stream` | POST | Streaming chat (SSE) |
| `/chat/{session_id}` | DELETE | Delete a chat session |
| `/docs` | GET | Swagger UI (auto-generated by FastAPI) |

## Common Workflows

### First-time setup

```bash
# 1. Clone and install
pip install -e .

# 2. Set credentials in .env
echo "VOYAGE_API_KEY=..." >> .env
echo "PINECONE_API_KEY=..." >> .env
echo "OPENROUTER_API_KEY=..." >> .env

# 3. Ingest a codebase
ll batch-ingest data/lapack --name lapack
ll batch-ingest data/pydantic/pydantic --name pydantic --no-reset

# 4. Verify
ll sources list
ll stats
```

### Investigating a Fortran routine

```bash
ll query "what does DGETRF do?" --source lapack
ll deps DGETRF --source lapack
ll document DGETRF --source lapack
ll view dgetrf.f
```

### Investigating a Python codebase

```bash
ll query "how does field validation work?" --source pydantic
ll deps BaseModel --source pydantic
ll document BaseModel --source pydantic
ll logic "how does JSON schema generation work?" --source pydantic
```

### Multi-source exploration

```bash
# List what's ingested
ll sources list

# Query specific sources
ll query "how are errors handled?" --source lapack
ll query "how are errors handled?" --source pydantic

# Interactive chat scoped to a source
ll chat --source pydantic
```

### Debugging retrieval quality

```bash
# See raw retrieval results without LLM
ll query "routines that call XERBLA" --no-answer --source lapack

# Increase candidate pool
ll query "routines that call XERBLA" --no-answer -k 10 --source lapack

# Check what's in the index
ll stats
ll sources info lapack
```

### Managing sources

```bash
# List all sources
ll sources list

# Detailed view
ll sources info pydantic

# Remove a source (deletes vectors + registry entry)
ll sources rm pydantic
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
ll chat              → LLM-driven        → explain          → rag/session.py + rag/retrieve.py
ll sources list      → (no retrieval)    → (no LLM)         → sources.py
ll sources info      → (no retrieval)    → (no LLM)         → sources.py
ll sources rm        → (no retrieval)    → (no LLM)         → sources.py + rag/storage.py
ll view              → (no retrieval)    → (no LLM)         → (local filesystem)
ll stats             → (no retrieval)    → (no LLM)         → rag/storage.py
ll serve             → (all of above)    → (all of above)   → api/server.py
```

## Evaluation Script

The retrieval evaluation script (`evals/eval_retrieval.py`) runs 15 golden queries against the live Pinecone index and reports per-query, per-category, and aggregate metrics.

```bash
# Baseline run (no PASS/FAIL gating)
PYTHONPATH=src .venv/bin/python evals/eval_retrieval.py --threshold 0.0

# With markdown output for README
PYTHONPATH=src .venv/bin/python evals/eval_retrieval.py --threshold 0.0 --markdown

# With end-to-end latency (retrieval + LLM generation)
PYTHONPATH=src .venv/bin/python evals/eval_retrieval.py --threshold 0.0 --markdown --e2e
```

| Option | Default | Purpose |
|--------|---------|---------|
| `--golden` | `evals/golden_queries.json` | Path to golden query set JSON |
| `--threshold` | `0.7` | Minimum aggregate recall for PASS (exit code 0) |
| `--top-k` | per-query (default 5) | Override top_k for all queries |
| `--no-cache` | `False` | Disable embedding cache (always call Voyage API) |
| `--markdown` | `False` | Print a markdown-formatted category table suitable for pasting into README |
| `--e2e` | `False` | Also measure end-to-end latency (retrieval + LLM generation per query) |

**Metrics reported:** Precision@k, Recall@k, MRR (Mean Reciprocal Rank), Hit Rate, Retrieval Latency, and optionally E2E Latency.

## Key Files

| File | Role |
|------|------|
| `src/legacylens/cli/main.py` | All CLI commands, Typer app definition, sources subcommands |
| `src/legacylens/api/server.py` | FastAPI server, REST endpoints |
| `src/legacylens/sources.py` | Source registry (scan, register, list, get, remove) |
| `src/legacylens/chunkers/base.py` | BaseChunker ABC, ChunkerRegistry, Chunk/ChunkMetadata dataclasses |
| `src/legacylens/chunkers/fortran.py` | Fortran chunker (regex-based, LAPACK-aware) |
| `src/legacylens/chunkers/python.py` | Python chunker (AST-based, class splitting) |
| `src/legacylens/rag/retrieve.py` | Retrieval pipeline (language-aware entity detection, tiered merge, reranking, diversification) |
| `src/legacylens/rag/generate.py` | LLM answer generation (OpenRouter → Claude), system prompts, mode instructions |
| `src/legacylens/rag/ingest.py` | Synchronous ingestion pipeline (chunk → embed → upsert) |
| `src/legacylens/rag/batch_ingest.py` | Batch ingestion pipeline (Voyage Batch API) |
| `src/legacylens/rag/embeddings.py` | Voyage Code 3 client (embedding + reranking) |
| `src/legacylens/rag/storage.py` | Pinecone client (query, upsert, stats, delete, namespace operations) |
| `src/legacylens/rag/source_reader.py` | Source file reader for `view` and `--show-code` |
| `src/legacylens/rag/session.py` | Multi-turn chat session with LLM-driven retrieval |
| `src/legacylens/config.py` | Environment variable loading |
| `evals/eval_retrieval.py` | Evaluation script with category breakdown and latency tracking |
| `evals/golden_queries.json` | 15 golden queries across 5 categories |
