# Pre-Search Checklist — LegacyLens

Completed: 2026-03-02

---

## Phase 1: Define Your Constraints

### 1. Scale & Load Profile

- **How large is your target codebase?**
  LAPACK (Fortran): ~500K LOC across ~1,850 files (SRC/ + BLAS/SRC/). Well above the 10K LOC / 50 file minimum.

- **Expected query volume?**
  Development: ~200 queries during sprint. Production demo: low volume (evaluator testing). Production projection: 5-50 queries/user/day.

- **Batch ingestion or incremental updates?**
  Batch ingestion for MVP. LAPACK is a stable library — no incremental update requirement. Full re-ingestion takes <5 minutes.

- **Latency requirements for queries?**
  <3 seconds end-to-end (PRD requirement). Breakdown budget: ~200ms embedding, ~500ms Pinecone search, ~1-2s LLM generation.

### 2. Budget & Cost Ceiling

- **Vector database hosting costs?**
  Pinecone free tier (Starter plan): $0/month. Supports up to ~100K vectors. LAPACK produces ~2K chunks — well within limits.

- **Embedding API costs?**
  Voyage Code 2: ~$0.12/1M tokens. Full LAPACK ingestion (~50K tokens): ~$0.006. Negligible.

- **LLM API costs for answer generation?**
  Claude Haiku 4.5: ~$0.0044/query (3K input + 500 output tokens). Development budget (~200 queries): ~$0.88. Upgrade to Sonnet for demo: ~$0.0165/query.

- **Where will you trade money for time?**
  Managed services (Pinecone, Fly.io) over self-hosted to minimize ops overhead in a 48-hour sprint. API-based embedding over local models to avoid Docker image bloat.

### 3. Time to Ship

- **MVP timeline?**
  24 hours. Hard gate — all 9 MVP requirements must pass.

- **Final timeline?**
  48 hours. All submission deliverables due.

- **Which features are must-have vs nice-to-have?**
  Must-have: Full RAG pipeline (ingest → embed → store → retrieve → generate), CLI, deployment, 4 code understanding features.
  Nice-to-have: Re-ranking, COBOL chunker, web frontend, MCP server, hierarchical chunking.

- **Framework learning curve acceptable?**
  No. Decision: custom pipeline (no framework). LangChain/LlamaIndex learning curve (1-2 days) is too expensive in a 24-hour MVP window. Direct SDK calls are faster.

### 4. Data Sensitivity

- **Is the codebase open source or proprietary?**
  Open source. LAPACK is freely available under a modified BSD license from netlib.org.

- **Can you send code to external APIs?**
  Yes. All code is public. No restrictions on sending to Voyage AI, Anthropic, or Pinecone APIs.

- **Data residency requirements?**
  None. Student project with open-source code.

### 5. Team & Skill Constraints

- **Familiarity with vector databases?**
  Moderate. Have used Railway before; Pinecone is new (chosen deliberately to broaden experience).

- **Experience with RAG frameworks?**
  Limited. Decision to use custom pipeline avoids framework learning curve entirely.

- **Comfort with the target legacy language?**
  Low familiarity with Fortran. Mitigated by: LAPACK's excellent English-language comment headers, regex-based parsing (no deep Fortran knowledge needed), and LLM-assisted code explanation.

---

## Phase 2: Architecture Discovery

### 6. Vector Database Selection

- **Managed vs self-hosted?**
  Managed (Pinecone). Eliminates ops overhead for a 48-hour sprint.

- **Filtering and metadata requirements?**
  Yes. Need to filter by: subroutine_name, precision (s/d/c/z), category (GE/SY/PO), file_path. Pinecone supports metadata filtering natively.

- **Hybrid search (vector + keyword) needed?**
  Not for MVP. Pure vector search with NL summary enrichment provides sufficient retrieval quality. Keyword search could be added post-MVP if precision is below target.

- **Scaling characteristics?**
  Not a concern. ~2K vectors is trivial for Pinecone. Free tier supports 100K+ vectors.

- **Selection rationale:**
  Pinecone chosen for: zero-ops managed service, free tier covers project scope, native metadata filtering, simple Python SDK, broadens experience beyond Railway.

  See: `Docs/Presearch/Q2_embedding_model.md` (Pinecone integration discussed)

### 7. Embedding Strategy

- **Code-specific vs general-purpose model?**
  Code-specific: Voyage Code 2. 10-20% better retrieval precision on code search benchmarks (CosQA, CodeSearchNet) vs general-purpose OpenAI models.

- **Dimension size tradeoffs?**
  1536 dimensions. Matches Pinecone's standard indexing. Same as OpenAI text-embedding-3-small (easy fallback). Lower dims (1024, 768) lose code nuance; higher dims (3072) add storage cost with marginal quality gain.

- **Local vs API-based embedding?**
  API-based (Voyage Code 2). Cost is negligible (~$0.006 total). Local models (CodeBERT) have 512-token context limits that truncate code chunks, and no Fortran training data.

- **Batch processing approach?**
  Voyage supports 128 texts per API call. Process LAPACK in batches of 128 chunks, upsert to Pinecone after each batch. Full ingestion: ~15 API calls.

- **Fortran gap mitigation:**
  No embedding model is trained on Fortran. Mitigated by: prepending natural-language summary lines extracted from LAPACK's comment headers (e.g., "DGESV: Computes the solution to a real system of linear equations A * X = B"). LAPACK's rich comments are effectively English text.

  See: `Docs/Presearch/Q2_embedding_model.md`

### 8. Chunking Approach

- **Syntax-aware vs fixed-size?**
  Syntax-aware. LAPACK follows one-subroutine-per-file, so each file = one chunk at the SUBROUTINE/FUNCTION boundary. Regex-based parser (not AST) because LAPACK's formatting is extremely consistent.

- **Optimal chunk size for your embedding model?**
  Voyage Code 2 supports up to 16,000 tokens. 85%+ of LAPACK subroutines fit in one chunk (<16K tokens). The ~3% that exceed the limit are split at comment-delimited section boundaries with preamble inclusion.

- **Overlap strategy?**
  Not needed for function-level chunks — each subroutine is a self-contained compilation unit. For the ~3% of split long routines: include the comment header (preamble) in every sub-chunk + 10-20 lines overlap between code sections.

- **Metadata to preserve?**
  13 fields per chunk: file_path, start_line, end_line, subroutine_name, unit_type, precision, parameters, calls (CALL statements), external_deps (EXTERNAL declarations), category (prefix mapping), purpose (extracted summary), chunk_index, chunk_total.

- **Pluggable interface:**
  BaseChunker ABC with ChunkerRegistry. Fortran chunker implemented for MVP; COBOL chunker uses same interface (PARAGRAPH/SECTION boundaries, PERFORM instead of CALL, COPY instead of USE).

  See: `Docs/Presearch/Q2_chunking_strategy.md`

### 9. Retrieval Pipeline

- **Top-k value for similarity search?**
  k=5 for default queries (matches PRD precision target: >70% relevant in top-5). k=10 for dependency mapping and pattern-oriented queries.

- **Re-ranking approach?**
  Not for MVP (PRD marks as optional). If retrieval precision is below 70%, add LLM-based re-ranking as a post-MVP improvement.

- **Context window management?**
  Claude Haiku 4.5 has 200K token context. Typical query assembles 5 chunks × ~2K tokens = ~10K tokens + system prompt (~500 tokens) + query. Well within limits. No truncation logic needed.

- **Multi-query or query expansion?**
  Not for MVP. Single query → single embedding → single Pinecone search. Query expansion (generating multiple query phrasings) is a post-MVP enhancement.

### 10. Answer Generation

- **Which LLM for synthesis?**
  Claude Haiku 4.5 for development and production. Claude Sonnet 4 for demo video (higher quality, same SDK — just change model name).

- **Prompt template design?**
  System prompt establishes role ("You are a Fortran code analysis expert"). User message contains: the query + retrieved chunks with metadata (file path, line numbers, function name). Prompt instructs LLM to cite specific files/lines and explain in plain English.

- **Citation/reference formatting?**
  Each answer must include: file path, line number range, subroutine name. Format: `[SRC/dgesv.f:45-120] DGESV`. Enforced via system prompt instructions.

- **Streaming vs batch response?**
  Streaming. Anthropic SDK supports `client.messages.stream()`. FastAPI uses `StreamingResponse`. CLI prints tokens as they arrive. Essential for the <3s perceived latency target.

  See: `Docs/Presearch/Q2_llm_answer_generation.md`

### 11. Framework Selection

- **LangChain vs LlamaIndex vs custom?**
  Custom pipeline (no framework). Rationale:
  1. Non-OpenAI stack (Voyage + Claude) means frameworks add integration friction — all tutorials assume OpenAI.
  2. The full RAG pipeline is ~150-200 lines of direct SDK calls — less code than framework configuration.
  3. Custom Fortran chunker works natively without conforming to framework interfaces.
  4. MCP server upgrade path is cleanest with plain Python functions.
  5. 24-hour MVP leaves no room for framework debugging.

- **Evaluation and observability needs?**
  DIY: log query latency (`time.perf_counter()`), token counts (from SDK response objects), Pinecone retrieval scores. ~20 lines of logging code. Test harness with 6-10 known queries. No need for LangSmith/LlamaTrace.

- **Integration requirements?**
  Three Python packages: `voyageai`, `pinecone`, `anthropic`. No framework wrappers. Clean composition in a single `rag.py` module.

  See: `Docs/Presearch/Q2_rag_framework.md`

---

## Phase 3: Post-Stack Refinement

### 12. Failure Mode Analysis

- **What happens when retrieval finds nothing relevant?**
  If top similarity score < threshold (e.g., 0.3), return "No relevant code found for this query" instead of hallucinating an answer. Log the failed query for analysis.

- **How to handle ambiguous queries?**
  LLM system prompt instructs: "If the query is ambiguous, list the possible interpretations and ask the user to clarify." For CLI, re-prompt the user. For API, return a clarification response.

- **Rate limiting and error handling?**
  Voyage API: retry with exponential backoff (3 attempts). Pinecone: retry on transient errors. Anthropic: handle rate limits (429) with backoff. All errors logged with query context for debugging.

  See: `Docs/Presearch/risks_and_open_questions.md`

### 13. Evaluation Strategy

- **How to measure retrieval precision?**
  Manual evaluation on 10+ test queries. For each query, judge whether retrieved chunks in top-5 are relevant (binary yes/no). Precision = relevant / total retrieved. Target: >70%.

- **Ground truth dataset for testing?**
  PRD provides 6 test scenarios. Add 4-6 more domain-specific queries (e.g., "Which routines compute eigenvalues?", "How does DGETRF handle pivoting?"). Manually label expected relevant subroutines.

- **User feedback collection?**
  Not for MVP/sprint. Post-sprint: add thumbs-up/down on CLI responses, log to file for analysis.

### 14. Performance Optimization

- **Caching strategy for embeddings?**
  Query embedding cache: LRU cache on query text → embedding vector. Avoids re-embedding repeated queries. Implementation: Python `functools.lru_cache` or dict-based cache. Saves ~200ms per cached query.

- **Index optimization?**
  Pinecone handles index optimization automatically (managed service). Use cosine similarity metric. Serverless index type for simplicity.

- **Query preprocessing?**
  Minimal for MVP: strip whitespace, normalize case. Post-MVP: extract entities (function names) from queries for metadata-filtered search.

### 15. Observability

- **Logging for debugging retrieval issues?**
  Structured logging (Python `logging` module) with: query text, embedding latency, Pinecone search latency, top-k scores, LLM generation latency, total latency, token counts (input/output).

- **Metrics to track?**
  - Query latency (p50, p95)
  - Retrieval scores (top-1, top-5 similarity)
  - LLM token usage (input + output per query)
  - Error rate by component (embedding, search, generation)
  - Total API cost (running sum)

- **Alerting needs?**
  None for sprint. Post-sprint: alert on latency >5s or error rate >5%.

### 16. Deployment & DevOps

- **CI/CD for index updates?**
  Not for sprint. Ingestion is run manually via CLI command (`legacylens ingest <path>`). Post-sprint: GitHub Action to re-index on code changes.

- **Environment management?**
  `.env` file for local development (not committed). Fly.io secrets for production (VOYAGE_API_KEY, PINECONE_API_KEY, ANTHROPIC_API_KEY). `.env.example` in repo for documentation.

- **Secrets handling for API keys?**
  Three API keys needed: Voyage AI, Pinecone, Anthropic. Loaded via `os.environ` / `python-dotenv`. Never committed to repo. Fly.io `flyctl secrets set` for production.

- **Deployment target:**
  Fly.io (FastAPI backend). Dockerfile-based deployment. Python 3.11+. ~256MB RAM (free tier). Health check endpoint at GET /health.
