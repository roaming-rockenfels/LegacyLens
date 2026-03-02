# Requirements & Scope

## Source
- PRD: G4 Week 3 - LegacyLens.pdf
- Date: 2026-03-02

## Core Features (Must-Have)

1. **Codebase Ingestion (LAPACK/Fortran)**: Ingest LAPACK Fortran library (10K+ LOC, 50+ files) with recursive file discovery, encoding normalization, and comment extraction
   - Acceptance: All .f/.f90 files discovered, parsed, and processed without errors

2. **Syntax-Aware Chunking (Pluggable)**: Split Fortran code at SUBROUTINE/FUNCTION/PROGRAM boundaries with a pluggable chunker interface so COBOL support can be added post-MVP
   - Acceptance: Each chunk maps to a single logical unit (function/subroutine) with metadata (file path, line numbers, function name)

3. **Embedding Generation**: Vectorize all chunks using a chosen embedding model (to be decided in CP2)
   - Acceptance: Every chunk has a corresponding vector stored with metadata

4. **Vector Storage (Pinecone)**: Store embeddings + metadata in Pinecone managed vector DB
   - Acceptance: All chunks queryable via Pinecone API, metadata filters work

5. **Semantic Search**: Top-k similarity search across the indexed codebase
   - Acceptance: Queries return relevant chunks with >70% precision in top-5

6. **CLI Query Interface**: Python CLI (click/typer) for natural language questions about the codebase
   - Acceptance: Developer can `pip install` and run queries from terminal, results show code snippets with file/line references and relevance scores

7. **Answer Generation**: LLM synthesizes human-readable responses from retrieved context
   - Acceptance: Responses cite specific files/lines, explain code in plain English

8. **Code Explanation**: Explain what a function/subroutine does in plain English
   - Acceptance: Accurate plain-English summaries for any retrieved function

9. **Dependency Mapping**: Show what calls what, data flow between modules
   - Acceptance: Given a function, list its callers and callees with file references

10. **Documentation Generation**: Generate documentation for undocumented code
    - Acceptance: Produce docstring-style documentation for any function on demand

11. **Business Logic Extraction**: Identify and explain business rules embedded in code
    - Acceptance: Extract and summarize computational logic / domain rules from retrieved code

12. **Deployment (Fly.io)**: FastAPI backend deployed on Fly.io, publicly accessible with Swagger UI
    - Acceptance: API reachable via public URL, Swagger UI functional for web-based demo

13. **AI Cost Analysis**: Track dev/testing spend + production projections at 100/1K/10K/100K users
    - Acceptance: Document with actual spend breakdown and projected costs with assumptions

14. **RAG Architecture Doc**: 1-2 page writeup covering DB selection, embedding strategy, chunking approach, retrieval pipeline, failure modes, performance results
    - Acceptance: Completed document following PRD template

15. **Pre-Search Document**: Completed checklist covering Phases 1-3 from PRD appendix
    - Acceptance: All 16 sections answered

## Nice-to-Have (Deferred)
- Re-ranking layer (optional per PRD)
- COBOL chunker (pluggable interface built in MVP, COBOL implementation post-MVP)
- More than 4 code understanding features (Translation Hints, Pattern Detection, Impact Analysis, Bug Pattern Search)
- Hierarchical chunking (file → section → function granularity)
- Query expansion / multi-query retrieval
- Web frontend beyond Swagger UI
- MCP server interface (architecture supports it, implementation deferred)

## Non-Functional Requirements
- Performance: Query latency <3 seconds end-to-end
- Performance: Retrieval precision >70% relevant chunks in top-5
- Performance: Ingestion throughput 10,000+ LOC in <5 minutes
- Coverage: 100% of files indexed
- Accuracy: Correct file/line references in answers
- Security: API keys managed via environment variables, not committed to repo

## Explicit Exclusions
- No web frontend (Swagger UI is sufficient for demo; CLI is the primary interface)
- No Vercel deployment
- No code editing/modification capabilities — read-only analysis tool
- No proprietary codebase support (all targets are open source)
- No multi-tenant/multi-user features — single developer tool
- No real-time incremental indexing for MVP (batch ingestion only)

## Open Questions (Resolved)
- Q: Which codebase? → A: LAPACK (Fortran), with pluggable chunker design for future COBOL support
- Q: Backend language? → A: Python/FastAPI
- Q: Which 4 code understanding features? → A: Explanation, Dependency Mapping, Documentation Gen, Business Logic Extraction
- Q: Deployment platform? → A: Fly.io (backend) + Pinecone (managed vector DB), CLI as primary interface
- Q: Web frontend needed? → A: No, Swagger UI + CLI. Geared toward developer use during planning/implementation.

## Scope Confirmation
- [x] User confirmed scope on 2026-03-02
