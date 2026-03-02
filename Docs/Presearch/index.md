# Presearch Index

| Document | Status | Summary |
|----------|--------|---------|
| Q1_requirements_scope.md | Complete | Scope lock: LAPACK Fortran, CLI-first, 4 code understanding features, Fly.io + Pinecone |
| Q2_embedding_model.md | Complete | Voyage Code 2 selected — code-optimized, 16K context, ~$0.006 total cost |
| Q2_llm_answer_generation.md | Complete | Claude Haiku 4.5 selected — best quality/cost at $0.0044/query, 200K context |
| Q2_rag_framework.md | Complete | Custom pipeline (no framework) — direct SDK calls, ~150-200 lines, cleanest MCP path |
| Q2_chunking_strategy.md | Complete | Regex-based, one chunk per file, NL summary enrichment, pluggable BaseChunker interface |
| presearch_checklist.md | Complete | All 16 Pre-Search sections answered (Phases 1-3) |
| risks_and_open_questions.md | Complete | 8 risks identified with mitigations, 4 open questions with resolution triggers |
| architecture.mermaid | Complete | System architecture: ingestion pipeline, query pipeline, pluggable chunker |
