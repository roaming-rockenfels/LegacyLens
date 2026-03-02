# Risks & Open Questions

## Coverage Check
- [x] All core features from Q1 mapped to milestones
  - Ingestion → M1
  - Chunking → M1
  - Embeddings → M2
  - Vector storage → M2
  - Semantic search → M3
  - CLI interface → M3
  - Answer generation → M4
  - Code Explanation → M5
  - Dependency Mapping → M5
  - Documentation Gen → M5
  - Business Logic Extraction → M5
  - Deployment → M4
  - AI Cost Analysis → M6
  - RAG Architecture Doc → M6
  - Pre-Search Document → M6
- [x] All non-functional requirements addressed
  - Query latency <3s → M4 acceptance criteria
  - Retrieval precision >70% → M6 evaluation
  - Ingestion throughput 10K+ LOC <5min → M2 acceptance criteria
  - 100% file coverage → M2 acceptance criteria
  - Correct file/line references → M3 acceptance criteria
- [x] No scope creep detected

## Risks
| Risk | Impact | Likelihood | Mitigation |
|------|--------|-----------|------------|
| Voyage Code 2 API unavailable or rate-limited | High | Low | Fallback to OpenAI text-embedding-3-small (same 1536 dims, swap SDK) |
| LAPACK files with non-standard formatting break regex parser | Medium | Low | LAPACK is extremely consistent; fallback to tree-sitter-fortran |
| Fly.io deployment issues (cold starts, memory limits) | Medium | Medium | Test deploy early in M4; Fly.io free tier has 256MB RAM — monitor. Fallback: Railway |
| Pinecone free tier limits hit during ingestion | Medium | Low | Free tier supports ~100K vectors; LAPACK will produce ~2K chunks — well within limits |
| Retrieval precision below 70% target | High | Medium | Tune: increase top-k, add NL summary enrichment, try re-ranking with LLM. Test early in M3 |
| Claude Haiku 4.5 produces shallow Fortran explanations | Medium | Low | Upgrade to Sonnet for demo; Haiku is strong on code but test quality early |
| 24-hour MVP timeline too aggressive | High | Medium | Critical path is M1-M4 (20 hours of work). Buffer of 4 hours. If behind at hour 14, cut streaming |
| Voyage Code 2 underperforms on Fortran specifically | Medium | Medium | NL summary enrichment mitigates; test with 5 queries in M2 before full ingestion |

## Open Questions
- **Voyage Code 2 vs Voyage Code 3?**
  - Impact: Newer model may exist with better performance
  - When to resolve: Check Voyage docs before starting M2

- **LAPACK download method?**
  - Impact: Need to decide git clone vs tarball vs specific release
  - When to resolve: Beginning of M1

- **Fly.io Python runtime specifics?**
  - Impact: Dockerfile setup, Python version, memory allocation
  - When to resolve: Beginning of M4

- **Pinecone index configuration (metric, pod type)?**
  - Impact: Retrieval quality and latency
  - When to resolve: Beginning of M2; use cosine similarity, serverless index

## Plan Status
- [x] Plan finalized on 2026-03-02
