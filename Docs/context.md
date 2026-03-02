# Session Context

## Project: LegacyLens
## Current Phase: CP4 Complete — Ready for Implementation
## Started: 2026-03-02

## Timeline
- MVP: 24 hours
- Final: 48 hours

## Decisions Made
- Codebase: LAPACK (Fortran), pluggable chunker for future COBOL support
- Backend: Python / FastAPI
- Vector DB: Pinecone (managed, free tier)
- Embedding: Voyage Code 2 (1536 dim, code-optimized)
- LLM: Claude Haiku 4.5 (upgrade to Sonnet for demo)
- Framework: Custom pipeline (no framework — direct SDK calls)
- Chunking: Regex-based, one chunk per file, NL summary enrichment
- Deployment: Fly.io (FastAPI backend) + CLI (pip-installable primary interface)
- Web demo: FastAPI Swagger UI
- Code Understanding Features: Explanation, Dependency Mapping, Documentation Gen, Business Logic Extraction

## Next Steps
- Begin M1: Project scaffold + LAPACK ingestion
- See Docs/Implementation/plan.md for full milestone breakdown
