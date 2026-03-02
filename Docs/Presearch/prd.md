# LegacyLens — Building RAG Systems for Legacy Enterprise Codebases

Source: G4 Week 3 - LegacyLens.pdf
Date: 2026-03-02

## Background

Enterprise systems running on COBOL, Fortran, and other legacy languages power critical infrastructure: banking transactions, insurance claims, government services, and scientific computing. These codebases contain decades of business logic, but few engineers understand them.

Build a RAG-powered system that makes large legacy codebases queryable and understandable. Work with real open source enterprise projects, implementing retrieval pipelines that help developers navigate unfamiliar code through natural language.

Focus: RAG architecture — vector databases, embedding strategies, chunking approaches, and retrieval pipelines that work on complex codebases.

## Project Overview

Sprint with two deadlines:
- **MVP:** 24 hours — Basic RAG pipeline working
- **Final:** 48 hours — Polish, documentation, deployment

Gate: Behavioral and technical interviews required for Austin admission.

## MVP Requirements (24 Hours) — Hard Gate

- [ ] Ingest at least one legacy codebase (COBOL, Fortran, or similar)
- [ ] Chunk code files with syntax-aware splitting
- [ ] Generate embeddings for all chunks
- [ ] Store embeddings in a vector database
- [ ] Implement semantic search across the codebase
- [ ] Natural language query interface (CLI or web)
- [ ] Return relevant code snippets with file/line references
- [ ] Basic answer generation using retrieved context
- [ ] Deployed and publicly accessible

## Target Codebases (Choose ONE)

| Project | Language | Description |
|---------|----------|-------------|
| GnuCOBOL | COBOL | Open source COBOL compiler |
| GNU Fortran (gfortran) | Fortran | Fortran compiler in GCC |
| LAPACK | Fortran | Linear algebra library |
| BLAS | Fortran | Basic linear algebra subprograms |
| OpenCOBOL Contrib | COBOL | Sample COBOL programs and utilities |
| Custom proposal | Any legacy | Get approval before starting |

Minimum codebase size: 10,000+ LOC across 50+ files.

## Core RAG Infrastructure

### Ingestion Pipeline
- File Discovery: Recursively scan codebase, filter by file extension
- Preprocessing: Handle encoding issues, normalize whitespace, extract comments
- Chunking: Syntax-aware splitting (functions, paragraphs, sections)
- Metadata Extraction: File path, line numbers, function names, dependencies
- Embedding Generation: Generate vectors for each chunk with chosen model
- Storage: Insert into vector database with metadata

### Retrieval Pipeline
- Query Processing: Parse natural language, extract intent and entities
- Embedding: Convert query to vector using same model as ingestion
- Similarity Search: Find top-k most similar chunks
- Re-ranking: Optional — reorder results by relevance score
- Context Assembly: Combine retrieved chunks with surrounding context
- Answer Generation: LLM generates response using retrieved context

### Chunking Strategies
- Function-level: Each function/subroutine as a chunk
- Paragraph-level (COBOL): COBOL PARAGRAPH as natural boundary
- Fixed-size + overlap: Fallback for unstructured sections
- Semantic splitting: Use LLM to identify logical boundaries
- Hierarchical: Multiple granularities (file → section → function)

## Testing Scenarios
1. "Where is the main entry point of this program?"
2. "What functions modify the CUSTOMER-RECORD?"
3. "Explain what the CALCULATE-INTEREST paragraph does"
4. "Find all file I/O operations"
5. "What are the dependencies of MODULE-X?"
6. "Show me error handling patterns in this codebase"

## Performance Targets
- Query latency: <3 seconds end-to-end
- Retrieval precision: >70% relevant chunks in top-5
- Codebase coverage: 100% of files indexed
- Ingestion throughput: 10,000+ LOC in <5 minutes
- Answer accuracy: Correct file/line references

## Required Features

### Query Interface
- Natural language input for questions about the code
- Display retrieved code snippets with syntax highlighting
- Show file paths and line numbers for each result
- Confidence/relevance scores for retrieved chunks
- Generated explanation/answer from LLM
- Ability to drill down into full file context

### Code Understanding Features (implement at least 4)
- Code Explanation: Explain what a function/section does in plain English
- Dependency Mapping: Show what calls what, data flow between modules
- Pattern Detection: Find similar code patterns across the codebase
- Impact Analysis: What would be affected if this code changes?
- Documentation Gen: Generate documentation for undocumented code
- Translation Hints: Suggest modern language equivalents
- Bug Pattern Search: Find potential issues based on known patterns
- Business Logic Extract: Identify and explain business rules in code

## Vector Database Options
- Pinecone (managed cloud, production scale, free tier)
- Weaviate (self-host or cloud, hybrid search, GraphQL API)
- Qdrant (self-host or cloud, filtering, Rust-based)
- ChromaDB (embedded/self-host, prototyping, simple API)
- pgvector (PostgreSQL extension, familiar SQL)
- Milvus (self-host or Zilliz, large scale, GPU acceleration)

## Embedding Model Options
- OpenAI text-embedding-3-small (1536 dim, good balance)
- OpenAI text-embedding-3-large (3072 dim, higher quality)
- Voyage Code 2 (1536 dim, optimized for code)
- Cohere embed-english-v3 (1024 dim, good for English)
- sentence-transformers (local, varies, free)

## Recommended Tech Stack
- Vector Database: Pinecone, Weaviate, or Qdrant
- Embeddings: OpenAI text-embedding-3-small or Voyage Code 2
- LLM: GPT-4, Claude, or open source (Llama, Mistral)
- Framework: LangChain, LlamaIndex, or custom pipeline
- Backend: Node.js/Express, Python/FastAPI, or serverless
- Frontend: React, Next.js, or CLI interface
- Deployment: Vercel, Railway, or cloud provider

## RAG Framework Options
- LangChain: Flexible pipelines, many integrations, good docs
- LlamaIndex: Document-focused RAG, structured data
- Haystack: Production pipelines, evaluation tools
- Custom: Full control, learning exercise

## AI Cost Analysis (Required)
- Track dev/testing costs: embedding API, LLM API, vector DB hosting
- Production cost projections at 100 / 1K / 10K / 100K users

## RAG Architecture Documentation (Required)
- Vector DB Selection rationale
- Embedding Strategy
- Chunking Approach
- Retrieval Pipeline
- Failure Modes
- Performance Results

## Submission Requirements (Deadline: Sunday 10:59 PM CT)
- GitHub Repository (setup guide, architecture overview, deployed link)
- Demo Video (3-5 min)
- Pre-Search Document
- RAG Architecture Doc
- AI Cost Analysis
- Deployed Application
- Social Post (X or LinkedIn, tag @GauntletAI)
