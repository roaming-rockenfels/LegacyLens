# Technical Research: Embedding Model

## Decision
Which embedding model to use for vectorizing legacy Fortran code chunks in a RAG pipeline with Pinecone.

## Constraints
- Budget-conscious student project (~1 week sprint)
- Target: LAPACK Fortran codebase (10K+ LOC, 50+ files)
- Vector DB: Pinecone (managed, free tier)
- Retrieval precision target: >70% relevant chunks in top-5
- No embedding model is explicitly trained on Fortran

## Candidates

### Voyage Code 2
- **Maturity:** Production-ready, purpose-built for code retrieval
- **Maintenance:** Active development by Voyage AI
- **Performance:** 10-20% higher than general-purpose models on code search benchmarks (CosQA, CodeSearchNet)
- **DX:** Clean Python SDK (`voyageai`), straightforward API
- **Community:** Growing, code-focused niche
- **License:** Commercial API
- **Dimensions:** 1536
- **Max input tokens:** 16,000 (generous for long subroutines)
- **Cost (50K tokens):** ~$0.006
- **Pros:** Code-optimized retrieval, 16K context handles long functions, same dimensionality as OpenAI small
- **Cons:** Smaller batch size (128 vs 2048), less ecosystem documentation than OpenAI, Fortran not explicitly in training data

### OpenAI text-embedding-3-small
- **Maturity:** Stable, widely deployed
- **Maintenance:** Actively maintained by OpenAI
- **Performance:** MTEB retrieval ~62-64% (general tasks), lower on code-specific
- **DX:** Excellent SDK (`openai`), extensive documentation
- **Community:** Largest ecosystem
- **License:** Commercial API
- **Dimensions:** 1536
- **Max input tokens:** 8,191
- **Cost (50K tokens):** ~$0.001
- **Pros:** Cheapest API option, battle-tested, largest batch size (2048), most documentation
- **Cons:** Not code-optimized, 8K token limit may truncate longer subroutines, 10-15% lower quality on code retrieval

### OpenAI text-embedding-3-large
- **Maturity:** Stable
- **Performance:** MTEB retrieval ~64-66%
- **Dimensions:** 3072 (reducible via MRL)
- **Max input tokens:** 8,191
- **Cost (50K tokens):** ~$0.007
- **Pros:** Marginally better than small, supports dimension reduction
- **Cons:** Same code quality gap as small, 2x storage, marginal gain doesn't justify complexity

### Cohere embed-english-v3
- **Maturity:** Stable, strong on English text
- **Performance:** MTEB retrieval ~66-68% (highest for general English)
- **Dimensions:** 1024
- **Max input tokens:** 512 (DEALBREAKER)
- **Cost (50K tokens):** ~$0.005
- **Pros:** Best English text retrieval, `input_type` asymmetric search feature
- **Cons:** 512-token input limit truncates most code chunks, not code-optimized

### Local sentence-transformers (CodeBERT)
- **Maturity:** Research model, stable
- **Performance:** MTEB ~53% (CodeBERT), no Fortran training
- **Dimensions:** 768
- **Max input tokens:** 512
- **Cost:** $0.00 (free)
- **Pros:** Zero API cost, no network dependency
- **Cons:** 512-token limit, no Fortran support, larger Docker image on Fly.io, 10-20% lower quality than Voyage

## Pro-Con Matrix
| Dimension    | Voyage Code 2 | OAI small | OAI large | Cohere v3 | CodeBERT |
|-------------|:---:|:---:|:---:|:---:|:---:|
| Code quality | High | Medium | Medium+ | Medium | Medium |
| Cost         | $0.006 | $0.001 | $0.007 | $0.005 | $0.00 |
| Token limit  | 16K | 8K | 8K | 512 | 512 |
| Integration  | Good | Excellent | Excellent | Good | Good |
| Dimensions   | 1536 | 1536 | 3072 | 1024 | 768 |

## Recommendation
**Voyage Code 2** — purpose-built for code retrieval with a 10-20% quality advantage on code search benchmarks. The 16K token context window avoids truncation on long LAPACK subroutines. Cost is negligible (~$0.006 total). Fallback: OpenAI text-embedding-3-small if Voyage has availability issues.

Trade-off: Slightly less ecosystem documentation than OpenAI, smaller batch sizes (128 vs 2048). Neither matters at this project's scale.

**Mitigation for Fortran gap:** Prepend natural-language summary lines to chunks (e.g., "Fortran subroutine DGEMM: Double-precision General Matrix Multiply") to leverage LAPACK's excellent mathematical comments.

## Decision Confirmation
- [x] User confirmed on 2026-03-02
