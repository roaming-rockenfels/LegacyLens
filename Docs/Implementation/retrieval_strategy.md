# Retrieval Strategy

## The Core Problem

LAPACK routines have opaque names (DGESV, DGETRF, XERBLA) and the codebase has ~1,850 files across four precision variants of every algorithm. A naive semantic search against this codebase produces three failure modes:

1. **Vocabulary gap.** A query like "solve linear equations" has no lexical overlap with `SUBROUTINE DGESV`. Embedding models handle this partially via chunk enrichment (see chunking_strategy.md), but Fortran identifiers like IPIV and DGETRF remain opaque tokens that embeddings treat as noise.

2. **Precision-variant flooding.** SGESV, DGESV, CGESV, and ZGESV are the same algorithm with different numeric types. Without intervention, all four consume top-5 slots, leaving one slot for everything else.

3. **Driver burial.** User-facing routines like DGESV score lower than internal helpers like DGETRF2 on raw cosine similarity, because the helpers have longer source code with more token overlap.

Our retrieval pipeline addresses all three with a multi-tier architecture.

## Architecture Overview

```
User question
    │
    ▼
┌──────────────┐
│  embed_query  │  Voyage Code 3, input_type="query", 1024-dim
└──────┬───────┘
       │
       ▼
┌──────────────────┐     ┌──────────────────────┐
│  Entity Detection │────▶│  Entity-Filtered Tier │
│  (regex + rules)  │     │  (metadata queries)   │
└──────────────────┘     └──────────┬───────────┘
                                    │
       ┌────────────────────────────┤
       │                            │
       ▼                            ▼
┌─────────────┐          ┌──────────────────┐
│ Driver Boost │          │ Semantic Search   │
│ (Pass 1)     │          │ (Pass 2)          │
└──────┬──────┘          └────────┬─────────┘
       │                          │
       └────────┬─────────────────┘
                │
                ▼
        ┌───────────────┐
        │  Non-Entity    │
        │  Tier          │
        └───────┬───────┘
                │
    ┌───────────┴───────────┐
    │                       │
    ▼                       ▼
Entity Tier            Non-Entity Tier
  dedup                  dedup vs entity IDs
  sort by role           rerank (Voyage rerank-2)
  rerank                 ───────────────────────
  diversify              │
  ─────────              │
    │                    │
    ▼                    ▼
┌──────────────────────────────┐
│  Final Diversification       │
│  (cross-tier dedup, top_k)   │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Score Threshold Filter      │
│  (entity results bypass)     │
└──────────────────────────────┘
```

There are three entry paths through this pipeline, selected automatically based on query content:

| Path | When | Pinecone queries | Example |
|------|------|-----------------|---------|
| **Pin-unit** | `pin_unit` argument set | 2 (filtered + semantic) | `ll deps DGESV` |
| **Entity-aware** | Uppercase Fortran identifiers detected | 4-7 entity + 2 semantic | "routines that call XERBLA" |
| **Semantic-only** | No identifiers detected | 2 (driver + general) | "how does LAPACK solve linear equations?" |

## Step 1: Query Embedding

```python
embedding = embed_query(question)  # Voyage Code 3, 1024-dim
```

Every retrieval path starts by embedding the question with Voyage Code 3 using `input_type="query"`. This is the same model used at ingestion time with `input_type="document"`, and Voyage internally adjusts the embedding space for asymmetric query-document similarity.

**Why Voyage Code 3 instead of OpenAI or Cohere:** Voyage Code 3 is trained specifically on source code and technical documentation. In benchmarks it outperforms `text-embedding-3-large` on code search tasks. Since our documents are Fortran source + natural-language headers, a code-specialized model produces better alignment. The 1024-dim vectors balance quality against Pinecone storage cost (half the size of OpenAI's 3072-dim).

## Step 2: Entity Detection

```python
entities = _extract_fortran_entities(question)
# → {"parameters": ["IPIV"], "routines": ["DGETRF"]}
```

Before any vector search, we scan the query for Fortran identifiers using a regex (`r"\b([A-Z][A-Z0-9_]{1,30})\b"`). This detects tokens like DGETRF, IPIV, XERBLA that embedding models treat as meaningless strings but that have precise structural meaning in LAPACK.

### Why regex instead of NER or an LLM

An NER model or LLM call could classify entities more accurately, but:

1. **Latency.** An LLM call adds 500ms-2s. Regex runs in microseconds. Retrieval is on the critical path of every query.
2. **LAPACK's naming is mechanical.** All-caps tokens with S/D/C/Z prefixes are unambiguous. There's no context-dependence that requires ML — the rules are purely structural.
3. **No training data needed.** LAPACK's naming convention IS the classifier.

### Classification Rules

Tokens are classified in priority order:

| Rule | Classification | Examples |
|------|---------------|----------|
| In `_KNOWN_UTILITIES` set | Routine | XERBLA, ILAENV, LSAME |
| Starts with S/D/C/Z + alpha, ≥4 chars | Routine | DGETRF, SGESV, CLAHEF_RK |
| ≤8 chars, not matched above | Parameter | IPIV, LDA, NRHS, INFO |

**Known utilities exist because** LAPACK has ~7 routines that don't follow S/D/C/Z naming: XERBLA (error handler), ILAENV (environment query), LSAME (character comparison), etc. Without this list, XERBLA would be classified as a parameter (it starts with X, not S/D/C/Z) and the query "routines that call XERBLA" would issue `filter={"parameters": "XERBLA"}` instead of `filter={"calls": "XERBLA"}` — returning nothing useful.

**English stopwords** (THE, AND, FOR, WHAT, etc.) and domain acronyms (LAPACK, BLAS, SVD, LU, QR) are excluded before classification. Without this, "WHAT DOES DGETRF DO" would classify WHAT and DOES as parameters.

**Why the ≤8 char cutoff for parameters:** LAPACK parameter names are short mnemonics (IPIV=integer pivot, LDA=leading dimension of A, NRHS=number of right-hand sides). Routine names are always ≥4 characters with a precision prefix. The 8-char threshold captures all standard parameter names without overlap.

## Step 3: Entity-Filtered Tier

When entities are detected, we issue targeted Pinecone queries using metadata filters. This bypasses the vocabulary gap entirely — we're not asking "what embeddings are similar to IPIV?" but rather "which vectors have IPIV in their `parameters` metadata field?"

### Parameter Entity Queries

For each detected parameter (e.g., IPIV):

```python
# Query 1: D-prefix canonical results
query_vectors(embedding, top_k=5, filter={"parameters": "IPIV", "precision": "double"})

# Query 2: All-precision coverage
query_vectors(embedding, top_k=5, filter={"parameters": "IPIV"})
```

### Routine Entity Queries

For each detected routine (e.g., XERBLA):

```python
# Query 1: The routine itself
query_vectors(embedding, top_k=3, filter={"unit_name": "XERBLA"})

# Query 2: D-prefix callers
query_vectors(embedding, top_k=5, filter={"calls": "XERBLA", "precision": "double"})

# Query 3: All-precision callers
query_vectors(embedding, top_k=5, filter={"calls": "XERBLA"})
```

### Why Dual Precision Queries

Even with metadata filters, Pinecone still ranks results by cosine similarity to the query embedding. This creates an embedding bias: within the set of routines that have IPIV as a parameter, Pinecone returns whichever ones' enriched content is most similar to the query text. This bias is unpredictable — SVD routines might score higher than LU routines for "What functions modify IPIV?" because their enriched content happens to overlap more with the query embedding.

The dual-query strategy mitigates this:

1. The `precision: "double"` query guarantees D-prefix (canonical) results in the candidate pool, regardless of their embedding score.
2. The all-precision query provides broader coverage, catching routines where only C/Z variants exist (e.g., Hermitian matrix routines like ZLAHEF_ROOK, which have no D-prefix equivalent because Hermitian matrices only exist for complex types).

**Why not just query with `precision: "double"` alone:** Some routines only exist in complex variants (C/Z). Hermitian factorizations (xLAHEF, xHETRF) have no real-valued equivalent. A double-only filter would miss them entirely.

**Why not use a neutral/uniform embedding vector:** We tried replacing the query embedding with a uniform vector `[1.0] * 1024` for entity queries, to eliminate embedding bias entirely. This backfired — the uniform vector creates its own bias (cosine similarity degenerates to a function of vector magnitude), producing near-zero scores and arbitrary orderings. The query embedding, while imperfect, at least biases toward topically relevant routines.

### Entity Pipeline Processing

Entity results go through a four-step pipeline:

```
Raw entity results (may have duplicates from multiple queries)
    │
    ▼
1. Deduplicate (by vector ID)
    │
    ▼
2. Sort by routine role (driver > computational > auxiliary > blas)
    │
    ▼
3. Rerank (Voyage rerank-2, top_k × 2 candidates)
    │
    ▼
4. Diversify (collapse precision variants, D-prefix preference, top_k)
```

**The order of steps 3 and 4 is critical.** Reranking must happen before diversification. If diversification runs first, the reranker can undo D-prefix preference by promoting a C/Z variant above its D equivalent based on purpose-text similarity. With rerank-then-diversify, the reranker orders by relevance and diversification makes the final call on which precision variant survives — a decision the reranker has no domain knowledge to make.

## Step 4: Two-Pass Semantic Tier

Regardless of whether entities were detected, we also run two semantic Pinecone queries:

```python
# Pass 1: Driver-filtered
driver_results = query_vectors(embedding, top_k=5, filter={"routine_role": "driver"})

# Pass 2: Unfiltered (over-fetch 3× for diversification headroom)
general_results = query_vectors(embedding, top_k=15)
```

### Why Two Passes Instead of One

LAPACK has ~262 driver routines (user-facing entry points like DGESV, DSYEV, DGEEV) and ~1,389 computational routines (internal workhorses like DGETRF, DGETRS). Computational routines have longer, more detailed source code that scores higher on cosine similarity. A single unfiltered query for "solve linear equations" returns DGETRF, DGETRS, DLASWP (internals) before DGESV (the actual user-facing solver).

The driver-filtered pass guarantees that user-facing routines appear in the candidate pool. These are merged with general results, deduplicated, and processed through the non-entity pipeline.

**Why not boost drivers with a score multiplier instead:** Score manipulation is fragile — the right multiplier depends on query type, and a fixed boost either over-promotes irrelevant drivers or under-promotes relevant ones. A filtered query is deterministic: it always returns the top-k drivers by similarity, and the reranker decides their final position.

### Non-Entity Pipeline Processing

Non-entity results are deduplicated against entity IDs (so the same routine doesn't appear in both tiers), then processed:

```
Driver results + General results (deduped against entity IDs)
    │
    ▼
1. Rerank (Voyage rerank-2, remaining_slots × 3 candidates)
    │
    ▼
(fed into final diversification — see Step 5)
```

The non-entity tier intentionally does NOT pre-diversify. If it did, the final cross-tier diversification would have too few spare candidates to backfill slots lost to cross-tier duplicate collapsing. By keeping the non-entity pool un-diversified, the final pass has a deep bench.

## Step 5: Cross-Tier Merge and Final Diversification

```python
combined = _diversify_results(diversified_entity + reranked_non_entity, top_k)
```

Entity results occupy the first tier (they are structurally matched and take priority). Non-entity results fill remaining slots. A final diversification pass runs across both tiers to catch precision duplicates that span tiers — for example, DGETRF from the entity tier and SGETRF from the semantic tier share the base name GETRF, so SGETRF is collapsed.

### How Diversification Works

```python
for each result in order:
    base = strip S/D/C/Z prefix from unit_name  (DGETRF → GETRF)
    if base not seen before:
        keep this result
    elif this result is D-prefix and current keeper is not:
        replace keeper with this result (same position)
    else:
        skip (duplicate precision variant)
```

The D-prefix replacement rule means that even if a Z-prefix variant appears first (ranked higher by the reranker), a D-prefix variant appearing later will replace it — as long as they share the same base name. This only fails when no D-prefix variant exists at all (e.g., Hermitian routines ZLAHEF_ROOK, which have no DLAHEF_ROOK).

**Why prefer D-prefix specifically:** Double-precision (D) is the canonical form in the LAPACK Users' Guide, academic papers, and tutorials. When someone writes about "DGESV", they mean the algorithm — the S/C/Z variants are understood as type substitutions. Showing ZGESV instead of DGESV would be technically correct but confusing to anyone who learned LAPACK from documentation.

## Step 6: Reranking

Both tiers use Voyage's `rerank-2` cross-encoder model to reorder candidates by relevance.

```python
def _rerank_results(question, results, top_k):
    documents = []
    for r in results:
        meta = r["metadata"]
        doc = f"{meta['unit_type']} {meta['unit_name']}: {meta['purpose']}"
        doc += f" Parameters: {', '.join(meta['parameters'][:10])}"
        doc += f" Calls: {', '.join(meta['calls'][:10])}"
        documents.append(doc)
    reranked = client.rerank(question, documents, model="rerank-2", top_k=top_k)
    return [results[rr.index] for rr in reranked.results]
```

### Why Rerank Instead of Trusting Cosine Similarity

Cosine similarity is a bi-encoder score — query and document are embedded independently, and similarity is a single dot product. This misses fine-grained relevance signals:

- "How does LAPACK compute eigenvalues?" scores DGESDD (SVD) almost as high as DSYEV (eigenvalues) because SVD and eigenvalue decomposition share mathematical vocabulary.
- "routines that call XERBLA" scores SVD drivers highest because they have the longest enriched content with the most lexical overlap.

The reranker is a cross-encoder — it processes the query and document together, attending to both simultaneously. It can distinguish "eigenvalues" from "singular values" and recognize that DGESDD's purpose ("singular value decomposition") doesn't match an eigenvalue query, even when cosine similarity says they're close.

### What Gets Reranked

We construct lightweight document strings from metadata rather than passing full Fortran source. The reranker sees:

```
SUBROUTINE DGESV: DGESV computes the solution to a real system of linear
equations A * X = B. Parameters: N, NRHS, A, LDA, IPIV, B, LDB, INFO
Calls: DGETRF, DGETRS, XERBLA
```

This keeps reranker input small (fast inference) while preserving the semantic signals that matter: routine type, name, purpose description, parameter list, and call graph.

### Graceful Degradation

If the reranker API fails (rate limit, network error, missing credentials), the pipeline falls back to the pre-rerank order truncated to `top_k`. This is strictly worse but ensures the system never hard-fails on a reranker outage.

## Step 7: Score Threshold

```python
_MIN_SCORE_THRESHOLD = 0.45

return [
    r for r in combined
    if r.get("_entity_match") or r.get("score", 0) >= _MIN_SCORE_THRESHOLD
]
```

A minimum cosine similarity threshold filters out low-confidence results. This prevents the system from returning 5 results when only 3 are genuinely relevant — padding with noise degrades trust.

**Entity-matched results bypass the threshold** because their relevance is established structurally (they matched a metadata filter for a detected identifier), not by embedding similarity. An entity result with a cosine score of 0.3 is still structurally correct — it genuinely contains the IPIV parameter or calls XERBLA.

**Why 0.45 specifically:** Empirically tuned. Below 0.45, results are typically topically adjacent but not actually relevant (e.g., a matrix scaling routine appearing for an eigenvalue query). Above 0.5 would be too aggressive, filtering out legitimate but lower-scoring computational routines.

## Pin-Unit Path

The `pin_unit` path is a separate, simpler retrieval mode used by the `deps` and `document` CLI commands:

```python
if pin_unit:
    pinned = query_vectors(embedding, top_k=3, filter={"unit_name": pin_unit.upper()})
    semantic = query_vectors(embedding, top_k=top_k * 3)
    # Merge: pinned first, then semantic (deduped)
    return merged
```

This bypasses entity detection, reranking, diversification, and score thresholds entirely. The use case is different: the user has already specified which routine they want (e.g., `ll deps DGESV`), so we guarantee it appears first and fill remaining slots with semantically related context.

**Why top_k=3 for the pinned query:** A routine split across multiple chunks (the ~3% of oversized files) produces 2-3 vectors with the same `unit_name`. Fetching 3 ensures we get all chunks.

## What We Deliberately Don't Do

### No query expansion or HyDE

Hypothetical Document Embeddings (HyDE) generates a synthetic document from the query, then embeds that instead. For LAPACK, this would require the LLM to know Fortran naming conventions to generate useful synthetic documents. It adds 1-2s latency per query for uncertain benefit, and our chunk enrichment already bridges the vocabulary gap at ingestion time rather than at query time.

### No recursive retrieval or graph traversal

We could follow the `calls` metadata to recursively retrieve callers-of-callers or build full dependency trees at query time. We don't because: (1) the `deps` command with `pin_unit` already provides single-hop context, (2) multi-hop traversal would require N additional Pinecone queries per hop, and (3) the LLM generation layer can synthesize relationships from the flat result set.

### No query-time LLM classification

An LLM could classify the query intent (e.g., "this is asking about dependencies" vs "this is asking about algorithm behavior") and select different retrieval strategies. We use regex-based entity detection instead because it's free, deterministic, and sufficient — LAPACK query intents are well-captured by whether or not the query contains identifiers.

### No caching layer

We don't cache embeddings or Pinecone results. The Voyage API is fast (~100ms per embed), Pinecone queries are fast (~50ms each), and LAPACK queries are low-volume. A cache would add complexity for negligible latency savings.

### No fusion ranking (RRF)

Reciprocal Rank Fusion is a common technique for merging results from multiple retrieval passes by combining rank positions. We use tiered merge instead (entity tier fills first, non-entity fills remaining) because our tiers have different trust levels — entity results are structurally correct while semantic results are probabilistic. RRF would mix them by rank, potentially demoting a structurally perfect entity match below a high-scoring but semantically ambiguous result.

## Pinecone Query Budget

The number of Pinecone queries varies by path:

| Path | Queries | Breakdown |
|------|---------|-----------|
| Pin-unit | 2 | 1 filtered + 1 semantic |
| Semantic-only (no entities) | 2 | 1 driver-filtered + 1 unfiltered |
| 1 parameter detected | 4 | 2 entity (D-prefix + all) + 2 semantic |
| 1 routine detected | 5 | 3 entity (self + D-callers + all-callers) + 2 semantic |
| 1 param + 1 routine | 7 | 5 entity + 2 semantic |

Plus 1-2 Voyage API calls (1 embed + 0-2 rerank depending on tier sizes). Total wall-clock time for a typical entity-aware query: ~400-600ms.

## Call Sites

| Call site | Arguments | Path taken |
|-----------|-----------|------------|
| `ll query <question>` | `retrieve(question, top_k=5)` | Entity-aware or semantic-only |
| `ll explain <question>` | `retrieve(question, top_k=5)` | Entity-aware or semantic-only |
| `ll logic <question>` | `retrieve(question, top_k=5)` | Entity-aware or semantic-only |
| `ll deps <unit>` | `retrieve(question, top_k=10, pin_unit=name)` | Pin-unit |
| `ll document <unit>` | `retrieve(question, top_k=5, pin_unit=name)` | Pin-unit |
| `POST /query` | `retrieve(req.question, top_k=req.top_k)` | Entity-aware or semantic-only |
| `GET /search` | `retrieve(question, top_k=top_k)` | Entity-aware or semantic-only |

## Key Files

| File | Role |
|------|------|
| `src/legacylens/rag/retrieve.py` | Full retrieval pipeline: entity detection, tiered merge, diversification, reranking, threshold |
| `src/legacylens/rag/embeddings.py` | Voyage Code 3 embedding + reranking client |
| `src/legacylens/rag/storage.py` | Pinecone index management, `query_vectors()`, `upsert_vectors()` |
| `tests/test_retrieve.py` | 29 tests covering entity detection, diversification, reranking, tiered merge, threshold |
