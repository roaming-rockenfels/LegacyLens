# AI Development Log & Cost Analysis

This document serves as a source of truth for documenting the AI development process for **LegacyLens**, covering the tools and techniques used, and providing a critical analysis of both development and projected production costs.

**Reporting period:** 2026-03-02 (Monday) – 2026-03-04 (Wednesday)

---

## AI Development Log

This section provides a detailed log of the AI development workflow, code analysis, and key learnings for the LegacyLens RAG pipeline (LAPACK Fortran codebase queryable via natural language).

### Tools & Workflow

| AI Tool Used       | Integration / Workflow                                                                                                                                                   |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Claude (Cursor)    | Primary implementation agent for features, refactors, and documentation. Used for RAG pipeline, FastAPI/CLI, chunking, retrieval, and frontend (streaming chat, demo UI). |
| Claude Code (ccusage) | CLI-based AI coding assistant (Claude Opus 4.6). Used for multi-file refactors, streaming chat implementation, UX iteration, test-driven development, and documentation. |
| Cursor             | IDE workflow for code search, test-driven iteration, and natural-language debugging. Models routed through Cursor as configured in the IDE.                              |
| OpenRouter         | API gateway for the **LegacyLens application** LLM path only. Routes to Claude Haiku 4.5 (default) or other models via `LEGACYLENS_MODEL`. Used for single-shot `/query` and multi-turn `/chat` answer generation. |
| Voyage AI          | Embeddings (Voyage Code 3) and reranking (rerank-2). Not an LLM; usage is for ingestion and retrieval only.                                                              |
| Pinecone           | Vector database; free tier for this project. No variable API cost.                                                                                                       |

**Note:** LegacyLens does not use OpenAI API directly. All production LLM traffic goes through OpenRouter.

---

### MCP Usage

| MCP Used                | Functionality Enabled                                                                                                                                 |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| Sequential Thinking MCP  | Structured multi-step reasoning for architecture planning, retrieval strategy, and design choices.    |

No other MCPs (Prisma, Context7, Memory) were used for LegacyLens.

---

### Effective Prompts

Analyzed from 48 user prompts across 9 Claude Code conversation sessions (extracted from `~/.claude/projects/` JSONL transcripts).

**Top 3 prompt categories by frequency:**

| Rank | Category | Count | Share | Example |
| ---: | -------- | ----: | ----: | ------- |
| 1 | Update / modify existing behavior | 20 | 41% | "Update the scroll logic so that it still does not auto-scroll with the streaming response, but if the user scrolls down and catches up to the stream, it should then auto-scroll with it." |
| 2 | Debug / fix issues | 6 | 12% | "Diagnose the cause of the following chat response: … ERROR: Method Not Allowed" |
| 3 | Investigative questions | 5 | 10% | "Does the check mark on vector search render when actual chunks are returned from the db?" |

Other categories: Plan / architecture decisions (4), Dev server / environment management (4), Add new features / content (3), Push / commit / open PR (2).

**Key insight:** The dominant development pattern was **iterative refinement** — specific, targeted modification instructions rather than broad feature requests. This reflects a workflow where AI handles the implementation while the developer drives UX decisions through rapid feedback loops.

**Most effective prompt patterns:**
1. "Update X so that Y" — specific behavior change with clear acceptance criteria.
2. "Make a plan to [improve/upgrade] Z. I want to be a part of the discovery and decision making process." — collaborative architecture with human-in-the-loop.
3. "Diagnose the cause of [error]" — pasting exact error output for targeted debugging.
4. "Are the API features for this project still in parity with the CLI?" — parity/audit checks across system boundaries.

Date captured: **2026-03-05**

---

### Code Analysis

Based on git history for the LegacyLens repository (37 total commits):

| Code Origin                          | Commits | Lines Inserted | Lines Deleted | Percentage (by commits) |
| ------------------------------------ | ------: | -------------: | ------------: | ----------------------: |
| AI-assisted (`Co-Authored-By` present) | 29      | +9,118         | -406          | 78%                     |
| Hand-written (merges, solo fixes)     | 8       | +4,751         | -251          | 22%                     |
| **Total**                            | **37**  | **+13,869**    | **-657**      | **100%**                |

---

### Strengths & Limitations

| Category                        | Description                                                                                                                                 |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| Strengths (Where AI excelled)   | Fast implementation of RAG pipeline (chunking, embeddings, Pinecone, retrieval with entity/driver tiers), FastAPI + Typer CLI, streaming chat, and demo UI. Clear separation of concerns (source_reader, generate, retrieve). |
| Limitations (Where AI struggled) | Retrieval precision and diversification required several iterations (precision-variant flooding, driver burial). Deployment and "two apps" on Fly.io needed manual debugging. |

**Test coverage:** Pytest suite for chunkers, retrieval, API, session, generate, and source_reader. See `tests/`.

**Retrieval / latency (from `Docs/Submission/rag_architecture.md`):**

| Metric                 | Value        | Notes                                      |
| ---------------------- | ------------ | ------------------------------------------ |
| Query latency (semantic) | ~400–600 ms  | 1 embed + 2 Pinecone + rerank              |
| Query latency (entity-aware) | ~600–900 ms | Extra Pinecone queries for entity tier     |
| Answer generation      | ~2–4 s       | Claude Haiku 4.5 via OpenRouter            |
| Cost per query (est.)  | ~$0.004      | Embed + Rerank + LLM (see RAG architecture doc) |

---

### Key Learnings

1. Enriching chunks with natural-language headers (purpose, file, calls) bridges the vocabulary gap between user questions and Fortran identifiers (e.g. DGESV, DGETRF).
2. Diversification by base routine name (S/D/C/Z variants) is necessary to avoid precision-variant flooding in top-k results.
3. Cost/usage telemetry should be tracked from day one (OpenRouter activity, Cursor/ccusage) to avoid retroactive reconstruction.

---

## AI Cost Analysis

Understanding the costs associated with using AI models is crucial for planning production deployments. This section tracks development costs and projects production expenditure for **LegacyLens**.

**Reporting period:** 2026-03-02 – 2026-03-04

### Data Sources

| Source        | How obtained | Time window |
| ------------- | ------------ | ----------- |
| Cursor usage  | CSV export from Cursor Settings → Usage/Billing | Mar 3 – Mar 4 (12 calls) |
| ccusage       | `ccusage session` CLI — filtered to LegacyLens project sessions | Project lifetime (session-level, not date-filterable) |
| OpenRouter    | `GET /api/v1/activity` with management API key, filtered to Mar 2 – Mar 4 | Mar 2 – Mar 4 |
| Voyage AI     | CSV export from Voyage dashboard | Mar 2 – Mar 3 |
| Pinecone      | Free tier — no variable cost | N/A |

---

### Development & Testing Costs (LegacyLens)

| Cost Metric                     | Actual Spend (USD) | Notes |
| ------------------------------- | -----------------: | ----- |
| Cursor (CSV) API costs          | $1.69              | 12 calls, 3.26M tokens total. Model: auto. |
| ccusage (Claude Code) API costs | $26.19             | LegacyLens session only (claude-opus-4-6). 39.7M tokens. |
| OpenRouter (LegacyLens app)     | $4.82              | 848 requests across 3 models (Mar 2–4). 100% attributable to LegacyLens. |
| Voyage AI (embeddings + rerank) | $1.15              | Batch embedding (Voyage Code 3): $1.15. File sync: <$0.01. |
| Pinecone                        | $0.00              | Free tier. |
| **Total variable LLM/API costs** | **$33.85**        | |

---

## Detailed Usage Breakdowns by Model

### Estimation Methodology (Reference)

- **Actual cost definitions:**
  - **Cursor (CSV):** Sum recorded per-event `Cost` values for the project period.
  - **ccusage:** Session-level costs from `ccusage session` CLI, filtered to the LegacyLens project directory.
  - **OpenRouter:** `usage` field (cost in USD) from `GET /api/v1/activity` with management key.
  - **Voyage:** CSV export from Voyage dashboard.
- **Formula (token-based estimate):**
  `Estimated Cost = (Input Tokens / 1,000,000 × Input $/1M) + (Output Tokens / 1,000,000 × Output $/1M)`
  For LegacyLens, the main variable cost is OpenRouter (Claude Haiku 4.5 or whatever `LEGACYLENS_MODEL` is). Per-query estimate from `Docs/Submission/rag_architecture.md`: ~$0.004 (embed + rerank + LLM).

---

### OpenRouter (LegacyLens application only)

Data retrieved via management API key: `GET https://openrouter.ai/api/v1/activity`

**Summary by model (Mar 2 – Mar 4):**

| Model                      | Requests | Prompt Tokens | Completion Tokens | Cost (USD) |
| -------------------------- | -------: | ------------: | ----------------: | ---------: |
| anthropic/claude-3-haiku   | 492      | 2,694,732     | 59,323            | $0.75      |
| anthropic/claude-haiku-4.5 | 91       | 581,786       | 82,714            | $1.00      |
| anthropic/claude-sonnet-4  | 265      | 845,313       | 36,082            | $3.08      |
| **Total**                  | **848**  | **4,121,831** | **178,119**       | **$4.82**  |

*Claude Sonnet 4 costs are from testing/evaluation only (comparing model quality via `LEGACYLENS_MODEL=anthropic/claude-sonnet-4`). It is not used in the default production configuration.*

**Daily breakdown:**

| Date       | Requests | Cost (USD) |
| ---------- | -------: | ---------: |
| 2026-03-02 | 726      | $3.47      |
| 2026-03-03 | 70       | $0.57      |
| 2026-03-04 | 52       | $0.78      |
| **Total**  | **848**  | **$4.82**  |

**Notes:** Claude 3 Haiku was used for batch ingestion (chunk enrichment). Claude Haiku 4.5 is the default runtime model for `/query` and `/chat`. Claude Sonnet 4 was used for testing/evaluation of higher-quality responses via `LEGACYLENS_MODEL` — not part of the default production path.

---

### Cursor (CSV)

Exported from Cursor Settings → Usage/Billing. All calls used model `auto` (Cursor-routed), Kind: `Included`.

| Date (UTC)  | Input Tokens | Cache Read | Output Tokens | Total Tokens | Cost (USD) |
| ----------- | -----------: | ---------: | ------------: | -----------: | ---------: |
| 2026-03-03  | 109,256      | 361,472    | 8,800         | 479,528      | $0.27      |
| 2026-03-04  | 230,272      | 912,576    | 9,737         | 1,152,585    | $0.58      |
| 2026-03-05* | 358,267      | 1,260,512  | 12,463        | 1,631,242    | $0.84      |
| **Total**   | **697,795**  | **2,534,560** | **31,000** | **3,263,355** | **$1.69** |

*\*Mar 5 UTC = Mar 4 local time (PST/PDT).*

---

### ccusage (Claude Code)

Data from `ccusage session` CLI, filtered to the LegacyLens project directory (`-Users-rockenfels-Documents-Gauntlet-3-LegacyLens`).

| Model            | Input Tokens | Output Tokens | Cache Write | Cache Read  | Cost (USD) |
| ---------------- | -----------: | ------------: | ----------: | ----------: | ---------: |
| claude-opus-4-6  | 3,126        | 69,781        | 800,113     | 38,856,343  | $26.19     |
| **Total**        | **3,126**    | **69,781**    | **800,113** | **38,856,343** | **$26.19** |

**Note:** ccusage reports session-level aggregates and cannot be filtered by date range. The $26.19 represents the total Claude Code cost for all LegacyLens development sessions. The vast majority of cost comes from cache read tokens (prompt context caching).

---

### Voyage AI

Exported from Voyage dashboard (Mar 2 – Mar 3):

| Date       | Event Type | Model          | Cost (USD) |
| ---------- | ---------- | -------------- | ---------: |
| 2026-03-02 | sync       | voyage-code-2  | $0.00      |
| 2026-03-03 | files      | —              | <$0.01     |
| 2026-03-03 | batch      | voyage-code-3  | $1.15      |
| **Total**  |            |                | **$1.15**  |

---

### Total Project Costs (Development Period)

| Source                  | Cost (USD) |
| ----------------------- | ---------: |
| Cursor (CSV)            | $1.69      |
| ccusage (Claude Code)   | $26.19     |
| OpenRouter (LegacyLens) | $4.82     |
| Voyage AI               | $1.15      |
| Pinecone                | $0.00      |
| **Variable subtotal**   | **$33.85** |
| Fixed subscriptions     | N/A (Claude Code Max subscription, Cursor Pro — not prorated to this project) |
| **Grand total**         | **$33.85** |

---

## Production Cost Projections

LegacyLens is a **single-query RAG pipeline**: one user question → one query embed + Pinecone retrieval + (optionally) one LLM call. There are no multi-tool agent loops; chat is multi-turn but each turn is still one retrieval + one LLM call.

### Observed / Designed Behavior

| Metric                     | Value        | Source |
| -------------------------- | ------------ | ------ |
| LLM calls per user query   | 1            | Single-shot `/query` or one turn of `/chat`. |
| Embed + rerank per query   | 1 each       | One query embedding, one rerank pass (Voyage). |
| Cost per query (est.)      | ~$0.004      | Embed (~$0.0001) + Rerank (~$0.002) + LLM (~$0.002). From `Docs/Submission/rag_architecture.md`. |
| Default model              | Claude Haiku 4.5 | `LEGACYLENS_MODEL` (or default in code). |

### Cost Projection Assumptions

| Metric                       | Value    | Basis |
| ---------------------------- | -------- | ----- |
| Queries per user per month   | 20       | 1–2 sessions per week, ~5–10 queries per session. |
| Cost per user query          | $0.004   | From RAG architecture doc (embed + rerank + LLM). |
| **Cost per user per month**  | **$0.08** | 20 × $0.004. |

### Estimated Monthly Production Costs by Scale

| User scale   | Queries/month | Estimated monthly cost |
| ------------ | ------------: | ----------------------: |
| 100 users    | 2,000         | $8 / month             |
| 1,000 users  | 20,000        | $80 / month            |
| 10,000 users | 200,000       | $800 / month           |
| 100,000 users | 2,000,000    | $8,000 / month         |

**Sensitivity:** If users double query volume (40 queries/user/month), double the costs. If you switch to a cheaper/faster model (e.g. GPT-4o-mini via OpenRouter), reduce the per-query LLM share; if you switch to Sonnet, increase it (see `Docs/Presearch/Q2_llm_answer_generation.md` for per-model cost per query).

---

Document prepared for: **LegacyLens**
Last updated: **2026-03-05**
File: **AI Development Log & Cost Analysis**
