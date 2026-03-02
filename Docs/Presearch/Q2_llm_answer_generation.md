# Technical Research: LLM for Answer Generation

## Decision
Which LLM to use for generating answers from retrieved Fortran code chunks in the RAG pipeline.

## Constraints
- Budget-conscious student project (~1 week sprint)
- Features requiring code reasoning: Explanation, Dependency Mapping, Doc Gen, Business Logic Extraction
- Backend: Python/FastAPI on Fly.io
- Streaming support needed for CLI responses
- Query latency target: <3 seconds end-to-end

## Candidates

### Claude Haiku 4.5
- **Maturity:** Production-ready, latest Anthropic small model
- **Maintenance:** Actively maintained by Anthropic
- **Performance:** Very good code comprehension, strong Fortran understanding
- **DX:** Excellent Python SDK (`anthropic`), async support, typed responses
- **Community:** Growing, well-documented
- **License:** Commercial API
- **Input cost:** ~$0.80 / 1M tokens
- **Output cost:** ~$4.00 / 1M tokens
- **Context window:** 200K tokens (largest of all candidates)
- **Latency:** First token ~300-500ms, full response ~1-2s
- **Cost per query:** ~$0.0044 (3K input + 500 output tokens)
- **Pros:** Best quality/cost ratio, 200K context window, fast latency, native streaming, already in Anthropic ecosystem
- **Cons:** Slightly less deep reasoning than Sonnet/GPT-4o on complex logic

### Claude Sonnet 4
- **Input cost:** ~$3.00 / 1M tokens
- **Output cost:** ~$15.00 / 1M tokens
- **Context window:** 200K tokens
- **Latency:** First token ~500-1000ms, full response ~2-4s
- **Cost per query:** ~$0.0165
- **Pros:** Excellent code comprehension (best in class), same 200K context, same SDK
- **Cons:** 3.75x more expensive than Haiku, slower

### GPT-4o
- **Input cost:** ~$2.50 / 1M tokens
- **Output cost:** ~$10.00 / 1M tokens
- **Context window:** 128K tokens
- **Latency:** First token ~500-800ms, full response ~2-3s
- **Cost per query:** ~$0.0125
- **Pros:** Strong code understanding, mature SDK, extensive ecosystem
- **Cons:** Smaller context window (128K vs 200K), more expensive than Haiku, occasionally hallucinates Fortran details

### GPT-4o-mini
- **Input cost:** ~$0.15 / 1M tokens
- **Output cost:** ~$0.60 / 1M tokens
- **Context window:** 128K tokens
- **Latency:** First token ~200-400ms, full response ~1-1.5s
- **Cost per query:** ~$0.00075
- **Pros:** Cheapest option by far, fastest latency
- **Cons:** Noticeably shallower explanations, struggles with dependency mapping and business logic extraction on dense code

### Llama 3.1 70B (via Together.ai/Groq)
- **Input cost:** ~$0.50 / 1M tokens
- **Output cost:** ~$0.50 / 1M tokens
- **Context window:** 128K tokens
- **Latency:** Varies by provider, ~200-500ms first token (Groq)
- **Cost per query:** ~$0.0018
- **Pros:** Cheap, fast (on Groq), OpenAI-compatible API
- **Cons:** Significant quality gap on Fortran comprehension, adds vendor dependency, provider reliability varies, training data skews to modern languages

## Pro-Con Matrix
| Dimension         | Haiku 4.5 | Sonnet 4 | GPT-4o | GPT-4o-mini | Llama 3.1 70B |
|-------------------|:---------:|:--------:|:------:|:-----------:|:-------------:|
| Fortran quality   | Very good | Excellent | Excellent | Moderate | Moderate |
| Cost/query        | $0.0044 | $0.0165 | $0.0125 | $0.00075 | $0.0018 |
| Context window    | 200K | 200K | 128K | 128K | 128K |
| Latency           | Fast | Medium | Medium | Fastest | Fast (Groq) |
| SDK quality       | Excellent | Excellent | Excellent | Excellent | Good |
| Streaming         | Yes | Yes | Yes | Yes | Yes |
| Explanation depth | Good | Excellent | Excellent | Shallow | Moderate |

## Cost Projection
| Scale | Haiku 4.5 | GPT-4o-mini | Sonnet 4 |
|-------|-----------|-------------|----------|
| Dev/testing (~200 queries) | $0.88 | $0.15 | $3.30 |
| 1,000 queries | $4.40 | $0.75 | $16.50 |
| 10,000 queries | $44.00 | $7.50 | $165.00 |

## Recommendation
**Claude Haiku 4.5** — best quality-to-cost ratio for this use case.

1. At ~$0.0044/query, budget-friendly while delivering significantly better Fortran comprehension than GPT-4o-mini or open-source alternatives.
2. 200K context window provides the most headroom for assembling retrieved chunks — direct quality advantage over 128K options.
3. First-token latency ~300-500ms plus streaming keeps total UX well within 3-second target.
4. Already in Anthropic ecosystem (SkillShed uses Claude Code), reducing vendor sprawl.

**Upgrade path:** Use Haiku for dev/testing, switch to Sonnet 4 for the demo video to showcase highest-quality answers.

**Fallback:** GPT-4o-mini if budget becomes extremely constrained (~6x cheaper, but noticeably shallower on dependency mapping and business logic).

Trade-off: Haiku is slightly less deep than Sonnet/GPT-4o on complex reasoning, but the 4x cost savings and faster latency make it the right default. The upgrade path to Sonnet is trivial (same SDK, change model name).

## Decision Confirmation
- [x] User confirmed on 2026-03-02
