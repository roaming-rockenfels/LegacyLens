# Technical Research: RAG Framework

## Decision
Whether to use LangChain, LlamaIndex, or a custom pipeline for the RAG system.

## Constraints
- Non-OpenAI stack: Voyage Code 2 (embeddings) + Claude Haiku 4.5 (LLM) + Pinecone (vector DB)
- Custom Fortran chunking required (no framework supports Fortran)
- MVP in 24 hours, full project in 1 week
- MCP server upgrade path needed later
- Python/FastAPI backend

## Candidates

### LangChain
- **Voyage integration:** `langchain-voyageai` wrapper exists but lags behind SDK updates
- **Pinecone integration:** `langchain-pinecone` works but imposes Document schema that conflicts with custom Fortran metadata
- **Claude integration:** `langchain-anthropic` works but adds indirection over the clean `anthropic` SDK
- **Learning curve:** High — Chains, Runnables, LCEL, callback handlers, output parsers. 1-2 days just to understand which abstractions to use. Every tutorial assumes OpenAI.
- **Custom chunking:** Hurts — must subclass `TextSplitter`, no Fortran support, conforming to interface contract for no benefit
- **Saves:** ~30 lines of retrieval chain wiring
- **Adds:** ~200+ lines of framework config, ~50 transitive dependencies, version conflict risk
- **MCP upgrade:** Constrains it — chain abstractions need unwrapping to expose as MCP tools
- **Pros:** Largest community (~100K GitHub stars), LangSmith for observability
- **Cons:** Off the golden path with Voyage+Claude, massive abstraction surface, fragmented docs across v0.2/v0.3/modular

### LlamaIndex
- **Voyage integration:** `llama-index-embeddings-voyageai` wrapper, thinner than LangChain's
- **Pinecone integration:** `VectorStoreIndex` + `llama-index-vector-stores-pinecone` — useful abstraction but forces node/document model
- **Claude integration:** `llama-index-llms-anthropic` wrapper
- **Learning curve:** Medium-High — Index, Node, QueryEngine, Retriever, Synthesizer concepts. Purpose-built for RAG (better than LangChain) but still requires internalization.
- **Custom chunking:** Neutral — `NodeParser` subclass is cleaner than LangChain, but wrapping Fortran chunks in `TextNode` is busywork. `CodeSplitter` has tree-sitter but Fortran 77 support is incomplete.
- **Saves:** ~40 lines (the `from_documents()` → `as_query_engine()` pattern)
- **Adds:** ~30 dependencies, node/document mapping ceremony
- **MCP upgrade:** Slightly constrains — query engine encapsulates the full pipeline, extracting just retrieval requires reaching into internals
- **Pros:** Better focused on RAG than LangChain, evaluation modules, ~40K GitHub stars
- **Cons:** Non-OpenAI integration docs are sparse, node schema doesn't fit custom code chunks well

### Custom Pipeline (No Framework)
- **Integration:** Direct SDK calls — `voyageai`, `pinecone`, `anthropic`. Three packages, three clean APIs, no wrappers.
- **Learning curve:** Lowest — learn 3 APIs with 1-2 page quickstarts each.
- **Custom chunking:** Best — Fortran chunker is a pure Python module with no interface contracts. Pluggable design is simpler without framework abstractions to conform to.
- **Total code:** ~150-200 lines for the full RAG pipeline (embed → store → retrieve → generate)
- **MCP upgrade:** Unconstrained — retrieval function is a plain async Python function, directly callable as MCP tool
- **Pros:** Full transparency, minimal dependencies, predictable development time, every line understood
- **Cons:** No automatic tracing (add ~20 lines of logging), no built-in evaluation (add ~50 lines of test harness)

## Pro-Con Matrix
| Dimension | LangChain | LlamaIndex | Custom |
|-----------|:---------:|:----------:|:------:|
| Time to first working RAG | 4-8 hours | 3-6 hours | 2-4 hours |
| Learning curve | High | Medium-High | Low |
| Fortran chunking fit | Poor | Neutral | Best |
| Dependency footprint | ~50 packages | ~30 packages | 3 SDKs |
| MCP upgrade path | Constrained | Slightly constrained | Unconstrained |
| Non-OpenAI stack fit | Poor | Fair | Best |
| Debug transparency | Low (deep stacks) | Medium | Full |
| Observability | LangSmith (paid) | LlamaTrace (paid) | DIY (~50 lines) |

## Recommendation
**Custom pipeline (no framework).** The three SDKs compose trivially, the custom Fortran chunker works natively, and the total RAG pipeline is ~150-200 lines — comparable to framework setup code. The MCP upgrade path is cleanest. For a 24-hour MVP, every hour of framework debugging is an hour not spent on the actual product.

Trade-off: No automatic tracing or built-in evaluation. Mitigated with ~70 lines of logging + test harness code.

**When to reconsider:** If the project later needs complex multi-step agent workflows, multiple retrieval strategies with routing, or production-grade distributed tracing — none of which apply to current scope.

## Decision Confirmation
- [x] User confirmed on 2026-03-02
