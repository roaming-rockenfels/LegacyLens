"""FastAPI server — REST API + Swagger UI for LegacyLens."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="LegacyLens API",
    description="RAG-powered API for querying legacy Fortran codebases (LAPACK).",
    version="0.1.0",
)


class QueryRequest(BaseModel):
    question: str = Field(..., description="Natural language question about the codebase")
    top_k: int = Field(5, ge=1, le=20, description="Number of results to retrieve")
    mode: str = Field("explain", description="Response mode: explain, deps, docs, business_logic")


class ChunkResult(BaseModel):
    id: str
    score: float
    unit_name: str
    unit_type: str
    file_path: str
    language: str
    purpose: str
    parameters: list[str] = []
    calls: list[str] = []


class QueryResponse(BaseModel):
    question: str
    answer: str
    chunks: list[ChunkResult]
    mode: str


class StatsResponse(BaseModel):
    total_vector_count: int
    dimension: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query_codebase(req: QueryRequest):
    from legacylens.rag.retrieve import retrieve
    from legacylens.rag.generate import generate_answer

    results = retrieve(req.question, top_k=req.top_k)

    if not results:
        raise HTTPException(status_code=404, detail="No relevant chunks found")

    answer = generate_answer(req.question, results, mode=req.mode)

    chunks = []
    for match in results:
        meta = match.get("metadata", {})
        chunks.append(
            ChunkResult(
                id=match["id"],
                score=match["score"],
                unit_name=meta.get("unit_name", ""),
                unit_type=meta.get("unit_type", ""),
                file_path=meta.get("file_path", ""),
                language=meta.get("language", ""),
                purpose=meta.get("purpose", ""),
                parameters=meta.get("parameters", []),
                calls=meta.get("calls", []),
            )
        )

    return QueryResponse(
        question=req.question,
        answer=answer,
        chunks=chunks,
        mode=req.mode,
    )


@app.get("/search")
def search(question: str, top_k: int = 5):
    """Search without LLM answer — just retrieval results."""
    from legacylens.rag.retrieve import retrieve

    results = retrieve(question, top_k=top_k)
    return {"question": question, "results": results}


@app.get("/stats", response_model=StatsResponse)
def index_stats():
    from legacylens.rag.storage import get_index_stats

    info = get_index_stats()
    return StatsResponse(
        total_vector_count=info.get("total_vector_count", 0),
        dimension=info.get("dimension", 0),
    )
