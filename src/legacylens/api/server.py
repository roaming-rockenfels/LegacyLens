"""FastAPI server — REST API + Swagger UI for LegacyLens."""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

STATIC_DIR = Path(__file__).parent.parent / "static"

app = FastAPI(
    title="LegacyLens API",
    description="RAG-powered API for querying legacy Fortran codebases (LAPACK).",
    version="0.1.0",
)


# ── Chat session store ──────────────────────────────────────────────
_sessions: dict[str, dict] = {}
_lock = threading.Lock()
_SESSION_TTL = 1800  # 30 minutes


def _cleanup_sessions() -> None:
    """Remove sessions idle longer than TTL. Called under _lock."""
    now = time.monotonic()
    expired = [sid for sid, s in _sessions.items() if now - s["last_used"] > _SESSION_TTL]
    for sid in expired:
        s = _sessions.pop(sid)
        s["session"].close()


# ── Request / Response models ───────────────────────────────────────
class QueryRequest(BaseModel):
    question: str = Field(..., description="Natural language question about the codebase")
    top_k: int = Field(5, ge=1, le=20, description="Number of results to retrieve")
    mode: str = Field("explain", description="Response mode: explain, deps, docs, business_logic")
    no_answer: bool = Field(False, description="Skip LLM answer, return chunks only")
    show_code: bool = Field(False, description="Include source code snippets")
    pin_unit: str | None = Field(None, description="Pin a specific routine/subroutine name to guarantee it appears in results")


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
    start_line: int = 0
    end_line: int = 0
    snippet: str = ""


class QueryResponse(BaseModel):
    question: str
    answer: str
    chunks: list[ChunkResult]
    mode: str


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(None, description="Existing session ID to continue")
    top_k: int = Field(5, ge=1, le=20, description="Number of results to retrieve")
    mode: str = Field("explain", description="Response mode: explain, deps, docs, business_logic")


class ChatResponse(BaseModel):
    session_id: str
    question: str
    answer: str
    is_new_session: bool


class StatsResponse(BaseModel):
    total_vector_count: int
    dimension: int


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def homepage():
    return (STATIC_DIR / "index.html").read_text()


@app.get("/easter-egg", response_class=HTMLResponse, include_in_schema=False)
def easter_egg():
    return (STATIC_DIR / "easter-egg.html").read_text()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query_codebase(req: QueryRequest):
    from legacylens.rag.retrieve import retrieve
    from legacylens.rag.generate import generate_answer

    results = retrieve(req.question, top_k=req.top_k, pin_unit=req.pin_unit)
    answer = "" if req.no_answer else generate_answer(req.question, results, mode=req.mode)

    from legacylens.rag.source_reader import read_source_snippet

    chunks = []
    for match in results:
        meta = match.get("metadata", {})
        fp = meta.get("file_path", "")
        sl = meta.get("start_line", 0)
        el = meta.get("end_line", 0)
        snippet = ""
        if isinstance(sl, int) and isinstance(el, int) and sl and el:
            snippet = read_source_snippet(fp, sl, el)
        chunks.append(
            ChunkResult(
                id=match["id"],
                score=match["score"],
                unit_name=meta.get("unit_name", ""),
                unit_type=meta.get("unit_type", ""),
                file_path=fp,
                language=meta.get("language", ""),
                purpose=meta.get("purpose", ""),
                parameters=meta.get("parameters", []),
                calls=meta.get("calls", []),
                start_line=sl if isinstance(sl, int) else 0,
                end_line=el if isinstance(el, int) else 0,
                snippet=snippet,
            )
        )

    return QueryResponse(
        question=req.question,
        answer=answer,
        chunks=chunks,
        mode=req.mode,
    )


@app.get("/search")
def search(question: str, top_k: int = 5, pin_unit: str | None = None):
    """Search without LLM answer — just retrieval results."""
    from legacylens.rag.retrieve import retrieve

    results = retrieve(question, top_k=top_k, pin_unit=pin_unit)
    return {"question": question, "results": results}


@app.get("/stats", response_model=StatsResponse)
def index_stats():
    from legacylens.rag.storage import get_index_stats

    info = get_index_stats()
    return StatsResponse(
        total_vector_count=info.get("total_vector_count", 0),
        dimension=info.get("dimension", 0),
    )


# ── Chat endpoints ──────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    from legacylens.rag.session import ChatSession

    with _lock:
        _cleanup_sessions()

        is_new = True
        session_id = req.session_id

        if session_id and session_id in _sessions:
            entry = _sessions[session_id]
            is_new = False
        else:
            session_id = uuid.uuid4().hex[:12]
            entry = {
                "session": ChatSession(top_k=req.top_k),
                "last_used": time.monotonic(),
            }
            _sessions[session_id] = entry

        entry["last_used"] = time.monotonic()
        session = entry["session"]

    answer = session.ask(req.message, top_k=req.top_k)

    return ChatResponse(
        session_id=session_id,
        question=req.message,
        answer=answer,
        is_new_session=is_new,
    )


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    from legacylens.rag.session import ChatSession

    with _lock:
        _cleanup_sessions()

        is_new = True
        session_id = req.session_id

        if session_id and session_id in _sessions:
            entry = _sessions[session_id]
            is_new = False
        else:
            session_id = uuid.uuid4().hex[:12]
            entry = {
                "session": ChatSession(top_k=req.top_k),
                "last_used": time.monotonic(),
            }
            _sessions[session_id] = entry

        entry["last_used"] = time.monotonic()
        session = entry["session"]

    def event_generator():
        yield f"data: {json.dumps({'type': 'session', 'session_id': session_id, 'is_new_session': is_new})}\n\n"
        retrieval_sent = False
        for delta in session.ask_stream(req.message, top_k=req.top_k):
            if not retrieval_sent:
                # Retrieval has completed by the time the first delta arrives.
                chunk_count = len(session._last_chunks)
                yield f"data: {json.dumps({'type': 'retrieval', 'chunk_count': chunk_count})}\n\n"
                retrieval_sent = True
            yield f"data: {json.dumps({'type': 'delta', 'content': delta})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.delete("/chat/{session_id}")
def delete_chat_session(session_id: str):
    with _lock:
        entry = _sessions.pop(session_id, None)
    if entry is None:
        raise HTTPException(status_code=404, detail="Session not found")
    entry["session"].close()
    return {"status": "ok", "session_id": session_id}
