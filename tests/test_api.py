"""Tests for FastAPI server."""

import json
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from legacylens.api.server import app, _sessions, _lock

client = TestClient(app)


@pytest.fixture(autouse=True)
def _clear_sessions():
    """Clear the server session store between tests."""
    yield
    with _lock:
        for entry in _sessions.values():
            entry["session"].close()
        _sessions.clear()

# Reusable mock metadata with start_line/end_line
_DGESV_METADATA = {
    "unit_name": "DGESV",
    "unit_type": "subroutine",
    "file_path": "dgesv.f",
    "language": "fortran",
    "purpose": "Solves Ax=B",
    "parameters": ["N", "NRHS"],
    "calls": ["DGETRF"],
    "start_line": 1,
    "end_line": 42,
}


def test_homepage():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "LegacyLens" in resp.text


def test_easter_egg():
    resp = client.get("/easter-egg")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "FORTRAN" in resp.text


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@patch("legacylens.rag.source_reader.read_source_snippet", return_value="      SUBROUTINE DGESV( N, NRHS, A, LDA, IPIV, B, LDB, INFO )")
@patch("legacylens.rag.generate.generate_answer", return_value="DGESV solves Ax=B using LU.")
@patch("legacylens.rag.retrieve.retrieve", return_value=[
    {
        "id": "fortran:dgesv",
        "score": 0.85,
        "metadata": _DGESV_METADATA,
    }
])
def test_query_endpoint(mock_retrieve, mock_generate, mock_snippet):
    resp = client.post("/query", json={
        "question": "How does DGESV work?",
        "top_k": 3,
        "mode": "explain",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["question"] == "How does DGESV work?"
    assert "DGESV" in data["answer"]
    assert len(data["chunks"]) == 1
    chunk = data["chunks"][0]
    assert chunk["unit_name"] == "DGESV"
    assert chunk["start_line"] == 1
    assert chunk["end_line"] == 42
    assert "SUBROUTINE DGESV" in chunk["snippet"]


@patch("legacylens.rag.source_reader.read_source_snippet", return_value="")
@patch("legacylens.rag.generate.generate_answer", return_value="No source available.")
@patch("legacylens.rag.retrieve.retrieve", return_value=[
    {
        "id": "fortran:dgesv",
        "score": 0.85,
        "metadata": _DGESV_METADATA,
    }
])
def test_query_endpoint_missing_source(mock_retrieve, mock_generate, mock_snippet):
    """API returns empty snippet when source file is not available."""
    resp = client.post("/query", json={"question": "DGESV?", "top_k": 1, "mode": "explain"})
    assert resp.status_code == 200
    chunk = resp.json()["chunks"][0]
    assert chunk["start_line"] == 1
    assert chunk["end_line"] == 42
    assert chunk["snippet"] == ""


@patch("legacylens.rag.generate.generate_answer", return_value="General LAPACK answer with citations.")
@patch("legacylens.rag.retrieve.retrieve", return_value=[])
def test_query_empty_results_returns_answer(mock_retrieve, mock_generate):
    """/query with no retrieval results returns 200 with fallback answer (no 404)."""
    resp = client.post("/query", json={"question": "What is LAPACK?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "General LAPACK answer with citations."
    assert data["chunks"] == []


@patch("legacylens.rag.retrieve.retrieve", return_value=[
    {"id": "test", "score": 0.5, "metadata": {"unit_name": "TEST", "unit_type": "sub", "file_path": "t.f", "language": "fortran", "purpose": ""}}
])
def test_search_endpoint(mock_retrieve):
    resp = client.get("/search", params={"question": "test", "top_k": 2})
    assert resp.status_code == 200
    assert len(resp.json()["results"]) == 1


@patch("legacylens.rag.storage.get_index_stats", return_value={"total_vector_count": 100, "dimension": 1024})
def test_stats_endpoint(mock_stats):
    resp = client.get("/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_vector_count"] == 100
    assert data["dimension"] == 1024


# ── Chat endpoint tests ─────────────────────────────────────────────


@patch("legacylens.rag.session.ChatSession.ask", return_value="Hello from session")
def test_chat_creates_session(mock_ask):
    resp = client.post("/chat", json={"message": "What is DGESV?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_new_session"] is True
    assert len(data["session_id"]) == 12
    assert data["question"] == "What is DGESV?"
    assert data["answer"] == "Hello from session"


@patch("legacylens.rag.session.ChatSession.ask", return_value="Follow-up answer")
def test_chat_continues_session(mock_ask):
    # First message — creates session
    resp1 = client.post("/chat", json={"message": "First question"})
    sid = resp1.json()["session_id"]
    assert resp1.json()["is_new_session"] is True

    # Second message — continues session
    resp2 = client.post("/chat", json={"message": "Follow up", "session_id": sid})
    data = resp2.json()
    assert data["is_new_session"] is False
    assert data["session_id"] == sid


@patch("legacylens.rag.session.ChatSession.ask", return_value="New session answer")
def test_chat_invalid_session_creates_new(mock_ask):
    resp = client.post("/chat", json={"message": "Hello", "session_id": "bogus_id_1234"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_new_session"] is True
    assert data["session_id"] != "bogus_id_1234"


@patch("legacylens.rag.session.ChatSession.ask", return_value="To be deleted")
def test_delete_chat_session(mock_ask):
    resp = client.post("/chat", json={"message": "Setup"})
    sid = resp.json()["session_id"]

    del_resp = client.delete(f"/chat/{sid}")
    assert del_resp.status_code == 200
    assert del_resp.json()["session_id"] == sid

    # Session should be gone — new POST with old ID creates new session
    resp2 = client.post("/chat", json={"message": "After delete", "session_id": sid})
    assert resp2.json()["is_new_session"] is True
    assert resp2.json()["session_id"] != sid


def test_delete_nonexistent_session():
    resp = client.delete("/chat/fake_session_id")
    assert resp.status_code == 404


# ── Streaming chat endpoint tests ────────────────────────────────────


@patch("legacylens.rag.session.ChatSession.ask_stream")
def test_chat_stream_endpoint(mock_stream):
    """POST /chat/stream returns SSE events with session, deltas, and done."""
    mock_stream.return_value = iter(["Hello", " ", "world"])

    resp = client.post("/chat/stream", json={"message": "What is DGESV?"})
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    # Parse SSE events
    events = []
    for line in resp.text.strip().split("\n\n"):
        line = line.strip()
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))

    assert len(events) == 6  # session + retrieval + 3 deltas + done
    assert events[0]["type"] == "session"
    assert "session_id" in events[0]
    assert events[0]["is_new_session"] is True
    assert events[1]["type"] == "retrieval"
    assert "chunk_count" in events[1]
    assert events[2] == {"type": "delta", "content": "Hello"}
    assert events[3] == {"type": "delta", "content": " "}
    assert events[4] == {"type": "delta", "content": "world"}
    assert events[5]["type"] == "done"


@patch("legacylens.rag.session.ChatSession.ask_stream")
def test_chat_stream_continues_session(mock_stream):
    """POST /chat/stream with existing session_id continues session."""
    mock_stream.return_value = iter(["First"])
    resp1 = client.post("/chat/stream", json={"message": "Q1"})
    events1 = []
    for line in resp1.text.strip().split("\n\n"):
        line = line.strip()
        if line.startswith("data: "):
            events1.append(json.loads(line[6:]))
    sid = events1[0]["session_id"]

    mock_stream.return_value = iter(["Second"])
    resp2 = client.post("/chat/stream", json={"message": "Q2", "session_id": sid})
    events2 = []
    for line in resp2.text.strip().split("\n\n"):
        line = line.strip()
        if line.startswith("data: "):
            events2.append(json.loads(line[6:]))

    assert events2[0]["type"] == "session"
    assert events2[0]["session_id"] == sid
    assert events2[0]["is_new_session"] is False
