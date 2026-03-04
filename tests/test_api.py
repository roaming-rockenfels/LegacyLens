"""Tests for FastAPI server."""

from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from legacylens.api.server import app

client = TestClient(app)

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


@patch("legacylens.rag.retrieve.retrieve", return_value=[])
def test_query_no_results(mock_retrieve):
    resp = client.post("/query", json={"question": "nonexistent"})
    assert resp.status_code == 404


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
