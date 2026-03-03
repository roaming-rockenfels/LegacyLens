"""Tests for FastAPI server."""

from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from legacylens.api.server import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@patch("legacylens.rag.generate.generate_answer", return_value="DGESV solves Ax=B using LU.")
@patch("legacylens.rag.retrieve.retrieve", return_value=[
    {
        "id": "fortran:dgesv",
        "score": 0.85,
        "metadata": {
            "unit_name": "DGESV",
            "unit_type": "subroutine",
            "file_path": "dgesv.f",
            "language": "fortran",
            "purpose": "Solves Ax=B",
            "parameters": ["N", "NRHS"],
            "calls": ["DGETRF"],
        },
    }
])
def test_query_endpoint(mock_retrieve, mock_generate):
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
    assert data["chunks"][0]["unit_name"] == "DGESV"


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
