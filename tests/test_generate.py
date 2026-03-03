"""Tests for answer generation."""

from unittest.mock import patch, MagicMock

import pytest


MOCK_RESULTS = [
    {
        "id": "fortran:dgesv",
        "score": 0.85,
        "metadata": {
            "unit_name": "DGESV",
            "unit_type": "subroutine",
            "file_path": "dgesv.f",
            "language": "fortran",
            "purpose": "Computes the solution to a system of linear equations",
            "start_line": 1,
            "end_line": 178,
            "parameters": ["N", "NRHS", "A", "LDA", "IPIV", "B", "LDB", "INFO"],
            "calls": ["DGETRF", "DGETRS", "XERBLA"],
        },
    },
]


def test_build_context():
    from legacylens.rag.generate import build_context

    context = build_context(MOCK_RESULTS)
    assert "DGESV" in context
    assert "0.850" in context
    assert "dgesv.f" in context
    assert "DGETRF" in context


def test_build_context_empty():
    from legacylens.rag.generate import build_context

    context = build_context([])
    assert context == ""


@patch("legacylens.rag.generate._chat_completion", return_value="DGESV solves linear systems using LU factorization.")
def test_generate_answer_calls_openrouter(mock_chat):
    from legacylens.rag.generate import generate_answer

    answer = generate_answer("What does DGESV do?", MOCK_RESULTS)

    assert "DGESV" in answer
    mock_chat.assert_called_once()
    call_args = mock_chat.call_args
    messages = call_args[0][0]
    assert any("DGESV" in m["content"] for m in messages)


@patch("legacylens.rag.generate._chat_completion", return_value="Test answer")
def test_generate_answer_modes(mock_chat):
    from legacylens.rag.generate import generate_answer

    for mode in ("explain", "deps", "docs", "business_logic"):
        generate_answer("test", MOCK_RESULTS, mode=mode)
        call_args = mock_chat.call_args
        messages = call_args[0][0]
        user_msg = [m for m in messages if m["role"] == "user"][0]["content"]
        if mode == "deps":
            assert "dependency" in user_msg.lower() or "call" in user_msg.lower()
        elif mode == "docs":
            assert "documentation" in user_msg.lower()
