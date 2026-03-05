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


@patch("legacylens.rag.generate.read_source_snippet", return_value="      SUBROUTINE DGESV( N, NRHS )")
def test_build_context_includes_source(mock_snippet):
    """build_context() appends actual source code with line numbers when snippet is available."""
    from legacylens.rag.generate import build_context

    context = build_context(MOCK_RESULTS)
    assert "Source:" in context
    assert "```fortran" in context
    assert "SUBROUTINE DGESV" in context
    # Line numbers should be present (start_line=1)
    assert "     1 | " in context
    mock_snippet.assert_called_once_with("dgesv.f", 1, 178)


@patch("legacylens.rag.generate.read_source_snippet", return_value="")
def test_build_context_omits_empty_source(mock_snippet):
    """build_context() does not add a Source block when snippet is empty."""
    from legacylens.rag.generate import build_context

    context = build_context(MOCK_RESULTS)
    assert "Source:" not in context


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


@patch("legacylens.rag.generate._chat_completion", return_value="Fallback answer with citations")
def test_generate_answer_empty_results(mock_chat):
    """generate_answer() with empty results uses fallback user message."""
    from legacylens.rag.generate import generate_answer

    answer = generate_answer("What is LAPACK?", [])
    assert answer == "Fallback answer with citations"
    mock_chat.assert_called_once()
    messages = mock_chat.call_args[0][0]
    user_msg = [m for m in messages if m["role"] == "user"][0]["content"]
    assert "No code chunks" in user_msg
    assert "expertise" in user_msg


@patch("legacylens.rag.generate.OPENROUTER_API_KEY", return_value="test-key")
def test_chat_completion_with_client(mock_key):
    """_chat_completion_with_client uses the provided httpx.Client."""
    from legacylens.rag.generate import _chat_completion_with_client

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Hello from client"}}]
    }
    mock_response.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.post.return_value = mock_response

    messages = [{"role": "user", "content": "test"}]
    result = _chat_completion_with_client(mock_client, messages)

    assert result == "Hello from client"
    mock_client.post.assert_called_once()
    call_kwargs = mock_client.post.call_args
    assert "chat/completions" in call_kwargs[0][0]


@patch("legacylens.rag.generate.OPENROUTER_API_KEY", return_value="test-key")
def test_chat_completion_with_tools(mock_key):
    """_chat_completion_with_tools passes tools and returns raw message dict."""
    from legacylens.rag.generate import _chat_completion_with_tools

    raw_msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_123",
                "type": "function",
                "function": {"name": "search_codebase", "arguments": '{"query": "test"}'},
            }
        ],
    }
    mock_response = MagicMock()
    mock_response.json.return_value = {"choices": [{"message": raw_msg}]}
    mock_response.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.post.return_value = mock_response

    tools = [{"type": "function", "function": {"name": "search_codebase"}}]
    messages = [{"role": "user", "content": "test"}]
    result = _chat_completion_with_tools(mock_client, messages, tools)

    assert result == raw_msg
    assert result["tool_calls"][0]["function"]["name"] == "search_codebase"
    mock_client.post.assert_called_once()
    # Verify tools were included in the payload.
    call_kwargs = mock_client.post.call_args
    payload = call_kwargs[1]["json"] if "json" in call_kwargs[1] else call_kwargs[0][1]
    assert "tools" in payload


@patch("legacylens.rag.generate.OPENROUTER_API_KEY", return_value="test-key")
def test_chat_completion_stream(mock_key):
    """_chat_completion_stream yields content deltas from SSE lines."""
    from legacylens.rag.generate import _chat_completion_stream

    sse_lines = [
        'data: {"choices":[{"delta":{"content":"Hello"}}]}',
        'data: {"choices":[{"delta":{"content":" world"}}]}',
        'data: [DONE]',
    ]

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.iter_lines.return_value = iter(sse_lines)
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)

    mock_client = MagicMock()
    mock_client.stream.return_value = mock_resp

    messages = [{"role": "user", "content": "test"}]
    deltas = list(_chat_completion_stream(mock_client, messages))

    assert deltas == ["Hello", " world"]
    mock_client.stream.assert_called_once()
