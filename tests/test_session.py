"""Tests for ChatSession multi-turn conversation with LLM tool-use retrieval."""

from unittest.mock import patch, MagicMock

import pytest


MOCK_CHUNKS = [
    {
        "id": "fortran:dgesv",
        "score": 0.85,
        "metadata": {
            "unit_name": "DGESV",
            "unit_type": "subroutine",
            "file_path": "dgesv.f",
            "language": "fortran",
            "purpose": "Solves Ax=B via LU factorization",
            "start_line": 1,
            "end_line": 178,
            "parameters": ["N", "NRHS", "A", "LDA", "IPIV", "B", "LDB", "INFO"],
            "calls": ["DGETRF", "DGETRS", "XERBLA"],
        },
    },
]

# Helper: build a tool-call message the mock LLM would return.
def _tool_call_msg(query: str = "DGESV error handling") -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_abc123",
                "type": "function",
                "function": {
                    "name": "search_codebase",
                    "arguments": f'{{"query": "{query}"}}',
                },
            }
        ],
    }


# Helper: build a direct-answer message (no tool calls).
def _direct_answer_msg(text: str = "The routine handles errors by ...") -> dict:
    return {"role": "assistant", "content": text}


@patch("legacylens.rag.session._chat_completion_with_client", return_value="Answer 1")
@patch("legacylens.rag.session.retrieve", return_value=MOCK_CHUNKS)
def test_first_turn_retrieves_without_tool_use(mock_retrieve, mock_chat):
    """First ask() calls retrieve() directly — no tool-use call."""
    from legacylens.rag.session import ChatSession

    with ChatSession() as s:
        answer = s.ask("What does DGESV do?")

    mock_retrieve.assert_called_once()
    mock_chat.assert_called_once()
    assert answer == "Answer 1"


@patch("legacylens.rag.session._chat_completion_with_client", return_value="Full answer")
@patch("legacylens.rag.session.retrieve", return_value=MOCK_CHUNKS)
@patch("legacylens.rag.session._chat_completion_with_tools")
def test_tool_call_triggers_retrieval(mock_tools, mock_retrieve, mock_chat):
    """When LLM returns tool_calls, retrieve() is called with the model's query."""
    from legacylens.rag.session import ChatSession

    mock_tools.return_value = _tool_call_msg("DGESV error handling")

    with ChatSession() as s:
        # First turn — always retrieves.
        s.ask("What does DGESV do?")
        assert mock_retrieve.call_count == 1

        # Second turn — model requests search.
        answer = s.ask("How does it handle errors?")
        assert mock_retrieve.call_count == 2
        assert answer == "Full answer"


@patch("legacylens.rag.session._chat_completion_with_client", return_value="First answer")
@patch("legacylens.rag.session.retrieve", return_value=MOCK_CHUNKS)
@patch("legacylens.rag.session._chat_completion_with_tools")
def test_direct_answer_skips_retrieval(mock_tools, mock_retrieve, mock_chat):
    """When LLM answers directly (no tool_calls), retrieve() is NOT called again."""
    from legacylens.rag.session import ChatSession

    mock_tools.return_value = _direct_answer_msg("It handles errors by checking INFO.")

    with ChatSession() as s:
        s.ask("What does DGESV do?")
        assert mock_retrieve.call_count == 1

        answer = s.ask("How does it handle errors?")
        assert mock_retrieve.call_count == 1  # NOT called again
        assert answer == "It handles errors by checking INFO."


@patch("legacylens.rag.session._chat_completion_with_client", return_value="Answer")
@patch("legacylens.rag.session.retrieve", return_value=MOCK_CHUNKS)
@patch("legacylens.rag.session._chat_completion_with_tools")
def test_tool_call_query_passed_to_retrieve(mock_tools, mock_retrieve, mock_chat):
    """The model's reformulated query (not raw user input) is passed to retrieve()."""
    from legacylens.rag.session import ChatSession

    mock_tools.return_value = _tool_call_msg("precision differences in DGESV")

    with ChatSession() as s:
        s.ask("What does DGESV do?")
        s.ask("what kind of unintended behavior does it create?")

    # Second retrieve call should use the model's reformulated query.
    assert mock_retrieve.call_count == 2
    second_call_args = mock_retrieve.call_args_list[1]
    assert second_call_args[0][0] == "precision differences in DGESV"


@patch("legacylens.rag.session._chat_completion_with_client", return_value="Answer")
@patch("legacylens.rag.session.retrieve", return_value=MOCK_CHUNKS)
@patch("legacylens.rag.session._chat_completion_with_tools")
def test_history_accumulates_with_tool_messages(mock_tools, mock_retrieve, mock_chat):
    """History contains user, assistant+tool_calls, tool, assistant messages."""
    from legacylens.rag.session import ChatSession

    mock_tools.return_value = _tool_call_msg()

    with ChatSession() as s:
        s.ask("What does DGESV do?")
        assert len(s._history) == 2  # user + assistant

        s.ask("Tell me more")
        # user + assistant(tool_calls) + tool + assistant = 4 more
        assert len(s._history) == 6
        assert s._history[2]["role"] == "user"
        assert "tool_calls" in s._history[3]
        assert s._history[4]["role"] == "tool"
        assert s._history[5]["role"] == "assistant"


@patch("legacylens.rag.session._chat_completion_with_client", return_value="Answer")
@patch("legacylens.rag.session.retrieve", return_value=MOCK_CHUNKS)
def test_reset_clears_state(mock_retrieve, mock_chat):
    """reset() clears history and chunks; next ask() does first-turn path."""
    from legacylens.rag.session import ChatSession

    with ChatSession() as s:
        s.ask("What does DGESV do?")
        assert mock_retrieve.call_count == 1
        s.reset()
        assert len(s._history) == 0
        assert len(s._last_chunks) == 0
        s.ask("What does DGESV do?")
        assert mock_retrieve.call_count == 2


def test_context_manager_closes_client():
    """Context manager calls close() on exit."""
    from legacylens.rag.session import ChatSession

    session = ChatSession()
    client = session._client
    with patch.object(client, "close") as mock_close:
        with session:
            pass
        mock_close.assert_called_once()


@patch("legacylens.rag.session._chat_completion_with_client", return_value="Fallback answer")
@patch("legacylens.rag.session.retrieve", return_value=MOCK_CHUNKS)
@patch("legacylens.rag.session._chat_completion_with_tools")
def test_malformed_tool_args_fallback(mock_tools, mock_retrieve, mock_chat):
    """Bad JSON in tool args falls back to using the raw user question."""
    from legacylens.rag.session import ChatSession

    mock_tools.return_value = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_bad",
                "type": "function",
                "function": {
                    "name": "search_codebase",
                    "arguments": "not valid json{{{",
                },
            }
        ],
    }

    with ChatSession() as s:
        s.ask("What does DGESV do?")  # first turn
        s.ask("tell me about errors")  # second turn with bad args

    # Should fall back to raw question.
    assert mock_retrieve.call_count == 2
    second_call_args = mock_retrieve.call_args_list[1]
    assert second_call_args[0][0] == "tell me about errors"


# ── Empty retrieval tests ────────────────────────────────────────────


@patch("legacylens.rag.session._chat_completion_with_client", return_value="Fallback with citations")
@patch("legacylens.rag.session.retrieve", return_value=[])
def test_empty_retrieval_first_turn(mock_retrieve, mock_chat):
    """When retrieval returns empty, user content uses fallback message."""
    from legacylens.rag.session import ChatSession

    with ChatSession() as s:
        answer = s.ask("What is LAPACK?")

    assert answer == "Fallback with citations"
    # Check user message contains fallback text
    user_msg = s._history[0]["content"]
    assert "No code chunks" in user_msg
    assert "expertise" in user_msg


@patch("legacylens.rag.session._chat_completion_with_client", return_value="Tool fallback")
@patch("legacylens.rag.session.retrieve")
@patch("legacylens.rag.session._chat_completion_with_tools")
def test_empty_retrieval_tool_call(mock_tools, mock_retrieve, mock_chat):
    """When tool-call retrieval returns empty, tool content uses fallback."""
    from legacylens.rag.session import ChatSession

    # First call returns chunks, second returns empty
    mock_retrieve.side_effect = [MOCK_CHUNKS, []]
    mock_tools.return_value = _tool_call_msg("something obscure")

    with ChatSession() as s:
        s.ask("What does DGESV do?")
        s.ask("Tell me about something obscure")

    # The tool message should have fallback content
    tool_msg = s._history[4]
    assert tool_msg["role"] == "tool"
    assert "No relevant code chunks" in tool_msg["content"]


# ── Streaming tests ──────────────────────────────────────────────────


@patch("legacylens.rag.session._chat_completion_stream")
@patch("legacylens.rag.session.retrieve", return_value=MOCK_CHUNKS)
def test_ask_stream_first_turn(mock_retrieve, mock_stream):
    """ask_stream() yields deltas on first turn."""
    from legacylens.rag.session import ChatSession

    mock_stream.return_value = iter(["Hello", " ", "world"])

    with ChatSession() as s:
        deltas = list(s.ask_stream("What does DGESV do?"))

    assert deltas == ["Hello", " ", "world"]
    mock_retrieve.assert_called_once()
    mock_stream.assert_called_once()
    # History should contain the full answer
    assert s._history[-1]["content"] == "Hello world"


@patch("legacylens.rag.session._chat_completion_stream")
@patch("legacylens.rag.session.retrieve", return_value=MOCK_CHUNKS)
@patch("legacylens.rag.session._chat_completion_with_tools")
def test_ask_stream_tool_call(mock_tools, mock_retrieve, mock_stream):
    """ask_stream() handles tool calls then streams final answer."""
    from legacylens.rag.session import ChatSession

    mock_tools.return_value = _tool_call_msg("error handling")
    mock_stream.return_value = iter(["Streamed", " answer"])

    with ChatSession() as s:
        s.ask("What does DGESV do?")  # first turn (non-streaming)
        deltas = list(s.ask_stream("How does it handle errors?"))

    assert deltas == ["Streamed", " answer"]
    assert mock_retrieve.call_count == 2


@patch("legacylens.rag.session._chat_completion_with_client", return_value="First answer")
@patch("legacylens.rag.session.retrieve", return_value=MOCK_CHUNKS)
@patch("legacylens.rag.session._chat_completion_with_tools")
def test_ask_stream_direct_answer(mock_tools, mock_retrieve, mock_chat):
    """ask_stream() yields full content as single chunk when model answers directly."""
    from legacylens.rag.session import ChatSession

    mock_tools.return_value = _direct_answer_msg("Direct answer")

    with ChatSession() as s:
        s.ask("What does DGESV do?")  # first turn
        deltas = list(s.ask_stream("Follow up"))

    assert deltas == ["Direct answer"]
    assert mock_retrieve.call_count == 1  # no new retrieval
