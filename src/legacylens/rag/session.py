"""Conversational chat session — multi-turn Q&A with LLM-driven retrieval decisions."""

from __future__ import annotations

import json
from collections.abc import Iterator

import httpx

from legacylens.rag.generate import (
    SYSTEM_PROMPT,
    build_context,
    _chat_completion_with_client,
    _chat_completion_with_tools,
    _chat_completion_stream,
    DEFAULT_MODEL,
)
from legacylens.rag.retrieve import retrieve

_SEARCH_CODEBASE_TOOL = {
    "type": "function",
    "function": {
        "name": "search_codebase",
        "description": (
            "Search the LAPACK Fortran codebase for relevant code chunks. "
            "Call this tool when you need additional context from the codebase to "
            "answer the user's question — for example, when the question asks about "
            "a new routine, concept, or topic not covered by the existing conversation "
            "context. Do NOT call this tool if you can already answer from the "
            "conversation history."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "A natural-language search query describing what code or "
                        "information to retrieve from the codebase."
                    ),
                },
            },
            "required": ["query"],
        },
    },
}


class ChatSession:
    """Multi-turn chat session with LLM-driven retrieval decisions.

    First query always retrieves.  Subsequent queries let the LLM decide
    whether to call the ``search_codebase`` tool (triggering retrieval)
    or answer directly from conversation history.
    """

    def __init__(self, *, top_k: int = 5, model: str = DEFAULT_MODEL) -> None:
        self._top_k = top_k
        self._model = model
        self._history: list[dict] = []
        self._last_chunks: list[dict] = []
        self._client: httpx.Client = httpx.Client(timeout=60)

    # -- context manager --------------------------------------------------

    def __enter__(self) -> ChatSession:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- public API -------------------------------------------------------

    def ask(self, question: str, *, top_k: int | None = None) -> str:
        """Ask a question.  Returns the LLM answer as a string."""
        k = top_k or self._top_k

        if not self._last_chunks:
            return self._ask_with_retrieval(question, k)

        # Let the LLM decide whether retrieval is needed.
        self._history.append({"role": "user", "content": question})
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self._history

        msg = _chat_completion_with_tools(
            self._client, messages, [_SEARCH_CODEBASE_TOOL], model=self._model,
        )

        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            # Model answered directly — no retrieval needed.
            answer = msg.get("content", "")
            self._history.append({"role": "assistant", "content": answer})
            return answer

        # Model wants to search — parse the query from tool call args.
        tool_call = tool_calls[0]
        try:
            args = json.loads(tool_call["function"]["arguments"])
            search_query = args["query"]
        except (json.JSONDecodeError, KeyError):
            search_query = question  # fallback to raw user question

        self._last_chunks = retrieve(search_query, top_k=k)
        context = build_context(self._last_chunks)

        # Append assistant message with tool_calls, then tool result.
        self._history.append(msg)
        self._history.append({
            "role": "tool",
            "tool_call_id": tool_call["id"],
            "content": context if context else "No relevant code chunks found. Answer from your knowledge and cite sources.",
        })

        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self._history

        answer = _chat_completion_with_client(
            self._client, messages, model=self._model,
        )
        self._history.append({"role": "assistant", "content": answer})
        return answer

    def ask_stream(self, question: str, *, top_k: int | None = None) -> Iterator[str]:
        """Ask a question, yielding answer content deltas as they arrive."""
        k = top_k or self._top_k

        if not self._last_chunks:
            yield from self._ask_with_retrieval_stream(question, k)
            return

        # Let the LLM decide whether retrieval is needed.
        self._history.append({"role": "user", "content": question})
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self._history

        msg = _chat_completion_with_tools(
            self._client, messages, [_SEARCH_CODEBASE_TOOL], model=self._model,
        )

        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            # Model answered directly — yield as single chunk (fast path).
            answer = msg.get("content", "")
            self._history.append({"role": "assistant", "content": answer})
            yield answer
            return

        # Model wants to search — execute tool then stream final answer.
        tool_call = tool_calls[0]
        try:
            args = json.loads(tool_call["function"]["arguments"])
            search_query = args["query"]
        except (json.JSONDecodeError, KeyError):
            search_query = question

        self._last_chunks = retrieve(search_query, top_k=k)
        context = build_context(self._last_chunks)

        self._history.append(msg)
        self._history.append({
            "role": "tool",
            "tool_call_id": tool_call["id"],
            "content": context if context else "No relevant code chunks found. Answer from your knowledge and cite sources.",
        })

        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self._history

        full_answer = []
        for delta in _chat_completion_stream(self._client, messages, model=self._model):
            full_answer.append(delta)
            yield delta
        self._history.append({"role": "assistant", "content": "".join(full_answer)})

    def reset(self) -> None:
        """Clear conversation history and cached chunks.  Keeps the HTTP client alive."""
        self._history = []
        self._last_chunks = []

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    # -- internals --------------------------------------------------------

    @staticmethod
    def _build_user_content(context: str, question: str) -> str:
        """Build user message content, conditional on whether context is empty."""
        if context:
            return (
                f"Here are relevant code chunks from the LAPACK Fortran codebase:\n\n"
                f"{context}\n\n"
                f"Question: {question}\n\n"
                f"Provide a clear explanation of what this code does, how it works, and why."
            )
        return (
            f"No code chunks were found in the codebase for this query. "
            f"Answer using your expertise in LAPACK and Fortran, and cite authoritative web sources.\n\n"
            f"Question: {question}\n\n"
            f"Provide a clear explanation."
        )

    def _ask_with_retrieval(self, question: str, top_k: int) -> str:
        """First-turn path: always retrieve, no tool-use overhead."""
        self._last_chunks = retrieve(question, top_k=top_k)
        context = build_context(self._last_chunks)
        user_content = self._build_user_content(context, question)

        self._history.append({"role": "user", "content": user_content})
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self._history

        answer = _chat_completion_with_client(
            self._client, messages, model=self._model,
        )
        self._history.append({"role": "assistant", "content": answer})
        return answer

    def _ask_with_retrieval_stream(self, question: str, top_k: int) -> Iterator[str]:
        """First-turn streaming path: always retrieve, no tool-use overhead."""
        self._last_chunks = retrieve(question, top_k=top_k)
        context = build_context(self._last_chunks)
        user_content = self._build_user_content(context, question)

        self._history.append({"role": "user", "content": user_content})
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self._history

        full_answer = []
        for delta in _chat_completion_stream(self._client, messages, model=self._model):
            full_answer.append(delta)
            yield delta
        self._history.append({"role": "assistant", "content": "".join(full_answer)})
