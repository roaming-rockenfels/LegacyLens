"""Answer generation via OpenRouter — synthesizes retrieved chunks into NL answers."""

from __future__ import annotations

import os

import json
from collections.abc import Iterator

import httpx

from legacylens.config import OPENROUTER_API_KEY, LEGACYLENS_MODEL
from legacylens.rag.source_reader import read_source_snippet

# Default model — configurable via LEGACYLENS_MODEL env var.
DEFAULT_MODEL = LEGACYLENS_MODEL()

OPENROUTER_BASE = "https://openrouter.ai/api/v1"


def _chat_completion(
    messages: list[dict],
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4096,
) -> str:
    """Call OpenRouter chat completions API."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY()}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/roaming-rockenfels/LegacyLens",
        "X-Title": "LegacyLens",
    }

    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }

    with httpx.Client(timeout=60) as client:
        resp = client.post(f"{OPENROUTER_BASE}/chat/completions", headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


def _chat_completion_with_client(
    client: httpx.Client,
    messages: list[dict],
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4096,
) -> str:
    """Call OpenRouter chat completions using a caller-provided httpx.Client."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY()}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/roaming-rockenfels/LegacyLens",
        "X-Title": "LegacyLens",
    }
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    resp = client.post(f"{OPENROUTER_BASE}/chat/completions", headers=headers, json=payload)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def _chat_completion_with_tools(
    client: httpx.Client,
    messages: list[dict],
    tools: list[dict],
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1024,
) -> dict:
    """Call OpenRouter with tool definitions.  Returns the raw message dict."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY()}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/roaming-rockenfels/LegacyLens",
        "X-Title": "LegacyLens",
    }
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
        "tools": tools,
    }
    resp = client.post(f"{OPENROUTER_BASE}/chat/completions", headers=headers, json=payload)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]


def _chat_completion_stream(
    client: httpx.Client,
    messages: list[dict],
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4096,
) -> Iterator[str]:
    """Stream chat completion from OpenRouter, yielding content deltas."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY()}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/roaming-rockenfels/LegacyLens",
        "X-Title": "LegacyLens",
    }
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
        "stream": True,
    }
    with client.stream("POST", f"{OPENROUTER_BASE}/chat/completions", headers=headers, json=payload) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            data_str = line[len("data: "):]
            if data_str.strip() == "[DONE]":
                break
            try:
                data = json.loads(data_str)
                delta = data["choices"][0].get("delta", {})
                content = delta.get("content")
                if content:
                    yield content
            except (json.JSONDecodeError, KeyError, IndexError):
                continue


SYSTEM_PROMPT = """You are LegacyLens, an expert assistant for understanding legacy Fortran codebases (specifically LAPACK — Linear Algebra PACKage).

You help developers understand, navigate, and work with legacy Fortran code.

Guidelines:
- When code context is provided, base your answer on it. Reference specific subroutine/function names, quote key snippets in ```fortran blocks, and explain call chains.
- Source code in the context has line numbers prefixed as "  LINE | CODE". When you quote code, you MUST keep the line-number prefixes inside the ```fortran block exactly as they appear in the context. For example:
  **`dgesv.f:140-145`**
  ```fortran
   140 |       CALL DGETRF( N, N, A, LDA, IPIV, INFO )
   141 |       IF( INFO.EQ.0 ) THEN
  ```
  Never strip the "N | " prefix from quoted lines.
- When no code context is available, answer using your knowledge of LAPACK, Fortran, and numerical linear algebra. Cite authoritative sources as markdown links (e.g. [LAPACK Users' Guide](https://netlib.org/lapack/lug/), [netlib.org](https://netlib.org/lapack/explore-html/), Intel MKL docs).
- Explain Fortran concepts in modern programming terms when helpful.
- Describe both what the code does and why.
- Use clear, concise language for developers who may not know Fortran.
"""


def build_context(results: list[dict]) -> str:
    """Build context string from retrieval results."""
    parts = []
    for i, match in enumerate(results, 1):
        meta = match.get("metadata", {})
        score = match.get("score", 0)
        unit_name = meta.get("unit_name", "unknown")
        unit_type = meta.get("unit_type", "unknown")
        file_path = meta.get("file_path", "unknown")
        purpose = meta.get("purpose", "")
        params = meta.get("parameters", [])
        calls = meta.get("calls", [])
        language = meta.get("language", "fortran")
        start_line = meta.get("start_line", "?")
        end_line = meta.get("end_line", "?")

        chunk_text = f"""--- Chunk {i} (relevance: {score:.3f}) ---
{unit_type.upper()} {unit_name}
File: {file_path} (lines {start_line}-{end_line})
Language: {language}
Purpose: {purpose}
Parameters: {', '.join(params) if params else 'none'}
Calls: {', '.join(calls) if calls else 'none'}
"""
        # Append actual source code with line numbers when available
        if isinstance(start_line, int) and isinstance(end_line, int):
            snippet = read_source_snippet(file_path, start_line, end_line)
            if snippet:
                numbered_lines = []
                for line_no, line in enumerate(snippet.splitlines(), start=start_line):
                    numbered_lines.append(f"{line_no:>6} | {line}")
                numbered_snippet = "\n".join(numbered_lines)
                chunk_text += f"\nSource:\n```fortran\n{numbered_snippet}\n```\n"

        parts.append(chunk_text)

    return "\n".join(parts)


def generate_answer(
    question: str,
    results: list[dict],
    model: str = DEFAULT_MODEL,
    mode: str = "explain",
) -> str:
    """Generate an answer using Claude (via OpenRouter) based on retrieved code chunks.

    Args:
        question: The user's natural language question.
        results: Retrieved chunks from Pinecone.
        model: Model to use (OpenRouter model ID).
        mode: Response mode — "explain", "deps", "docs", or "business_logic".

    Returns:
        Generated answer text.
    """
    context = build_context(results)

    mode_instructions = {
        "explain": "Provide a clear explanation of what this code does, how it works, and why.",
        "deps": "Focus on the dependency relationships: what calls what, what external libraries are used, and the data flow between routines.",
        "docs": "Generate concise modern documentation for the code, including function signature, parameter table, computation description, 1-2 usage examples, and related routines. Keep it focused — no performance tuning advice or implementation notes.",
        "business_logic": "Extract and explain the core business/mathematical logic, the algorithm being implemented, and its practical applications.",
    }

    instruction = mode_instructions.get(mode, mode_instructions["explain"])

    if context:
        user_message = f"""Here are relevant code chunks from the LAPACK Fortran codebase:

{context}

Question: {question}

{instruction}"""
    else:
        user_message = f"""No code chunks were found in the codebase for this query. Answer using your expertise in LAPACK and Fortran, and cite authoritative web sources.

Question: {question}

{instruction}"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    # Docs mode generates longer output (parameter tables, examples, etc.)
    max_tokens = 8192 if mode == "docs" else 4096

    return _chat_completion(messages, model=model, max_tokens=max_tokens)


def generate_answer_stream(
    question: str,
    results: list[dict],
    model: str = DEFAULT_MODEL,
    mode: str = "explain",
) -> Iterator[str]:
    """Stream an answer, yielding content deltas.

    Same as generate_answer but returns an iterator of token strings
    suitable for measuring time-to-first-token.
    """
    context = build_context(results)

    mode_instructions = {
        "explain": "Provide a clear explanation of what this code does, how it works, and why.",
        "deps": "Focus on the dependency relationships: what calls what, what external libraries are used, and the data flow between routines.",
        "docs": "Generate concise modern documentation for the code, including function signature, parameter table, computation description, 1-2 usage examples, and related routines. Keep it focused — no performance tuning advice or implementation notes.",
        "business_logic": "Extract and explain the core business/mathematical logic, the algorithm being implemented, and its practical applications.",
    }

    instruction = mode_instructions.get(mode, mode_instructions["explain"])

    if context:
        user_message = f"""Here are relevant code chunks from the LAPACK Fortran codebase:

{context}

Question: {question}

{instruction}"""
    else:
        user_message = f"""No code chunks were found in the codebase for this query. Answer using your expertise in LAPACK and Fortran, and cite authoritative web sources.

Question: {question}

{instruction}"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    max_tokens = 8192 if mode == "docs" else 4096

    with httpx.Client(timeout=60) as client:
        yield from _chat_completion_stream(client, messages, model=model, max_tokens=max_tokens)
