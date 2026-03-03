"""Answer generation via OpenRouter — synthesizes retrieved chunks into NL answers."""

from __future__ import annotations

import os

import httpx

from legacylens.config import OPENROUTER_API_KEY

# Default model — Haiku for speed/cost, upgrade to Sonnet for quality.
DEFAULT_MODEL = "anthropic/claude-haiku-4.5"

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


SYSTEM_PROMPT = """You are LegacyLens, an expert assistant for understanding legacy Fortran codebases (specifically LAPACK — Linear Algebra PACKage).

You help developers understand, navigate, and work with legacy Fortran code by answering questions based on retrieved code chunks from the codebase.

Guidelines:
- Answer based on the provided code context. If the context doesn't contain enough information, say so.
- Explain Fortran concepts in modern programming terms when helpful.
- Reference specific subroutine/function names and their relationships.
- When describing algorithms, explain both what the code does and why.
- Use clear, concise language appropriate for developers who may not know Fortran.
- When relevant, mention the call chain (which routines call which).
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

    user_message = f"""Here are relevant code chunks from the LAPACK Fortran codebase:

{context}

Question: {question}

{instruction}"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    # Docs mode generates longer output (parameter tables, examples, etc.)
    max_tokens = 8192 if mode == "docs" else 4096

    return _chat_completion(messages, model=model, max_tokens=max_tokens)
