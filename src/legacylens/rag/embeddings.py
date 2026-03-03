"""Embedding generation using Voyage Code 3."""

from __future__ import annotations

import time

import voyageai

from legacylens.config import VOYAGE_API_KEY

MODEL = "voyage-code-3"
DIMENSION = 1024

_client: voyageai.Client | None = None


def _get_client() -> voyageai.Client:
    global _client
    if _client is None:
        _client = voyageai.Client(api_key=VOYAGE_API_KEY())
    return _client


def embed_texts(
    texts: list[str],
    input_type: str = "document",
    max_retries: int = 10,
) -> list[list[float]]:
    """Embed a batch of texts using Voyage Code 3.

    Handles rate limiting with exponential backoff.

    Args:
        texts: List of strings to embed.
        input_type: "document" for ingestion, "query" for search queries.
        max_retries: Max retry attempts on rate limit errors.

    Returns:
        List of embedding vectors (1024 dimensions each).
    """
    client = _get_client()

    for attempt in range(max_retries):
        try:
            result = client.embed(texts, model=MODEL, input_type=input_type)
            return result.embeddings
        except Exception as e:
            if "rate" in str(e).lower() or "429" in str(e):
                wait = min(2 ** attempt * 5, 60)
                print(f"  Rate limited, waiting {wait}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise

    raise RuntimeError(f"Failed to embed after {max_retries} retries")


def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    return embed_texts([query], input_type="query")[0]
