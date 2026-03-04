"""Embedding generation using Voyage Code 3."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import voyageai

from legacylens.config import VOYAGE_API_KEY

MODEL = "voyage-code-3"
DIMENSION = 1024

_client: voyageai.Client | None = None
_embedding_cache: dict[str, list[float]] | None = None
_embedding_cache_path: Path | None = None


def _get_client() -> voyageai.Client:
    global _client
    if _client is None:
        _client = voyageai.Client(api_key=VOYAGE_API_KEY())
    return _client


def _cache_key(text: str, input_type: str) -> str:
    """Deterministic cache key from model + input_type + text."""
    raw = f"{MODEL}:{input_type}:{text}"
    return hashlib.sha256(raw.encode()).hexdigest()


def enable_embedding_cache(path: str | Path) -> int:
    """Enable a disk-backed embedding cache.

    Loads existing cached embeddings from *path*. New embeddings are
    appended automatically and flushed to disk via save_embedding_cache().

    The cache is keyed by (model, input_type, text) so it self-invalidates
    when the embedding model changes.

    Args:
        path: JSON file to load/save cached embeddings.

    Returns:
        Number of cached embeddings loaded.
    """
    global _embedding_cache, _embedding_cache_path
    _embedding_cache_path = Path(path)
    if _embedding_cache_path.exists():
        with open(_embedding_cache_path) as f:
            _embedding_cache = json.load(f)
    else:
        _embedding_cache = {}
    return len(_embedding_cache)


def save_embedding_cache() -> None:
    """Flush the in-memory embedding cache to disk."""
    if _embedding_cache is not None and _embedding_cache_path is not None:
        _embedding_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(_embedding_cache_path, "w") as f:
            json.dump(_embedding_cache, f)


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
    # Check cache for all texts
    if _embedding_cache is not None:
        results: list[list[float] | None] = []
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []
        for i, text in enumerate(texts):
            key = _cache_key(text, input_type)
            cached = _embedding_cache.get(key)
            if cached is not None:
                results.append(cached)
            else:
                results.append(None)
                uncached_indices.append(i)
                uncached_texts.append(text)

        if not uncached_texts:
            return results  # type: ignore[return-value]

        # Embed only uncached texts
        fresh = _embed_texts_api(uncached_texts, input_type, max_retries)
        for idx, embedding in zip(uncached_indices, fresh):
            key = _cache_key(texts[idx], input_type)
            _embedding_cache[key] = embedding
            results[idx] = embedding

        return results  # type: ignore[return-value]

    return _embed_texts_api(texts, input_type, max_retries)


def _embed_texts_api(
    texts: list[str],
    input_type: str,
    max_retries: int,
) -> list[list[float]]:
    """Call the Voyage API directly (no cache)."""
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
