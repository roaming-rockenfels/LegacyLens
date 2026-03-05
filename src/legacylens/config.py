"""Configuration loaded from environment variables."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _require(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise RuntimeError(f"Missing required environment variable: {key}")
    return val


VOYAGE_API_KEY = lambda: _require("VOYAGE_API_KEY")  # noqa: E731
PINECONE_API_KEY = lambda: _require("PINECONE_API_KEY")  # noqa: E731
OPENROUTER_API_KEY = lambda: _require("OPENROUTER_API_KEY")  # noqa: E731
PINECONE_INDEX_NAME = lambda: os.getenv("PINECONE_INDEX_NAME", "legacylens")  # noqa: E731
LAPACK_DATA_DIR = lambda: Path(os.getenv("LAPACK_DATA_DIR", "data/lapack"))  # noqa: E731
LEGACYLENS_MODEL = lambda: os.getenv("LEGACYLENS_MODEL", "anthropic/claude-haiku-4.5")  # noqa: E731
