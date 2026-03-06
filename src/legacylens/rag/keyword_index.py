"""BM25 keyword index for hybrid search backstop.

Provides a lightweight keyword search alongside vector search to catch
thin-wrapper code that has weak embedding signal but strong keyword matches.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

from legacylens.chunkers.base import Chunk

_INDEX_DIR = Path.home() / ".legacylens" / "indexes"

# Split camelCase and snake_case identifiers into tokens
_RE_CAMEL_SPLIT = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_RE_WORD_SPLIT = re.compile(r"[_\s.,:;()\[\]{}<>=/\\|@#$%^&*!?\"'`~+\-]+")

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "as", "be", "was", "are",
    "this", "that", "self", "cls", "none", "true", "false", "def", "class",
    "return", "import", "if", "else", "elif", "try", "except", "pass",
    "how", "does", "what", "when", "where", "which", "who", "why",
    "do", "did", "has", "have", "had", "not", "can", "will", "would",
    "should", "could", "about", "into", "each", "all", "its",
}


def _tokenize(text: str) -> list[str]:
    """Tokenize text by splitting identifiers and lowercasing."""
    # First split on punctuation/whitespace
    parts = _RE_WORD_SPLIT.split(text)
    tokens: list[str] = []
    for part in parts:
        if not part:
            continue
        # Split camelCase
        sub_parts = _RE_CAMEL_SPLIT.split(part)
        for sp in sub_parts:
            tok = sp.lower().strip()
            if tok and tok not in _STOPWORDS and len(tok) >= 2:
                tokens.append(tok)
    return tokens


def _chunk_to_document(chunk: Chunk) -> str:
    """Build a searchable document string from chunk metadata."""
    meta = chunk.metadata
    parts = [meta.unit_name]
    if meta.purpose:
        parts.append(meta.purpose)
    if meta.calls:
        parts.extend(meta.calls[:15])
    if meta.parameters:
        parts.extend(meta.parameters[:10])
    if meta.base_classes:
        parts.extend(meta.base_classes)
    if meta.decorators:
        parts.extend(meta.decorators)
    if meta.dunder_methods:
        parts.extend(meta.dunder_methods)
    return " ".join(parts)


class KeywordIndex:
    """BM25-based keyword index for code chunks."""

    def __init__(self) -> None:
        self._bm25 = None
        self._chunk_ids: list[str] = []

    def build(self, chunks: list[Chunk]) -> None:
        """Build the BM25 index from a list of chunks."""
        from rank_bm25 import BM25Plus

        self._chunk_ids = [c.chunk_id for c in chunks]
        corpus = [_tokenize(_chunk_to_document(c)) for c in chunks]
        self._bm25 = BM25Plus(corpus)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Search the index and return (chunk_id, score) pairs."""
        if self._bm25 is None or not self._chunk_ids:
            return []

        tokens = _tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        # Get top-k indices by score
        indexed_scores = [(i, float(s)) for i, s in enumerate(scores) if s > 0]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        results: list[tuple[str, float]] = []
        for i, score in indexed_scores[:top_k]:
            results.append((self._chunk_ids[i], score))
        return results

    def save(self, source_name: str) -> None:
        """Persist the index to disk."""
        _INDEX_DIR.mkdir(parents=True, exist_ok=True)
        path = _INDEX_DIR / f"{source_name}_bm25.pkl"
        with open(path, "wb") as f:
            pickle.dump({"bm25": self._bm25, "chunk_ids": self._chunk_ids}, f)

    def load(self, source_name: str) -> bool:
        """Load a previously saved index. Returns True if successful."""
        path = _INDEX_DIR / f"{source_name}_bm25.pkl"
        if not path.exists():
            return False
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self._bm25 = data["bm25"]
            self._chunk_ids = data["chunk_ids"]
            return True
        except Exception:
            return False
