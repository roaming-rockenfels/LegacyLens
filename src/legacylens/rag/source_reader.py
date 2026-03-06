"""Shared utility for reading source code snippets from LAPACK files."""

from __future__ import annotations

from pathlib import Path

from legacylens.config import LAPACK_DATA_DIR, CODEBASE_DATA_DIR


def resolve_source_path(file_path: str, base_dir: Path | None = None) -> Path | None:
    """Resolve a file_path from chunk metadata to an actual file on disk.

    Tries ``base/file_path`` then ``base/SRC/file_path``.
    Returns None if the file cannot be found.
    """
    try:
        base = base_dir or CODEBASE_DATA_DIR()
    except Exception:
        return None

    full = base / file_path
    if full.is_file():
        return full

    full = base / "SRC" / file_path
    if full.is_file():
        return full

    return None


def read_source_snippet(
    file_path: str,
    start_line: int,
    end_line: int,
    base_dir: Path | None = None,
    max_chars: int = 8000,
) -> str:
    """Read a line range from a source file and return it as a string.

    Returns ``""`` on any failure — never raises.
    """
    try:
        resolved = resolve_source_path(file_path, base_dir=base_dir)
        if resolved is None:
            return ""

        lines = resolved.read_text(encoding="utf-8", errors="replace").splitlines()

        # Convert to 0-based indexing; clamp to file bounds
        s = max(0, start_line - 1)
        e = min(len(lines), end_line)
        snippet = "\n".join(lines[s:e])

        if len(snippet) > max_chars:
            snippet = snippet[:max_chars] + "\n[truncated]"

        return snippet
    except Exception:
        return ""
