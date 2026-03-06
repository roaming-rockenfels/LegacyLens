"""Source management — track ingested codebases with language metadata and namespace isolation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class SourceInfo:
    """Metadata for an ingested codebase."""

    name: str
    path: str
    namespace: str
    languages: list[str]
    extensions: dict[str, int]
    chunk_count: int
    ingested_at: str


_SOURCES_DIR = Path.home() / ".legacylens"
_SOURCES_FILE = _SOURCES_DIR / "sources.json"


def _slugify(name: str) -> str:
    """Convert a name to a Pinecone-safe namespace slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "default"


def _load_sources() -> dict[str, dict]:
    if not _SOURCES_FILE.exists():
        return {}
    return json.loads(_SOURCES_FILE.read_text())


def _save_sources(data: dict[str, dict]) -> None:
    _SOURCES_DIR.mkdir(parents=True, exist_ok=True)
    _SOURCES_FILE.write_text(json.dumps(data, indent=2))


# Extension-to-language mapping — kept in sync with registered chunkers
_EXTENSION_LANGUAGES: dict[str, str] = {
    ".f": "fortran", ".f90": "fortran", ".f95": "fortran",
    ".f03": "fortran", ".for": "fortran",
    ".py": "python", ".pyi": "python",
}


def scan_directory(path: str, known_extensions: list[str] | None = None) -> dict[str, int]:
    """Walk a directory and count files by extension.

    Args:
        path: Directory to scan.
        known_extensions: If provided, only count these extensions.

    Returns:
        Dict of extension -> file count.
    """
    root = Path(path)
    counts: dict[str, int] = {}
    for p in root.rglob("*"):
        if p.is_file():
            ext = p.suffix.lower()
            if ext and (known_extensions is None or ext in known_extensions):
                counts[ext] = counts.get(ext, 0) + 1
    return counts


def detect_languages(extensions: dict[str, int]) -> list[str]:
    """Determine languages from file extension counts."""
    langs: set[str] = set()
    for ext in extensions:
        lang = _EXTENSION_LANGUAGES.get(ext)
        if lang:
            langs.add(lang)
    return sorted(langs)


def register_source(
    name: str,
    path: str,
    languages: list[str],
    extensions: dict[str, int],
    chunk_count: int,
) -> SourceInfo:
    """Register a new ingested source."""
    sources = _load_sources()
    info = SourceInfo(
        name=name,
        path=str(Path(path).resolve()),
        namespace=_slugify(name),
        languages=languages,
        extensions=extensions,
        chunk_count=chunk_count,
        ingested_at=datetime.now(timezone.utc).isoformat(),
    )
    sources[name] = asdict(info)
    _save_sources(sources)
    return info


def list_sources() -> list[SourceInfo]:
    """List all registered sources."""
    sources = _load_sources()
    return [SourceInfo(**v) for v in sources.values()]


def get_source(name: str) -> SourceInfo | None:
    """Look up a source by name."""
    sources = _load_sources()
    data = sources.get(name)
    if data:
        return SourceInfo(**data)
    return None


def remove_source(name: str) -> bool:
    """Remove a source from the registry. Returns True if found."""
    sources = _load_sources()
    if name in sources:
        del sources[name]
        _save_sources(sources)
        return True
    return False


def get_source_languages(name: str) -> list[str]:
    """Get the detected languages for a source."""
    info = get_source(name)
    return info.languages if info else []
