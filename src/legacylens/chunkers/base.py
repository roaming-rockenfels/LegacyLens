"""Base chunker interface and registry for pluggable language support."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ChunkMetadata:
    """Metadata extracted from a code chunk."""

    file_path: str
    start_line: int
    end_line: int
    unit_name: str
    unit_type: str  # "subroutine" | "function" | "program" | "paragraph" | ...
    language: str  # "fortran" | "cobol"
    parameters: list[str] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)
    external_deps: list[str] = field(default_factory=list)
    uses: list[str] = field(default_factory=list)
    purpose: str = ""
    precision: str | None = None  # fortran: "single" | "double" | "complex" | "double_complex"
    category: str | None = None
    routine_role: str | None = None  # "driver" | "computational" | "auxiliary" | "blas"
    module_tier: str = "current"  # "current" | "legacy" | "deprecated" | "internal"
    base_classes: list[str] = field(default_factory=list)
    decorators: list[str] = field(default_factory=list)
    return_type: str = ""
    visibility: str = "public"  # "public" | "private" | "protected" | "dunder"
    dunder_methods: list[str] = field(default_factory=list)
    chunk_index: int | None = None
    chunk_total: int | None = None

    def to_pinecone_metadata(self) -> dict:
        """Flatten metadata for Pinecone storage (must be str/int/float/bool/list[str])."""
        meta = {
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "unit_name": self.unit_name,
            "unit_type": self.unit_type,
            "language": self.language,
            "purpose": self.purpose,
        }
        if self.parameters:
            meta["parameters"] = self.parameters
        if self.calls:
            meta["calls"] = self.calls
        if self.external_deps:
            meta["external_deps"] = self.external_deps
        if self.uses:
            meta["uses"] = self.uses
        if self.precision:
            meta["precision"] = self.precision
        if self.category:
            meta["category"] = self.category
        if self.routine_role:
            meta["routine_role"] = self.routine_role
        meta["module_tier"] = self.module_tier
        if self.base_classes:
            meta["base_classes"] = self.base_classes
        if self.decorators:
            meta["decorators"] = self.decorators
        if self.return_type:
            meta["return_type"] = self.return_type
        if self.visibility != "public":
            meta["visibility"] = self.visibility
        if self.dunder_methods:
            meta["dunder_methods"] = self.dunder_methods
        if self.chunk_index is not None:
            meta["chunk_index"] = self.chunk_index
            meta["chunk_total"] = self.chunk_total
        return meta


def classify_visibility(name: str) -> str:
    """Classify a Python identifier's visibility from its name."""
    if name.startswith("__") and name.endswith("__"):
        return "dunder"
    if name.startswith("__"):
        return "private"
    if name.startswith("_"):
        return "protected"
    return "public"


def classify_module_tier(file_path: str) -> str:
    """Classify a file's module tier from its path."""
    fp = file_path.replace("\\", "/")
    if fp.startswith("v1/") or "/v1/" in fp:
        return "legacy"
    if fp.startswith("deprecated/") or "/deprecated/" in fp:
        return "deprecated"
    if fp.startswith("_internal/") or "/_internal/" in fp:
        return "internal"
    return "current"


@dataclass
class Chunk:
    """A code chunk ready for embedding."""

    content: str  # raw source code
    enriched_content: str  # with NL summary prefix (this gets embedded)
    metadata: ChunkMetadata

    @property
    def chunk_id(self) -> str:
        """Unique ID for this chunk (used as Pinecone vector ID)."""
        # Include file path to disambiguate variants and same-named routines
        path = self.metadata.file_path.replace("/", ":").replace("\\", ":").lower()
        # Strip extension for cleaner IDs
        if "." in path:
            path = path.rsplit(".", 1)[0]
        name = self.metadata.unit_name.lower()
        if self.metadata.chunk_index is not None:
            return f"{self.metadata.language}:{path}:{name}:{self.metadata.chunk_index}"
        return f"{self.metadata.language}:{path}:{name}"


class BaseChunker(ABC):
    """Abstract base class for language-specific code chunkers."""

    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """Return file extensions this chunker handles, e.g. ['.f', '.f90']."""
        ...

    @abstractmethod
    def chunk_file(self, file_path: str, content: str) -> list[Chunk]:
        """Parse a single file and return a list of chunks."""
        ...

    def chunk_directory(self, directory: str) -> list[Chunk]:
        """Walk a directory, filter by supported extensions, chunk each file."""
        chunks: list[Chunk] = []
        extensions = set(self.supported_extensions())
        root = Path(directory)

        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in extensions:
                try:
                    content = path.read_text(encoding="utf-8", errors="replace")
                    rel_path = str(path.relative_to(root))
                    file_chunks = self.chunk_file(rel_path, content)
                    chunks.extend(file_chunks)
                except Exception as e:
                    print(f"Warning: failed to chunk {path}: {e}")

        return chunks


class ChunkerRegistry:
    """Registry that auto-selects the right chunker based on file extension."""

    def __init__(self) -> None:
        self._chunkers: dict[str, BaseChunker] = {}

    def register(self, chunker: BaseChunker) -> None:
        for ext in chunker.supported_extensions():
            self._chunkers[ext.lower()] = chunker

    def get_chunker(self, file_path: str) -> BaseChunker | None:
        ext = Path(file_path).suffix.lower()
        if not ext and file_path.startswith("."):
            ext = file_path.lower()  # handle bare extensions like ".f"
        return self._chunkers.get(ext)

    def supported_extensions(self) -> list[str]:
        return list(self._chunkers.keys())

    def chunk_directory(self, directory: str) -> list[Chunk]:
        """Walk a directory and dispatch each file to the correct chunker by extension."""
        chunks: list[Chunk] = []
        root = Path(directory)
        extensions = set(self.supported_extensions())

        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in extensions:
                chunker = self._chunkers.get(path.suffix.lower())
                if chunker is None:
                    continue
                try:
                    content = path.read_text(encoding="utf-8", errors="replace")
                    rel_path = str(path.relative_to(root))
                    file_chunks = chunker.chunk_file(rel_path, content)
                    chunks.extend(file_chunks)
                except Exception as e:
                    print(f"Warning: failed to chunk {path}: {e}")

        return chunks
