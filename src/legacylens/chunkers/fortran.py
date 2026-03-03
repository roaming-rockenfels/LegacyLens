"""Fortran-specific chunker for LAPACK and similar codebases.

Uses regex-based parsing. LAPACK follows one-subroutine-per-file convention,
so each file typically produces one chunk. Long files (>16K tokens) are split
at comment-delimited section boundaries with preamble inclusion.
"""

from __future__ import annotations

import re

from legacylens.chunkers.base import BaseChunker, Chunk, ChunkMetadata

# Approximate tokens per character for Fortran (conservative estimate)
_CHARS_PER_TOKEN = 4
_MAX_CHUNK_TOKENS = 15_000  # leave headroom below Voyage's 16K limit
_MAX_CHUNK_CHARS = _MAX_CHUNK_TOKENS * _CHARS_PER_TOKEN
_PREAMBLE_MAX_CHARS = 4000 * _CHARS_PER_TOKEN  # ~4K tokens for preamble

# Precision mapping from first letter of LAPACK routine name
_PRECISION_MAP = {
    "s": "single",
    "d": "double",
    "c": "complex",
    "z": "double_complex",
}

# LAPACK driver routines (base names without precision prefix)
# These are the user-facing entry points documented in LAPACK Users' Guide
_DRIVER_BASE_NAMES = {
    # Linear solve drivers
    "gesv", "gesvx", "gesvxx", "gbsv", "gbsvx", "gtsv", "gtsvx",
    "posv", "posvx", "posvxx", "ppsv", "ppsvx", "pbsv", "pbsvx",
    "ptsv", "ptsvx", "sysv", "sysvx", "sysv_rk", "sysv_rook", "sysv_aa",
    "hesv", "hesvx", "hesv_rk", "hesv_rook", "hesv_aa",
    "spsv", "spsvx", "hpsv", "hpsvx",
    # Least squares drivers
    "gels", "gelsy", "gelss", "gelsd",
    # Eigenvalue drivers (symmetric/hermitian)
    "syev", "syevd", "syevx", "syevr",
    "heev", "heevd", "heevx", "heevr",
    "spev", "spevd", "spevx", "hpev", "hpevd", "hpevx",
    "sbev", "sbevd", "sbevx", "hbev", "hbevd", "hbevx",
    "stev", "stevd", "stevx", "stevr",
    # Eigenvalue drivers (nonsymmetric)
    "geev", "geevx", "gees", "geesx",
    # SVD drivers
    "gesvd", "gesdd", "gesvdx",
    # Generalized eigenvalue drivers
    "sygv", "sygvd", "sygvx", "hegv", "hegvd", "hegvx",
    "spgv", "spgvd", "spgvx", "hpgv", "hpgvd", "hpgvx",
    "sbgv", "sbgvd", "sbgvx", "hbgv", "hbgvd", "hbgvx",
    "ggev", "ggevx", "gges", "ggesx", "ggsvd",
}

# Category mapping from 2nd-3rd characters of LAPACK routine name
_CATEGORY_MAP = {
    "ge": "general",
    "sy": "symmetric",
    "he": "hermitian",
    "po": "positive_definite",
    "tr": "triangular",
    "or": "orthogonal",
    "un": "unitary",
    "gb": "general_band",
    "sb": "symmetric_band",
    "pb": "positive_definite_band",
    "la": "auxiliary",
    "st": "symmetric_tridiagonal",
    "gt": "general_tridiagonal",
    "pt": "positive_definite_tridiagonal",
    "gg": "generalized",
    "sp": "symmetric_packed",
    "hp": "hermitian_packed",
    "tp": "triangular_packed",
    "bd": "bidiagonal",
}

# Regex patterns for Fortran parsing
_RE_SUBROUTINE = re.compile(
    r"^\s{0,6}\s*(?:RECURSIVE\s+|PURE\s+|ELEMENTAL\s+)*SUBROUTINE\s+(\w+)\s*\(([^)]*)\)",
    re.IGNORECASE | re.MULTILINE,
)
_RE_FUNCTION = re.compile(
    r"^\s{0,6}\s*(?:RECURSIVE\s+|PURE\s+|ELEMENTAL\s+)*(?:\w+(?:\*\d+)?\s+)*FUNCTION\s+(\w+)\s*\(([^)]*)\)",
    re.IGNORECASE | re.MULTILINE,
)
_RE_PROGRAM = re.compile(
    r"^\s{0,6}\s*PROGRAM\s+(\w+)",
    re.IGNORECASE | re.MULTILINE,
)
_RE_CALL = re.compile(r"\bCALL\s+(\w+)", re.IGNORECASE)
_RE_EXTERNAL = re.compile(r"^\s+EXTERNAL\s+(.+)", re.IGNORECASE | re.MULTILINE)
_RE_USE = re.compile(r"^\s*USE\s+(\w+)", re.IGNORECASE | re.MULTILINE)
_RE_PURPOSE = re.compile(
    r"(?:Purpose|PURPOSE)\s*[:\n=]",
    re.IGNORECASE,
)
# Section delimiter in LAPACK comment blocks
_RE_SECTION_DELIM = re.compile(r"^\*\s*={10,}", re.MULTILINE)


def _extract_purpose(content: str) -> str:
    """Extract the Purpose description from LAPACK comment headers.

    Handles the LAPACK format with \\verbatim blocks:
        *> \\par Purpose:
        *  =============
        *> \\verbatim
        *> DGESV computes the solution to ...
        *> \\endverbatim
    """
    match = _RE_PURPOSE.search(content)
    if not match:
        return ""

    start = match.end()
    lines = content[start:].split("\n")
    purpose_lines: list[str] = []
    in_verbatim = False

    for line in lines:
        stripped = line.strip()

        # Handle verbatim blocks
        if "\\verbatim" in stripped and "\\endverbatim" not in stripped:
            in_verbatim = True
            continue
        if "\\endverbatim" in stripped:
            break

        # Extract text from comment lines
        if stripped.startswith("*>"):
            text = stripped[2:].strip()
            # Skip doxygen commands
            if text.startswith("\\") and not in_verbatim:
                if text.startswith("\\param"):
                    break
                continue
            if text:
                purpose_lines.append(text)
        elif stripped.startswith("*"):
            text = stripped[1:].strip()
            if text.startswith("=") and len(text) > 5:
                continue  # skip delimiter lines like ============
            if text.startswith("Arguments") or text.startswith("Author"):
                break
            if text:
                purpose_lines.append(text)
        elif not stripped:
            continue
        else:
            break

    purpose = " ".join(purpose_lines).strip()
    # Take first two sentences max
    sentences = re.split(r"(?<=[.!?])\s+", purpose)
    return " ".join(sentences[:2]).strip()


def _extract_unit_info(
    content: str,
) -> tuple[str, str, list[str]]:
    """Extract (unit_name, unit_type, parameters) from Fortran source."""
    for pattern, unit_type in [
        (_RE_SUBROUTINE, "subroutine"),
        (_RE_FUNCTION, "function"),
    ]:
        match = pattern.search(content)
        if match:
            name = match.group(1).upper()
            params_str = match.group(2) if match.lastindex and match.lastindex >= 2 else ""
            params = [p.strip().upper() for p in params_str.split(",") if p.strip()]
            return name, unit_type, params

    match = _RE_PROGRAM.search(content)
    if match:
        return match.group(1).upper(), "program", []

    return "UNKNOWN", "unknown", []


def _extract_calls(content: str) -> list[str]:
    """Extract all CALL targets from the source."""
    calls = _RE_CALL.findall(content)
    return sorted(set(c.upper() for c in calls))


def _extract_externals(content: str) -> list[str]:
    """Extract EXTERNAL declarations."""
    externals: list[str] = []
    for match in _RE_EXTERNAL.finditer(content):
        names = re.split(r"[,\s]+", match.group(1).strip())
        externals.extend(n.upper() for n in names if n)
    return sorted(set(externals))


def _extract_uses(content: str) -> list[str]:
    """Extract USE module statements."""
    return sorted(set(m.upper() for m in _RE_USE.findall(content)))


def _classify_routine(name: str) -> tuple[str | None, str | None]:
    """Classify a LAPACK routine by precision and category."""
    if len(name) < 2:
        return None, None

    precision = _PRECISION_MAP.get(name[0].lower())

    category = None
    if len(name) >= 3:
        prefix = name[1:3].lower()
        category = _CATEGORY_MAP.get(prefix)

    return precision, category


def _classify_role(name: str, file_path: str) -> str | None:
    """Classify a routine's role: driver, computational, auxiliary, or blas.

    Args:
        name: Uppercase routine name (e.g., "DGESV").
        file_path: Relative file path (used to detect BLAS directory).

    Returns:
        One of "driver", "computational", "auxiliary", "blas", or None.
    """
    if not name or name == "UNKNOWN":
        return None

    # BLAS routines live in BLAS/ directory
    path_lower = file_path.lower().replace("\\", "/")
    if "/blas/" in path_lower or path_lower.startswith("blas/"):
        return "blas"

    name_lower = name.lower()

    # Strip precision prefix to get base name
    if len(name_lower) >= 2 and name_lower[0] in "sdcz":
        base = name_lower[1:]
    else:
        base = name_lower

    # Check driver list
    if base in _DRIVER_BASE_NAMES:
        return "driver"

    # Auxiliary routines: xLA prefix (2nd-3rd chars = "la")
    if len(name_lower) >= 3 and name_lower[1:3] == "la":
        return "auxiliary"

    # Everything else is computational
    return "computational"


def _build_enriched_content(
    content: str,
    name: str,
    unit_type: str,
    purpose: str,
    calls: list[str],
    file_path: str,
    start_line: int,
    end_line: int,
    precision: str | None,
) -> str:
    """Build enriched content with NL summary prefix."""
    precision_label = precision.replace("_", " ") if precision else ""
    type_label = unit_type.capitalize()

    header_parts = [f"# {name}"]
    if precision_label:
        header_parts.append(f"- {precision_label.title()} precision Fortran {type_label}")
    else:
        header_parts.append(f"- Fortran {type_label}")

    if purpose:
        header_parts.append(f"# Purpose: {purpose}")

    meta_parts = [f"File: {file_path}", f"Lines: {start_line}-{end_line}"]
    if calls:
        meta_parts.append(f"Calls: {', '.join(calls[:10])}")
    header_parts.append(f"# {' | '.join(meta_parts)}")

    header = "\n".join(header_parts)
    return f"{header}\n\n{content}"


def _find_preamble_end(lines: list[str]) -> int:
    """Find where the preamble (comment header + declarations) ends and executable code begins.

    Returns the line index of the first executable statement.
    """
    in_declarations = False
    for i, line in enumerate(lines):
        stripped = line.strip().upper()
        # Skip empty lines and comments
        if not stripped or stripped.startswith("*") or stripped.startswith("!") or stripped.startswith("C "):
            continue
        # Declaration keywords
        if any(stripped.startswith(kw) for kw in [
            "SUBROUTINE", "FUNCTION", "PROGRAM",
            "RECURSIVE", "PURE", "ELEMENTAL",
            "INTEGER", "REAL", "DOUBLE", "COMPLEX", "LOGICAL", "CHARACTER",
            "IMPLICIT", "PARAMETER", "EXTERNAL", "INTRINSIC", "DIMENSION",
            "DATA", "SAVE", "COMMON", "EQUIVALENCE",
        ]):
            in_declarations = True
            continue
        # First non-declaration, non-comment line = start of executable code
        if in_declarations:
            return i

    return len(lines)


class FortranChunker(BaseChunker):
    """Chunker for Fortran source files (fixed-form and free-form)."""

    def supported_extensions(self) -> list[str]:
        return [".f", ".f90", ".f95", ".f03", ".for"]

    def chunk_file(self, file_path: str, content: str) -> list[Chunk]:
        if not content.strip():
            return []

        lines = content.split("\n")
        total_lines = len(lines)

        name, unit_type, parameters = _extract_unit_info(content)
        purpose = _extract_purpose(content)
        calls = _extract_calls(content)
        externals = _extract_externals(content)
        uses = _extract_uses(content)
        precision, category = _classify_routine(name)
        routine_role = _classify_role(name, file_path)

        # Check if we need to split
        estimated_chars = len(content)
        if estimated_chars <= _MAX_CHUNK_CHARS:
            # Single chunk — the common case
            enriched = _build_enriched_content(
                content, name, unit_type, purpose, calls,
                file_path, 1, total_lines, precision,
            )
            metadata = ChunkMetadata(
                file_path=file_path,
                start_line=1,
                end_line=total_lines,
                unit_name=name,
                unit_type=unit_type,
                language="fortran",
                parameters=parameters,
                calls=calls,
                external_deps=externals,
                uses=uses,
                purpose=purpose,
                precision=precision,
                category=category,
                routine_role=routine_role,
            )
            return [Chunk(content=content, enriched_content=enriched, metadata=metadata)]

        # Split long files at comment-delimited sections
        return self._split_long_file(
            file_path, content, lines, total_lines,
            name, unit_type, parameters, purpose,
            calls, externals, uses, precision, category, routine_role,
        )

    def _split_long_file(
        self,
        file_path: str,
        content: str,
        lines: list[str],
        total_lines: int,
        name: str,
        unit_type: str,
        parameters: list[str],
        purpose: str,
        calls: list[str],
        externals: list[str],
        uses: list[str],
        precision: str | None,
        category: str | None,
        routine_role: str | None = None,
    ) -> list[Chunk]:
        """Split a long file into sub-chunks with preamble inclusion."""
        preamble_end = _find_preamble_end(lines)
        preamble_lines = lines[:preamble_end]
        preamble_text = "\n".join(preamble_lines)

        # If preamble is too long, truncate it
        if len(preamble_text) > _PREAMBLE_MAX_CHARS:
            preamble_text = preamble_text[:_PREAMBLE_MAX_CHARS] + "\n*     ... (preamble truncated)"

        code_lines = lines[preamble_end:]
        available_chars = _MAX_CHUNK_CHARS - len(preamble_text) - 500  # headroom for enrichment

        # Find split points at major comment blocks
        split_points = [0]
        for i, line in enumerate(code_lines):
            stripped = line.strip()
            if stripped.startswith("*") and len(stripped) > 10 and "=" * 5 in stripped:
                split_points.append(i)

        # If no good split points, fall back to fixed-size splits
        if len(split_points) < 2:
            chars_per_chunk = available_chars
            current = 0
            split_points = [0]
            running_len = 0
            for i, line in enumerate(code_lines):
                running_len += len(line) + 1
                if running_len >= chars_per_chunk:
                    split_points.append(i)
                    running_len = 0
            split_points.append(len(code_lines))
        else:
            split_points.append(len(code_lines))

        # Build chunks
        chunks: list[Chunk] = []
        # Deduplicate and sort split points
        split_points = sorted(set(split_points))
        chunk_total = len(split_points) - 1

        for idx in range(chunk_total):
            start_code_idx = split_points[idx]
            end_code_idx = split_points[idx + 1]

            # Add overlap from previous chunk
            overlap_start = max(0, start_code_idx - 15) if idx > 0 else start_code_idx
            section_lines = code_lines[overlap_start:end_code_idx]
            section_text = preamble_text + "\n" + "\n".join(section_lines)

            abs_start = preamble_end + overlap_start + 1  # 1-indexed
            abs_end = preamble_end + end_code_idx

            enriched = _build_enriched_content(
                section_text, name, unit_type, purpose, calls,
                file_path, abs_start, abs_end, precision,
            )

            metadata = ChunkMetadata(
                file_path=file_path,
                start_line=abs_start,
                end_line=abs_end,
                unit_name=name,
                unit_type=unit_type,
                language="fortran",
                parameters=parameters,
                calls=calls,
                external_deps=externals,
                uses=uses,
                purpose=purpose,
                precision=precision,
                category=category,
                routine_role=routine_role,
                chunk_index=idx + 1,
                chunk_total=chunk_total,
            )
            chunks.append(Chunk(content="\n".join(section_lines), enriched_content=enriched, metadata=metadata))

        return chunks
