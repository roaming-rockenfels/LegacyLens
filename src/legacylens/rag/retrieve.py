"""Retrieval pipeline: query → embed → search → format results."""

from __future__ import annotations

import re

from rich.console import Console
from rich.table import Table

from legacylens.rag.embeddings import embed_query, _get_client as _get_voyage_client
from legacylens.rag.storage import query_vectors

console = Console()

# Minimum score threshold for non-entity results
_MIN_SCORE_THRESHOLD = 0.45

# Role priority for sorting entity results (lower = more important)
_ROLE_PRIORITY = {"driver": 0, "computational": 1, "auxiliary": 2, "blas": 3}


def _sort_by_role(results: list[dict]) -> list[dict]:
    """Sort results by routine_role priority: drivers first, then computational, auxiliary, blas."""
    return sorted(
        results,
        key=lambda r: _ROLE_PRIORITY.get(
            r.get("metadata", {}).get("routine_role", ""), 99
        ),
    )

# Matches LAPACK precision prefix: single letter S/D/C/Z at start of routine name
_RE_PRECISION_PREFIX = re.compile(r"^[SDCZ](?=[A-Z])", re.IGNORECASE)

# Matches all-caps Fortran-style identifiers (2-31 chars, starts with letter)
_RE_FORTRAN_IDENT = re.compile(r"\b([A-Z][A-Z0-9_]{1,30})\b")

# LAPACK utility routines that don't follow the S/D/C/Z naming convention
_KNOWN_UTILITIES = {
    "XERBLA", "ILAENV", "LSAME", "LSAMEN",
    "IEEECK", "IPARMQ", "ILAVER",
}

# English words and domain acronyms that should not be treated as Fortran identifiers
_ENGLISH_STOPWORDS = {
    "THE", "AND", "FOR", "NOT", "ARE", "BUT", "HOW", "WHAT", "DOES", "THIS",
    "WITH", "FROM", "THAT", "WHICH", "WHERE", "WHEN", "INTO", "EACH", "BOTH",
    "ALL", "ITS", "HAS", "USE", "FIND", "SHOW", "DO", "IS", "IT", "OF", "TO",
    "LAPACK", "BLAS", "FORTRAN", "SVD", "LU", "QR",
}


def _extract_fortran_entities(question: str) -> dict[str, list[str]]:
    """Detect Fortran identifiers in a query and classify as parameters or routines.

    Returns:
        Dict with "parameters" and "routines" lists.
    """
    tokens = set(_RE_FORTRAN_IDENT.findall(question))
    tokens -= _ENGLISH_STOPWORDS

    parameters: list[str] = []
    routines: list[str] = []
    for t in tokens:
        if t in _KNOWN_UTILITIES:
            routines.append(t)
        elif len(t) >= 4 and t[0] in "SDCZ" and t[1].isalpha():
            routines.append(t)
        elif len(t) <= 8:
            parameters.append(t)

    return {"parameters": parameters, "routines": routines}


def _base_routine_name(unit_name: str) -> str:
    """Strip the LAPACK precision prefix to get the base routine name.

    Examples:
        DGESV  → GESV
        SLAQZ0 → LAQZ0
        CLAQZ0 → LAQZ0
        XERBLA → XERBLA  (no precision prefix)
    """
    if not unit_name:
        return unit_name
    name = unit_name.upper()
    if len(name) >= 2 and name[0] in "SDCZ" and name[1].isalpha():
        return name[1:]
    return name


def _diversify_results(results: list[dict], top_k: int) -> list[dict]:
    """Deduplicate results so only one variant of each base routine is kept.

    This prevents precision variants (SLAQZ0/DLAQZ0/CLAQZ0) from consuming
    multiple top_k slots. When variants collide, prefers the double-precision
    (D-prefix) form since that's the canonical reference in LAPACK docs.
    """
    # First pass: collect best representative per base name
    # For each base, keep the D-prefix variant if available, else the first seen
    base_to_result: dict[str, dict] = {}
    base_order: list[str] = []

    for r in results:
        unit_name = r.get("metadata", {}).get("unit_name", "")
        base = _base_routine_name(unit_name)

        if base not in base_to_result:
            base_to_result[base] = r
            base_order.append(base)
        elif unit_name and unit_name[0] == "D" and base_to_result[base].get("metadata", {}).get("unit_name", "")[0] != "D":
            # Replace non-D variant with D variant
            base_to_result[base] = r

    # Second pass: emit in original order, capped at top_k
    diversified: list[dict] = []
    for base in base_order:
        diversified.append(base_to_result[base])
        if len(diversified) >= top_k:
            break

    return diversified


def _rerank_results(question: str, results: list[dict], top_k: int) -> list[dict]:
    """Rerank results using Voyage rerank-2 for better relevance ordering.

    Falls back to truncation if reranking fails.
    """
    if not results:
        return results
    try:
        client = _get_voyage_client()
        documents = []
        for r in results:
            meta = r.get("metadata", {})
            doc = f"{meta.get('unit_type', '')} {meta.get('unit_name', '')}: {meta.get('purpose', '')}"
            if meta.get("parameters"):
                doc += f" Parameters: {', '.join(meta['parameters'][:10])}"
            if meta.get("calls"):
                doc += f" Calls: {', '.join(meta['calls'][:10])}"
            documents.append(doc)
        reranked = client.rerank(question, documents, model="rerank-2", top_k=top_k)
        return [results[rr.index] for rr in reranked.results]
    except Exception:
        return results[:top_k]


def retrieve(question: str, top_k: int = 5, pin_unit: str | None = None) -> list[dict]:
    """Retrieve relevant code chunks for a natural language question.

    Uses two-pass retrieval:
    1. Driver-filtered query to boost user-facing routines
    2. Unfiltered semantic search for breadth

    Results are diversified to avoid precision-variant clusters.

    Args:
        question: Natural language query about the codebase.
        top_k: Number of results to return.
        pin_unit: If set, ensure this unit_name appears in results
                  by doing a filtered lookup first, then filling
                  remaining slots with semantic search.

    Returns:
        List of match dicts with id, score, and metadata.
    """
    embedding = embed_query(question)

    if pin_unit:
        # First: get the exact routine via metadata filter
        pinned = query_vectors(
            embedding,
            top_k=3,
            filter={"unit_name": pin_unit.upper()},
        )

        # Then: get semantic results for surrounding context
        semantic = query_vectors(embedding, top_k=top_k * 3)

        # Merge: pinned first, then semantic (deduped + diversified)
        seen_ids = {r["id"] for r in pinned}
        merged = list(pinned)
        for r in semantic:
            if r["id"] not in seen_ids and len(merged) < top_k:
                merged.append(r)
                seen_ids.add(r["id"])

        return merged

    # Entity-filtered pass: detect Fortran identifiers in the query
    # Uses dual queries (D-prefix + all-precision) to guarantee canonical results
    entities = _extract_fortran_entities(question)
    entity_results: list[dict] = []

    for param in entities.get("parameters", []):
        # D-prefix query: guaranteed canonical (double-precision) results
        entity_results.extend(
            query_vectors(embedding, top_k=top_k, filter={"parameters": param, "precision": "double"})
        )
        # All-precision query: broader coverage
        entity_results.extend(
            query_vectors(embedding, top_k=top_k, filter={"parameters": param})
        )
    for routine in entities.get("routines", []):
        # Get the routine itself (may have multiple chunks if split)
        entity_results.extend(
            query_vectors(embedding, top_k=3, filter={"unit_name": routine.upper()})
        )
        # D-prefix callers
        entity_results.extend(
            query_vectors(embedding, top_k=top_k, filter={"calls": routine.upper(), "precision": "double"})
        )
        # All-precision callers
        entity_results.extend(
            query_vectors(embedding, top_k=top_k, filter={"calls": routine.upper()})
        )

    # Tag, deduplicate, sort by role, rerank, then diversify entity results
    # Order matters: rerank first (relevance ordering), then diversify last
    # (D-prefix preference). This prevents the reranker from undoing
    # diversification's precision-variant deduplication.
    for r in entity_results:
        r["_entity_match"] = True
    seen_ids: set[str] = set()
    deduped_entity: list[dict] = []
    for r in entity_results:
        if r["id"] not in seen_ids:
            deduped_entity.append(r)
            seen_ids.add(r["id"])
    sorted_entity = _sort_by_role(deduped_entity)
    reranked_entity = _rerank_results(question, sorted_entity, top_k * 2)
    diversified_entity = _diversify_results(reranked_entity, top_k)

    # Two-pass retrieval: drivers first, then general
    over_fetch = top_k * 3

    # Pass 1: driver-filtered query
    driver_results = query_vectors(
        embedding,
        top_k=top_k,
        filter={"routine_role": "driver"},
    )

    # Pass 2: unfiltered semantic search (over-fetch for diversification)
    general_results = query_vectors(embedding, top_k=over_fetch)

    # Merge non-entity results (deduped against entity IDs)
    non_entity: list[dict] = []
    for r in driver_results + general_results:
        if r["id"] not in seen_ids:
            non_entity.append(r)
            seen_ids.add(r["id"])

    # Rerank non-entity results (over-fetch to give final diversification
    # spare candidates when cross-tier duplicates are collapsed).
    remaining_slots = max(0, top_k - len(diversified_entity))
    if remaining_slots > 0 and non_entity:
        reranked_non_entity = _rerank_results(question, non_entity, remaining_slots * 3)
    else:
        reranked_non_entity = []

    # Final diversification across both tiers to collapse cross-tier
    # precision variants (e.g., DGETRF from entity tier + SGETRF from
    # non-entity tier). Non-entity results are not pre-diversified so the
    # final pass has enough candidates to backfill collapsed slots.
    combined = _diversify_results(diversified_entity + reranked_non_entity, top_k)

    # Filter low-confidence results (entity matches bypass threshold)
    return [
        r for r in combined
        if r.get("_entity_match") or r.get("score", 0) >= _MIN_SCORE_THRESHOLD
    ]


def format_results(results: list[dict], show_code: bool = False) -> None:
    """Pretty-print retrieval results to the console."""
    if not results:
        console.print("[yellow]No results found.[/]")
        return

    console.print(f"\n[bold]Found {len(results)} relevant chunks:[/]\n")

    for i, match in enumerate(results, 1):
        meta = match.get("metadata", {})
        score = match.get("score", 0)
        unit_name = meta.get("unit_name", "unknown")
        unit_type = meta.get("unit_type", "unknown")
        file_path = meta.get("file_path", "unknown")
        language = meta.get("language", "unknown")
        purpose = meta.get("purpose", "")
        start_line = meta.get("start_line", "?")
        end_line = meta.get("end_line", "?")
        params = meta.get("parameters", [])
        calls = meta.get("calls", [])
        role = meta.get("routine_role", "")

        # Header
        score_color = "green" if score > 0.8 else "yellow" if score > 0.6 else "red"
        role_tag = f"  [dim]({role})[/]" if role else ""
        header = f"[bold]{i}. {unit_type.upper()} {unit_name}[/]  [{score_color}]score: {score:.3f}[/]{role_tag}"
        console.print(header)

        # Details table
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column(style="dim")
        table.add_column()
        table.add_row("File", f"{file_path}:{start_line}-{end_line}")
        table.add_row("Language", language)
        if purpose:
            display_purpose = purpose[:200] + "..." if len(purpose) > 200 else purpose
            table.add_row("Purpose", display_purpose)
        if params:
            table.add_row("Parameters", ", ".join(params[:10]))
        if calls:
            table.add_row("Calls", ", ".join(calls[:10]))

        console.print(table)
        console.print()
