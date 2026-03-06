"""Retrieval pipeline: query → embed → search → format results."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

from legacylens.rag.embeddings import embed_query, _get_client as _get_voyage_client
from legacylens.rag.keyword_index import KeywordIndex
from legacylens.rag.storage import fetch_vectors, query_vectors

console = Console()


@dataclass
class RetrievalMetrics:
    """Counters returned alongside retrieval results for instrumentation."""

    entity_matches: int = 0
    threshold_filtered: int = 0
    variants_collapsed: int = 0
    score_distribution: dict = field(default_factory=lambda: {"min": 0.0, "max": 0.0, "mean": 0.0})


@dataclass
class RetrievalResult:
    """Wraps results list + metrics from a retrieval call."""

    results: list[dict] = field(default_factory=list)
    metrics: RetrievalMetrics = field(default_factory=RetrievalMetrics)


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


# Python identifier patterns
_RE_SNAKE_CASE = re.compile(r"\b([a-z][a-z0-9_]{2,30})\b")
_RE_CAMEL_CASE = re.compile(r"\b([A-Z][a-z][a-zA-Z0-9]{1,30})\b")
_RE_DOTTED_PATH = re.compile(r"\b([a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+)\b")

_PYTHON_STOPWORDS = {
    "the", "and", "for", "not", "are", "but", "how", "what", "does", "this",
    "with", "from", "that", "which", "where", "when", "into", "each", "both",
    "all", "its", "has", "use", "find", "show", "can", "why", "who", "will",
    "get", "set", "put", "run", "let", "try", "was", "had", "been", "have",
    "about", "class", "method", "function", "module", "import", "return",
    "self", "none", "true", "false", "type", "list", "dict", "str", "int",
}

# Concept-to-decorator mapping for decorator-aware queries
_DECORATOR_CONCEPTS = {
    "validate": ["validator", "field_validator", "model_validator", "validate_call"],
    "serialize": ["serializer", "field_serializer", "model_serializer"],
    "config": ["dataclass", "ConfigDict"],
}


_RE_LEGACY_INTENT = re.compile(
    r"\b(v1|legacy|compat|compatibility|migrate|migration|upgrade|upgrading|old\s+api)\b",
    re.IGNORECASE,
)


def _query_wants_legacy(question: str) -> bool:
    """Return True if the query explicitly asks about v1/legacy code."""
    return bool(_RE_LEGACY_INTENT.search(question))


def _extract_python_entities(question: str) -> dict[str, list[str]]:
    """Detect Python-style identifiers in a query.

    Returns:
        Dict with "classes", "functions", and "modules" lists.
    """
    classes = [m for m in _RE_CAMEL_CASE.findall(question) if m.lower() not in _PYTHON_STOPWORDS]
    functions = [m for m in _RE_SNAKE_CASE.findall(question) if m not in _PYTHON_STOPWORDS]
    modules = _RE_DOTTED_PATH.findall(question)

    return {"classes": classes, "functions": functions, "modules": modules}


def _base_routine_name(unit_name: str, language: str = "fortran") -> str:
    """Strip the LAPACK precision prefix to get the base routine name.

    Only applies SDCZ stripping for Fortran. For other languages, returns as-is.
    """
    if not unit_name:
        return unit_name
    if language != "fortran":
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

    For non-Fortran chunks, tier-based dedup collapses same-named units across
    module tiers, preferring current over legacy/deprecated/internal.
    """
    _TIER_PRIORITY = {"current": 0, "internal": 1, "deprecated": 2, "legacy": 3}

    base_to_result: dict[str, dict] = {}
    base_order: list[str] = []

    for r in results:
        meta = r.get("metadata", {})
        unit_name = meta.get("unit_name", "")
        language = meta.get("language", "fortran")
        base = _base_routine_name(unit_name, language=language)

        if base not in base_to_result:
            base_to_result[base] = r
            base_order.append(base)
        elif language == "fortran" and unit_name and unit_name[0] == "D" and base_to_result[base].get("metadata", {}).get("unit_name", "")[0] != "D":
            base_to_result[base] = r
        else:
            # Tier-based preference for Python/other languages
            new_tier = _TIER_PRIORITY.get(meta.get("module_tier", "current"), 0)
            old_tier = _TIER_PRIORITY.get(base_to_result[base].get("metadata", {}).get("module_tier", "current"), 0)
            if new_tier < old_tier:
                base_to_result[base] = r

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
            doc = f"{meta.get('unit_type', '')} {meta.get('unit_name', '')}"
            if meta.get("base_classes"):
                doc += f" inherits {', '.join(meta['base_classes'])}"
            if meta.get("decorators"):
                doc += f" @{', @'.join(meta['decorators'])}"
            doc += f": {meta.get('purpose', '')}"
            if meta.get("parameters"):
                doc += f" Parameters: {', '.join(meta['parameters'][:10])}"
            if meta.get("calls"):
                doc += f" Calls: {', '.join(meta['calls'][:10])}"
            documents.append(doc)
        reranked = client.rerank(question, documents, model="rerank-2", top_k=top_k)
        return [results[rr.index] for rr in reranked.results]
    except Exception:
        return results[:top_k]


def retrieve_with_metrics(
    question: str,
    top_k: int = 5,
    pin_unit: str | None = None,
    namespace: str | None = None,
    languages: list[str] | None = None,
) -> RetrievalResult:
    """Retrieve relevant code chunks with instrumentation metrics.

    Args:
        question: Natural language query about the codebase.
        top_k: Number of results to return.
        pin_unit: If set, ensure this unit_name appears in results.
        namespace: Pinecone namespace to query (source isolation).
        languages: Pre-scanned languages for this source (drives strategy).

    Returns:
        RetrievalResult with .results and .metrics.
    """
    metrics = RetrievalMetrics()
    embedding = embed_query(question)
    ns = {"namespace": namespace} if namespace else {}

    if pin_unit:
        pinned = query_vectors(
            embedding, top_k=3,
            filter={"unit_name": pin_unit.upper()},
            **ns,
        )
        semantic = query_vectors(embedding, top_k=top_k * 3, **ns)

        seen_ids = {r["id"] for r in pinned}
        merged = list(pinned)
        for r in semantic:
            if r["id"] not in seen_ids and len(merged) < top_k:
                merged.append(r)
                seen_ids.add(r["id"])

        metrics.entity_matches = len(pinned)
        if merged:
            scores = [r.get("score", 0) for r in merged]
            metrics.score_distribution = {
                "min": min(scores),
                "max": max(scores),
                "mean": sum(scores) / len(scores),
            }
        return RetrievalResult(results=merged, metrics=metrics)

    # Language-aware entity extraction
    use_fortran_strategy = not languages or "fortran" in languages
    use_python_strategy = languages and "python" in languages and "fortran" not in languages

    # Skip legacy filtering when the query explicitly asks about v1/legacy code
    _wants_legacy = use_python_strategy and _query_wants_legacy(question)

    if use_python_strategy:
        entities = _extract_python_entities(question)
        entity_results: list[dict] = []

        def _py_entity_filter(base_filter: dict) -> dict:
            if _wants_legacy:
                return base_filter
            return {**base_filter, "module_tier": {"$ne": "legacy"}}

        for cls in entities.get("classes", []):
            entity_results.extend(
                query_vectors(embedding, top_k=top_k, filter=_py_entity_filter({"unit_name": cls}), **ns)
            )
            # Find classes that inherit from the queried class
            entity_results.extend(
                query_vectors(embedding, top_k=top_k, filter=_py_entity_filter({"base_classes": cls}), **ns)
            )
        for func in entities.get("functions", []):
            entity_results.extend(
                query_vectors(embedding, top_k=top_k, filter=_py_entity_filter({"unit_name": func}), **ns)
            )
            entity_results.extend(
                query_vectors(embedding, top_k=top_k, filter=_py_entity_filter({"calls": func}), **ns)
            )
        for mod in entities.get("modules", []):
            entity_results.extend(
                query_vectors(embedding, top_k=top_k, filter=_py_entity_filter({"uses": mod.split(".")[0]}), **ns)
            )

        # Decorator-aware queries: when query contains a concept keyword
        q_lower = question.lower()
        for concept, decorator_names in _DECORATOR_CONCEPTS.items():
            if concept in q_lower:
                for dec_name in decorator_names:
                    entity_results.extend(
                        query_vectors(embedding, top_k=top_k, filter=_py_entity_filter({"decorators": dec_name}), **ns)
                    )
    elif use_fortran_strategy:
        entities = _extract_fortran_entities(question)
        entity_results = []

        for param in entities.get("parameters", []):
            entity_results.extend(
                query_vectors(embedding, top_k=top_k, filter={"parameters": param, "precision": "double"}, **ns)
            )
            entity_results.extend(
                query_vectors(embedding, top_k=top_k, filter={"parameters": param}, **ns)
            )
        for routine in entities.get("routines", []):
            entity_results.extend(
                query_vectors(embedding, top_k=3, filter={"unit_name": routine.upper()}, **ns)
            )
            entity_results.extend(
                query_vectors(embedding, top_k=top_k, filter={"calls": routine.upper(), "precision": "double"}, **ns)
            )
            entity_results.extend(
                query_vectors(embedding, top_k=top_k, filter={"calls": routine.upper()}, **ns)
            )
    else:
        entity_results = []

    # Tag, deduplicate, sort by role, rerank, then diversify entity results
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

    # Two-pass retrieval: drivers first (Fortran), then general
    over_fetch = top_k * 3

    if use_python_strategy and not _wants_legacy:
        # Pass 1: Prefer current code (exclude legacy)
        current_results = query_vectors(
            embedding, top_k=top_k,
            filter={"module_tier": {"$ne": "legacy"}},
            **ns,
        )
        driver_results = []
    elif use_python_strategy:
        current_results = []
        driver_results = []
    elif use_fortran_strategy:
        current_results = []
        driver_results = query_vectors(
            embedding, top_k=top_k,
            filter={"routine_role": "driver"},
            **ns,
        )
    else:
        current_results = []
        driver_results = []

    # Pass 2: Unfiltered (catch unique concepts from all tiers)
    general_results = query_vectors(embedding, top_k=over_fetch, **ns)

    # BM25 keyword search backstop
    keyword_results: list[dict] = []
    kw_index = KeywordIndex()
    kw_source = namespace or "default"
    if kw_index.load(kw_source):
        kw_hits = kw_index.search(question, top_k=top_k * 2)
        if kw_hits:
            kw_ids = [cid for cid, _score in kw_hits]
            try:
                fetched = fetch_vectors(kw_ids, namespace=namespace)
                for cid, kw_score in kw_hits:
                    vec = fetched.get(cid)
                    if vec and cid not in seen_ids:
                        keyword_results.append({
                            "id": cid,
                            "score": kw_score,
                            "metadata": vec.metadata if hasattr(vec, "metadata") else {},
                            "_keyword_match": True,
                        })
            except Exception:
                pass  # Graceful fallback if fetch fails

    # Merge non-entity results
    non_entity: list[dict] = []
    for r in current_results + driver_results + general_results + keyword_results:
        if r["id"] not in seen_ids:
            non_entity.append(r)
            seen_ids.add(r["id"])

    remaining_slots = max(0, top_k - len(diversified_entity))
    if remaining_slots > 0 and non_entity:
        reranked_non_entity = _rerank_results(question, non_entity, remaining_slots * 3)
    else:
        reranked_non_entity = []

    pre_diversify_count = len(diversified_entity + reranked_non_entity)
    combined = _diversify_results(diversified_entity + reranked_non_entity, top_k)
    metrics.variants_collapsed = pre_diversify_count - len(combined)

    pre_filter = combined
    filtered = [
        r for r in pre_filter
        if r.get("_entity_match") or r.get("_keyword_match") or r.get("score", 0) >= _MIN_SCORE_THRESHOLD
    ]
    # If threshold filtering is too aggressive, keep at least min(top_k, 3) results
    # so the LLM always has some context to work with
    min_results = min(top_k, 3)
    if len(filtered) < min_results and len(pre_filter) >= min_results:
        filtered = pre_filter[:min_results]
    metrics.threshold_filtered = len(pre_filter) - len(filtered)
    metrics.entity_matches = sum(1 for r in filtered if r.get("_entity_match"))

    if filtered:
        scores = [r.get("score", 0) for r in filtered]
        metrics.score_distribution = {
            "min": min(scores),
            "max": max(scores),
            "mean": sum(scores) / len(scores),
        }

    return RetrievalResult(results=filtered, metrics=metrics)


def retrieve(
    question: str,
    top_k: int = 5,
    pin_unit: str | None = None,
    namespace: str | None = None,
    languages: list[str] | None = None,
) -> list[dict]:
    """Retrieve relevant code chunks for a natural language question.

    Thin wrapper around retrieve_with_metrics() that returns only the results list.
    """
    return retrieve_with_metrics(question, top_k, pin_unit, namespace=namespace, languages=languages).results


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

        if show_code:
            from legacylens.rag.source_reader import read_source_snippet

            sl = start_line if isinstance(start_line, int) else 0
            el = end_line if isinstance(end_line, int) else 0
            if sl and el:
                snippet = read_source_snippet(file_path, sl, el)
                if snippet:
                    lexer = meta.get("language", "text")
                    syntax = Syntax(
                        snippet,
                        lexer,
                        line_numbers=True,
                        start_line=sl,
                        theme="monokai",
                    )
                    console.print(syntax)

        console.print()
