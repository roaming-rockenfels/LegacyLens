"""Retrieval evaluation library: precision, recall, MRR, hit rate, golden set loading."""

from __future__ import annotations

import json
from pathlib import Path


def precision_at_k(expected: set[str], retrieved: list[str], k: int) -> float:
    """Compute precision@k: fraction of top-k results that are expected.

    Args:
        expected: Set of expected unit names (case-insensitive).
        retrieved: Ordered list of retrieved unit names.
        k: Number of top results to consider.

    Returns:
        |expected ∩ top-k| / k
    """
    if k == 0:
        return 0.0
    expected_lower = {e.lower() for e in expected}
    top_k = [r.lower() for r in retrieved[:k]]
    hits = sum(1 for r in top_k if r in expected_lower)
    return hits / k


def recall_at_k(expected: set[str], retrieved: list[str], k: int) -> float:
    """Compute recall@k: fraction of expected units found in top-k.

    Args:
        expected: Set of expected unit names (case-insensitive).
        retrieved: Ordered list of retrieved unit names.
        k: Number of top results to consider.

    Returns:
        |expected ∩ top-k| / |expected|
    """
    if not expected:
        return 0.0
    expected_lower = {e.lower() for e in expected}
    top_k = {r.lower() for r in retrieved[:k]}
    hits = len(expected_lower & top_k)
    return hits / len(expected_lower)


def mrr(expected: set[str], retrieved: list[str]) -> float:
    """Compute Mean Reciprocal Rank: 1/rank of first expected hit.

    Args:
        expected: Set of expected unit names (case-insensitive).
        retrieved: Ordered list of retrieved unit names.

    Returns:
        1/rank of first hit, or 0.0 if no hit found.
    """
    expected_lower = {e.lower() for e in expected}
    for i, r in enumerate(retrieved):
        if r.lower() in expected_lower:
            return 1.0 / (i + 1)
    return 0.0


def hit_rate(expected: set[str], retrieved: list[str], k: int) -> float:
    """Compute hit rate: 1.0 if any expected unit is in top-k, else 0.0.

    Args:
        expected: Set of expected unit names (case-insensitive).
        retrieved: Ordered list of retrieved unit names.
        k: Number of top results to consider.

    Returns:
        1.0 or 0.0
    """
    expected_lower = {e.lower() for e in expected}
    top_k = {r.lower() for r in retrieved[:k]}
    return 1.0 if expected_lower & top_k else 0.0


def load_golden_set(path: str | Path) -> list[dict]:
    """Load a golden query set from a JSON file.

    Each entry must have:
        - "query": str
        - "expected_units": list[str]

    Optional:
        - "top_k": int (default 5)

    Args:
        path: Path to the JSON file.

    Returns:
        List of query dicts with defaults applied.

    Raises:
        ValueError: If any entry is missing the "query" key.
    """
    with open(path) as f:
        data = json.load(f)

    for i, entry in enumerate(data):
        if "query" not in entry:
            raise ValueError(f"Entry {i} missing required 'query' key")
        if "expected_units" not in entry:
            raise ValueError(f"Entry {i} missing required 'expected_units' key")
        entry.setdefault("top_k", 5)

    return data


def aggregate_metrics(per_query: list[dict]) -> dict:
    """Average per-query metric dicts into aggregate metrics.

    Args:
        per_query: List of dicts, each with keys like "precision", "recall", "mrr", "hit_rate".

    Returns:
        Dict with averaged values for each metric key.
    """
    if not per_query:
        return {}
    keys = [k for k in per_query[0] if k != "query"]
    return {k: sum(q[k] for q in per_query) / len(per_query) for k in keys}


def format_report(per_query: list[dict], aggregates: dict, threshold: float) -> str:
    """Format an evaluation report as a text table with PASS/FAIL verdict.

    Args:
        per_query: List of per-query metric dicts (must include "query").
        aggregates: Aggregated metrics dict.
        threshold: Recall threshold for PASS/FAIL.

    Returns:
        Formatted report string.
    """
    lines: list[str] = []

    # Header
    lines.append(f"{'Query':<50} {'Precision':>10} {'Recall':>10} {'MRR':>10} {'Hit Rate':>10}")
    lines.append("-" * 92)

    # Per-query rows
    for q in per_query:
        query_text = q["query"][:48]
        lines.append(
            f"{query_text:<50} {q['precision']:>10.3f} {q['recall']:>10.3f} "
            f"{q['mrr']:>10.3f} {q['hit_rate']:>10.3f}"
        )

    # Aggregates
    lines.append("-" * 92)
    lines.append(
        f"{'AVERAGE':<50} {aggregates['precision']:>10.3f} {aggregates['recall']:>10.3f} "
        f"{aggregates['mrr']:>10.3f} {aggregates['hit_rate']:>10.3f}"
    )

    # Verdict
    lines.append("")
    recall = aggregates.get("recall", 0.0)
    verdict = "PASS" if recall >= threshold else "FAIL"
    lines.append(f"Recall threshold: {threshold:.1%}  |  Aggregate recall: {recall:.1%}  |  {verdict}")

    return "\n".join(lines)
