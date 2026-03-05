#!/usr/bin/env python3
"""Retrieval evaluation script.

Runs golden queries against the live Pinecone index, computes per-query
and aggregate metrics, prints a report, and exits non-zero if aggregate
recall falls below the threshold.

Usage:
    .venv/bin/python evals/eval_retrieval.py
    .venv/bin/python evals/eval_retrieval.py --threshold 0.0  # baseline run
    .venv/bin/python evals/eval_retrieval.py --golden evals/golden_queries.json --threshold 0.7
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import sys
import time
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from legacylens.rag.embeddings import enable_embedding_cache, save_embedding_cache
from legacylens.rag.generate import generate_answer_stream
from legacylens.rag.retrieve import retrieve_with_metrics
from legacylens.rag.eval import (
    precision_at_k,
    recall_at_k,
    mrr,
    hit_rate,
    load_golden_set,
    aggregate_metrics,
    format_report,
)

EVALS_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_PATH = EVALS_DIR / ".embedding_cache.json"

CATEGORY_LABELS = {
    "entity-direct": "Entity (direct)",
    "entity-callers": "Entity (callers)",
    "parameter-based": "Parameter-based",
    "semantic": "Semantic",
    "utility": "Utility",
}

# Preserve display order
CATEGORY_ORDER = ["entity-direct", "entity-callers", "parameter-based", "semantic", "utility"]


def format_category_report(per_query_with_category: list[dict]) -> str:
    """Format a per-category summary table."""
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for entry in per_query_with_category:
        by_cat[entry["category"]].append(entry)

    has_ttfs = "ttfs_ms" in per_query_with_category[0]
    lines: list[str] = []
    header = f"\n{'Category':<20} {'Queries':>8} {'Precision':>10} {'Recall':>10} {'MRR':>10} {'Hit Rate':>10} {'Retrieval':>10}"
    if has_ttfs:
        header += f" {'TTFS':>10}"
    lines.append(header)
    lines.append("-" * (90 if has_ttfs else 80))

    for cat_key in CATEGORY_ORDER:
        if cat_key not in by_cat:
            continue
        entries = by_cat[cat_key]
        agg = aggregate_metrics(entries)
        label = CATEGORY_LABELS.get(cat_key, cat_key)
        line = (
            f"{label:<20} {len(entries):>8} {agg['precision']:>10.3f} {agg['recall']:>10.3f} "
            f"{agg['mrr']:>10.3f} {agg['hit_rate']:>10.3f} {agg['latency_ms']:>8.0f}ms"
        )
        if has_ttfs:
            line += f" {agg['ttfs_ms']:>8.0f}ms"
        lines.append(line)

    overall = aggregate_metrics(per_query_with_category)
    lines.append("-" * (90 if has_ttfs else 80))
    line = (
        f"{'Overall':<20} {len(per_query_with_category):>8} {overall['precision']:>10.3f} {overall['recall']:>10.3f} "
        f"{overall['mrr']:>10.3f} {overall['hit_rate']:>10.3f} {overall['latency_ms']:>8.0f}ms"
    )
    if has_ttfs:
        line += f" {overall['ttfs_ms']:>8.0f}ms"
    lines.append(line)
    return "\n".join(lines)


def format_category_markdown(per_query_with_category: list[dict], top_k: int = 5) -> str:
    """Format a markdown table of per-category results for README."""
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for entry in per_query_with_category:
        by_cat[entry["category"]].append(entry)

    has_ttfs = "ttfs_ms" in per_query_with_category[0]

    lines: list[str] = []
    lines.append(f"Retrieval quality evaluated against {len(per_query_with_category)} golden queries across {len(by_cat)} categories (top-k={top_k}):")
    lines.append("")
    header = f"| Category | Queries | Precision@{top_k} | Recall@{top_k} | MRR | Hit Rate | Retrieval (ms) |"
    sep = "|---|---|---|---|---|---|---|"
    if has_ttfs:
        header += " TTFS (ms) |"
        sep += "---|"
    lines.append(header)
    lines.append(sep)

    for cat_key in CATEGORY_ORDER:
        if cat_key not in by_cat:
            continue
        entries = by_cat[cat_key]
        agg = aggregate_metrics(entries)
        label = CATEGORY_LABELS.get(cat_key, cat_key)
        row = f"| {label} | {len(entries)} | {agg['precision']:.2f} | {agg['recall']:.2f} | {agg['mrr']:.2f} | {agg['hit_rate']:.2f} | {agg['latency_ms']:.0f} |"
        if has_ttfs:
            row += f" {agg['ttfs_ms']:.0f} |"
        lines.append(row)

    overall = aggregate_metrics(per_query_with_category)
    row = f"| **Overall** | **{len(per_query_with_category)}** | **{overall['precision']:.2f}** | **{overall['recall']:.2f}** | **{overall['mrr']:.2f}** | **{overall['hit_rate']:.2f}** | **{overall['latency_ms']:.0f}** |"
    if has_ttfs:
        row += f" **{overall['ttfs_ms']:.0f}** |"
    lines.append(row)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality against golden queries.")
    parser.add_argument(
        "--golden",
        default=str(EVALS_DIR / "golden_queries.json"),
        help="Path to golden query set JSON (default: evals/golden_queries.json)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Minimum aggregate recall for PASS (default: 0.7)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Override top_k for all queries (default: use per-query top_k)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable embedding cache (always call Voyage API)",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Print markdown-formatted category table (for README)",
    )
    parser.add_argument(
        "--e2e",
        action="store_true",
        help="Also measure time-to-first-stream (retrieval + LLM TTFS)",
    )
    args = parser.parse_args()

    if not args.no_cache:
        n = enable_embedding_cache(DEFAULT_CACHE_PATH)
        if n:
            print(f"Loaded {n} cached embeddings from {DEFAULT_CACHE_PATH}")

    golden = load_golden_set(args.golden)
    per_query: list[dict] = []

    for entry in golden:
        query = entry["query"]
        expected = set(entry["expected_units"])
        top_k = args.top_k if args.top_k is not None else entry.get("top_k", 5)

        t0 = time.perf_counter()
        result = retrieve_with_metrics(query, top_k=top_k)
        latency_ms = (time.perf_counter() - t0) * 1000

        ttfs_ms = None
        if args.e2e:
            stream = generate_answer_stream(query, result.results)
            first_token = next(stream, None)  # time to first token
            ttfs_ms = (time.perf_counter() - t0) * 1000  # retrieval + TTFT
            # Drain remaining stream to avoid connection issues
            for _ in stream:
                pass

        retrieved_names = [
            r.get("metadata", {}).get("unit_name", "") for r in result.results
        ]

        row: dict = {
            "query": query,
            "category": entry.get("category", "unknown"),
            "precision": precision_at_k(expected, retrieved_names, k=top_k),
            "recall": recall_at_k(expected, retrieved_names, k=top_k),
            "mrr": mrr(expected, retrieved_names),
            "hit_rate": hit_rate(expected, retrieved_names, k=top_k),
            "latency_ms": latency_ms,
        }
        if ttfs_ms is not None:
            row["ttfs_ms"] = ttfs_ms
        per_query.append(row)

    if not args.no_cache:
        save_embedding_cache()

    # Per-query report (existing)
    agg = aggregate_metrics(per_query)
    report = format_report(per_query, agg, threshold=args.threshold)
    print(report)

    # Category breakdown
    print(format_category_report(per_query))

    # Markdown output
    if args.markdown:
        effective_top_k = args.top_k if args.top_k is not None else 5
        print("\n--- Markdown for README ---\n")
        print(format_category_markdown(per_query, top_k=effective_top_k))

    if agg.get("recall", 0.0) < args.threshold:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
