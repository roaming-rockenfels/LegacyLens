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
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from legacylens.rag.embeddings import enable_embedding_cache, save_embedding_cache
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

        result = retrieve_with_metrics(query, top_k=top_k)
        retrieved_names = [
            r.get("metadata", {}).get("unit_name", "") for r in result.results
        ]

        per_query.append({
            "query": query,
            "precision": precision_at_k(expected, retrieved_names, k=top_k),
            "recall": recall_at_k(expected, retrieved_names, k=top_k),
            "mrr": mrr(expected, retrieved_names),
            "hit_rate": hit_rate(expected, retrieved_names, k=top_k),
        })

    if not args.no_cache:
        save_embedding_cache()

    agg = aggregate_metrics(per_query)
    report = format_report(per_query, agg, threshold=args.threshold)
    print(report)

    if agg.get("recall", 0.0) < args.threshold:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
