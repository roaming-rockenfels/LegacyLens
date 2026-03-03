#!/usr/bin/env python3
"""One-time script to update existing Pinecone vectors with routine_role metadata.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/update_roles.py [--dry-run]

Classifies every vector in the index as: driver, computational, auxiliary, or blas.
Uses in-place metadata update (no re-embedding needed).
"""

from __future__ import annotations

import sys
import time

from legacylens.chunkers.fortran import _classify_role, _DRIVER_BASE_NAMES
from legacylens.rag.storage import get_index, update_metadata

# How many IDs to fetch per pagination page
_PAGE_SIZE = 100


def classify_from_metadata(vector_id: str, metadata: dict) -> str | None:
    """Determine routine_role from existing metadata."""
    unit_name = metadata.get("unit_name", "")
    file_path = metadata.get("file_path", "")
    return _classify_role(unit_name, file_path)


def main(dry_run: bool = False) -> None:
    index = get_index()
    stats = index.describe_index_stats()
    total_vectors = stats.total_vector_count
    print(f"Index has {total_vectors} vectors")
    print(f"Driver base names: {len(_DRIVER_BASE_NAMES)}")

    # Paginate through all vectors
    updates: list[tuple[str, dict]] = []
    role_counts: dict[str, int] = {}
    processed = 0

    # Use list() to paginate through all IDs
    for id_batch in index.list():
        if not id_batch:
            break
        # Fetch metadata for this batch
        fetch_result = index.fetch(ids=list(id_batch))
        for vec_id, vec_data in fetch_result.vectors.items():
            metadata = vec_data.metadata or {}
            role = classify_from_metadata(vec_id, metadata)

            if role:
                existing_role = metadata.get("routine_role")
                if existing_role != role:
                    updates.append((vec_id, {"routine_role": role}))
                role_counts[role] = role_counts.get(role, 0) + 1
            else:
                role_counts["unknown"] = role_counts.get("unknown", 0) + 1

            processed += 1

        print(f"  Processed {processed} vectors...")

    print(f"\nRole distribution:")
    for role, count in sorted(role_counts.items()):
        print(f"  {role}: {count}")
    print(f"\nVectors to update: {len(updates)}")

    if dry_run:
        print("\n[DRY RUN] No updates applied.")
        return

    if not updates:
        print("No updates needed.")
        return

    print(f"\nApplying {len(updates)} metadata updates...")
    updated = update_metadata(updates)
    print(f"Updated {updated} vectors.")


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    main(dry_run=dry_run)
