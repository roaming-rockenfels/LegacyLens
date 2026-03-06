#!/usr/bin/env python3
"""One-time migration: add module_tier metadata to existing vectors.

Patches module_tier onto existing Pinecone vectors without re-embedding,
by classifying each vector's file_path using classify_module_tier().

Usage:
    python scripts/migrate_module_tier.py --namespace pydantic
"""

from __future__ import annotations

import argparse
import sys

from legacylens.chunkers.base import classify_module_tier
from legacylens.rag.storage import get_index


def migrate(namespace: str, batch_size: int = 100, dry_run: bool = False) -> None:
    index = get_index()
    ids: list[str] = []
    for id_list in index.list(namespace=namespace):
        ids.extend(id_list)

    print(f"Found {len(ids)} vectors in namespace '{namespace}'")

    # Fetch metadata in batches and compute updates
    updates: list[tuple[str, dict]] = []
    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i : i + batch_size]
        fetched = index.fetch(ids=batch_ids, namespace=namespace)
        for vec_id, vec in fetched.vectors.items():
            file_path = vec.metadata.get("file_path", "")
            tier = classify_module_tier(file_path)
            if vec.metadata.get("module_tier") != tier:
                updates.append((vec_id, {"module_tier": tier}))

    print(f"Vectors needing update: {len(updates)}")
    tier_counts: dict[str, int] = {}
    for _, meta in updates:
        t = meta["module_tier"]
        tier_counts[t] = tier_counts.get(t, 0) + 1
    for t, count in sorted(tier_counts.items()):
        print(f"  {t}: {count}")

    if dry_run:
        print("Dry run — no updates applied.")
        return

    # Apply updates
    applied = 0
    for vec_id, metadata in updates:
        index.update(id=vec_id, set_metadata=metadata, namespace=namespace)
        applied += 1
        if applied % 100 == 0:
            print(f"  Updated {applied}/{len(updates)}...")

    print(f"Done. Updated {applied} vectors.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate module_tier metadata onto existing vectors.")
    parser.add_argument("--namespace", required=True, help="Pinecone namespace to migrate.")
    parser.add_argument("--batch-size", type=int, default=100, help="Fetch batch size.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change without applying.")
    args = parser.parse_args()

    migrate(args.namespace, batch_size=args.batch_size, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
