#!/usr/bin/env python3
"""Build BM25 keyword index for an already-ingested source.

Fetches all vectors from Pinecone for the given namespace, extracts metadata,
and builds a local BM25 index.

Usage:
    python scripts/build_keyword_index.py --namespace pydantic
"""

from __future__ import annotations

import argparse
import sys

from legacylens.chunkers.base import Chunk, ChunkMetadata
from legacylens.rag.keyword_index import KeywordIndex
from legacylens.rag.storage import list_vectors, fetch_vectors


def main() -> None:
    parser = argparse.ArgumentParser(description="Build BM25 keyword index from Pinecone vectors")
    parser.add_argument("--namespace", required=True, help="Pinecone namespace (source name)")
    parser.add_argument("--batch-size", type=int, default=100, help="Fetch batch size")
    args = parser.parse_args()

    print(f"Listing vectors in namespace '{args.namespace}'...")
    all_ids = list_vectors(limit=100_000)
    if not all_ids:
        print("No vectors found.")
        sys.exit(1)
    print(f"  Found {len(all_ids)} vector IDs")

    # Fetch metadata in batches
    print("Fetching metadata...")
    chunks: list[Chunk] = []
    for i in range(0, len(all_ids), args.batch_size):
        batch_ids = all_ids[i : i + args.batch_size]
        fetched = fetch_vectors(batch_ids, namespace=args.namespace)
        for vid, vec in fetched.items():
            meta = vec.metadata if hasattr(vec, "metadata") else {}
            if not meta:
                continue
            cm = ChunkMetadata(
                file_path=meta.get("file_path", ""),
                start_line=meta.get("start_line", 0),
                end_line=meta.get("end_line", 0),
                unit_name=meta.get("unit_name", ""),
                unit_type=meta.get("unit_type", ""),
                language=meta.get("language", ""),
                parameters=meta.get("parameters", []),
                calls=meta.get("calls", []),
                external_deps=meta.get("external_deps", []),
                uses=meta.get("uses", []),
                purpose=meta.get("purpose", ""),
                module_tier=meta.get("module_tier", "current"),
                base_classes=meta.get("base_classes", []),
                decorators=meta.get("decorators", []),
                return_type=meta.get("return_type", ""),
                visibility=meta.get("visibility", "public"),
                dunder_methods=meta.get("dunder_methods", []),
            )
            # Build a minimal Chunk — content/enriched_content not needed for BM25
            chunk = Chunk(content="", enriched_content="", metadata=cm)
            # Override chunk_id to match the stored vector ID
            chunk._override_id = vid
            chunks.append(chunk)

    print(f"  Built {len(chunks)} chunk objects")

    # Monkey-patch chunk_id for these synthetic chunks
    for chunk in chunks:
        original_prop = type(chunk).chunk_id.fget
        chunk.__class__ = type("_Chunk", (Chunk,), {
            "chunk_id": property(lambda self: self._override_id)
        })

    kw_index = KeywordIndex()
    kw_index.build(chunks)
    kw_index.save(args.namespace)
    print(f"Keyword index saved for '{args.namespace}'")


if __name__ == "__main__":
    main()
