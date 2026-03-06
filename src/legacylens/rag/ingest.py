"""Ingestion pipeline: chunk → embed → store."""

from __future__ import annotations

import time

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from legacylens.chunkers import FortranChunker, PythonChunker, ChunkerRegistry
from legacylens.chunkers.base import Chunk
from legacylens.rag.embeddings import embed_texts
from legacylens.rag.keyword_index import KeywordIndex
from legacylens.rag.storage import upsert_vectors, get_index_stats

console = Console()

# Batch size for embedding calls.
# Free tier (no payment): 3 RPM / 10K TPM → use batch=3 with 25s delay.
# With payment method: 300 RPM / 1M TPM → use batch=128, delay=0.
EMBED_BATCH_SIZE = 3
EMBED_DELAY = 25.0  # seconds between batches — keeps us under 3 RPM free tier


def create_default_registry() -> ChunkerRegistry:
    """Create a registry with all available chunkers."""
    registry = ChunkerRegistry()
    registry.register(FortranChunker())
    registry.register(PythonChunker())
    return registry


def ingest_directory(directory: str, namespace: str | None = None) -> dict:
    """Ingest a codebase directory into Pinecone.

    Args:
        directory: Path to codebase directory.
        namespace: Pinecone namespace for source isolation.

    Returns:
        Stats dict with counts and timings.
    """
    start_time = time.time()

    # Step 1: Chunk
    console.print(f"\n[bold blue]Step 1/3:[/] Chunking files in {directory}...")
    registry = create_default_registry()
    chunks = registry.chunk_directory(directory)
    chunk_time = time.time() - start_time
    console.print(f"  Found [green]{len(chunks)}[/] chunks in {chunk_time:.1f}s")

    if not chunks:
        console.print("[red]No chunks found. Check the directory path.[/]")
        return {"chunks": 0, "vectors": 0, "time": 0}

    # Step 2: Embed
    console.print(f"\n[bold blue]Step 2/3:[/] Generating embeddings with Voyage Code 3...")
    embed_start = time.time()
    all_vectors: list[tuple[str, list[float], dict]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Embedding", total=len(chunks))

        for i in range(0, len(chunks), EMBED_BATCH_SIZE):
            batch = chunks[i : i + EMBED_BATCH_SIZE]
            texts = [c.enriched_content for c in batch]
            embeddings = embed_texts(texts, input_type="document")

            for chunk, embedding in zip(batch, embeddings):
                meta = chunk.metadata.to_pinecone_metadata()
                all_vectors.append((chunk.chunk_id, embedding, meta))

            progress.advance(task, len(batch))
            time.sleep(EMBED_DELAY)

    embed_time = time.time() - embed_start
    console.print(f"  Generated [green]{len(all_vectors)}[/] embeddings in {embed_time:.1f}s")

    # Step 3: Store
    console.print(f"\n[bold blue]Step 3/3:[/] Upserting to Pinecone...")
    store_start = time.time()
    stored = upsert_vectors(all_vectors, namespace=namespace)
    store_time = time.time() - store_start
    console.print(f"  Upserted [green]{stored}[/] vectors in {store_time:.1f}s")

    # Build BM25 keyword index
    kw_index = KeywordIndex()
    kw_index.build(chunks)
    kw_index.save(namespace or "default")
    console.print(f"  Built keyword index for [green]{namespace or 'default'}[/]")

    total_time = time.time() - start_time
    stats = get_index_stats()

    console.print(f"\n[bold green]Ingestion complete![/] {total_time:.1f}s total")
    console.print(f"  Index stats: {stats.get('total_vector_count', 'N/A')} total vectors")

    return {
        "chunks": len(chunks),
        "vectors": stored,
        "chunk_time": chunk_time,
        "embed_time": embed_time,
        "store_time": store_time,
        "total_time": total_time,
    }
