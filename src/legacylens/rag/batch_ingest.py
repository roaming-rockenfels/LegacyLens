"""Batch ingestion pipeline using Voyage Batch API.

Workflow:
1. Chunk all files
2. Write JSONL batch file
3. Upload to Voyage Files API
4. Create batch job
5. Poll for completion
6. Download embeddings
7. Upsert to Pinecone
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from legacylens.chunkers.fortran import FortranChunker
from legacylens.chunkers.base import Chunk
from legacylens.config import VOYAGE_API_KEY
from legacylens.rag.embeddings import MODEL, DIMENSION
from legacylens.rag.storage import upsert_vectors, get_index_stats, delete_index

console = Console()

VOYAGE_BASE = "https://api.voyageai.com/v1"
# Max inputs per batch request line.
# voyage-code-3 has a ~120K token limit per request.
# Average chunk is ~3K tokens, so 20 per request stays safely under.
INPUTS_PER_REQUEST = 20


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {VOYAGE_API_KEY()}",
        "Content-Type": "application/json",
    }


def batch_ingest_directory(directory: str, reset_index: bool = True) -> dict:
    """Ingest a codebase directory using the Voyage Batch API.

    Returns:
        Stats dict with counts and timings.
    """
    start_time = time.time()

    # Step 1: Chunk
    console.print(f"\n[bold blue]Step 1/6:[/] Chunking files in {directory}...")
    chunker = FortranChunker()
    chunks = chunker.chunk_directory(directory)
    chunk_time = time.time() - start_time
    console.print(f"  Found [green]{len(chunks)}[/] chunks in {chunk_time:.1f}s")

    if not chunks:
        console.print("[red]No chunks found.[/]")
        return {"chunks": 0, "vectors": 0, "time": 0}

    # Step 2: Write JSONL batch file
    console.print(f"\n[bold blue]Step 2/6:[/] Writing batch input file...")
    batch_file = Path("/tmp/legacylens_batch_input.jsonl")
    request_map: dict[str, list[int]] = {}  # custom_id -> chunk indices

    with open(batch_file, "w") as f:
        request_idx = 0
        for i in range(0, len(chunks), INPUTS_PER_REQUEST):
            batch = chunks[i : i + INPUTS_PER_REQUEST]
            texts = [c.enriched_content for c in batch]
            custom_id = f"req_{request_idx}"
            request_map[custom_id] = list(range(i, i + len(batch)))

            line = json.dumps({
                "custom_id": custom_id,
                "body": {
                    "input": texts,
                },
            })
            f.write(line + "\n")
            request_idx += 1

    console.print(f"  Wrote {request_idx} batch requests to {batch_file}")

    # Step 3: Upload file
    console.print(f"\n[bold blue]Step 3/6:[/] Uploading batch file to Voyage...")
    with httpx.Client(timeout=60) as client:
        resp = client.post(
            f"{VOYAGE_BASE}/files",
            headers={"Authorization": f"Bearer {VOYAGE_API_KEY()}"},
            files={"file": ("batch_input.jsonl", open(batch_file, "rb"), "application/jsonl")},
            data={"purpose": "batch"},
        )
        resp.raise_for_status()
        file_data = resp.json()
        file_id = file_data["id"]
        console.print(f"  Uploaded: file_id={file_id}")

    # Step 4: Create batch job
    console.print(f"\n[bold blue]Step 4/6:[/] Creating batch embedding job...")
    with httpx.Client(timeout=60) as client:
        resp = client.post(
            f"{VOYAGE_BASE}/batches",
            headers=_headers(),
            json={
                "endpoint": "/v1/embeddings",
                "completion_window": "12h",
                "request_params": {
                    "model": MODEL,
                    "input_type": "document",
                },
                "input_file_id": file_id,
                "metadata": {"corpus": "lapack"},
            },
        )
        resp.raise_for_status()
        batch_data = resp.json()
        batch_id = batch_data["id"]
        status = batch_data["status"]
        console.print(f"  Batch created: id={batch_id}, status={status}")

    # Step 5: Poll for completion
    console.print(f"\n[bold blue]Step 5/6:[/] Waiting for batch completion...")
    poll_interval = 10  # seconds
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Waiting for Voyage batch...", total=None)

        while True:
            time.sleep(poll_interval)
            with httpx.Client(timeout=30) as client:
                resp = client.get(
                    f"{VOYAGE_BASE}/batches/{batch_id}",
                    headers=_headers(),
                )
                resp.raise_for_status()
                batch_status = resp.json()

            status = batch_status["status"]
            progress.update(task, description=f"Batch status: {status}")

            if status == "completed":
                output_file_id = batch_status.get("output_file_id")
                console.print(f"  Batch completed! Output file: {output_file_id}")
                break
            elif status in ("failed", "expired", "cancelled"):
                error_file_id = batch_status.get("error_file_id")
                console.print(f"[red]Batch {status}![/] Error file: {error_file_id}")
                return {"chunks": len(chunks), "vectors": 0, "time": time.time() - start_time, "error": status}

            # Increase poll interval gradually
            poll_interval = min(poll_interval + 5, 60)

    embed_time = time.time() - start_time - chunk_time

    # Step 6: Download results and upsert to Pinecone
    console.print(f"\n[bold blue]Step 6/6:[/] Downloading embeddings and upserting to Pinecone...")

    with httpx.Client(timeout=120, follow_redirects=True) as client:
        resp = client.get(
            f"{VOYAGE_BASE}/files/{output_file_id}/content",
            headers=_headers(),
        )
        resp.raise_for_status()
        output_lines = resp.text.strip().split("\n")

    # Reset Pinecone index if requested (dimension changed)
    if reset_index:
        console.print("  Deleting old Pinecone index...")
        delete_index()
        console.print("  Recreating index with new dimensions...")
        time.sleep(5)  # Give Pinecone time to process deletion

    # Parse output and build vectors
    all_vectors: list[tuple[str, list[float], dict]] = []
    for line in output_lines:
        result = json.loads(line)
        custom_id = result["custom_id"]
        response = result.get("response", {})
        body = response.get("body", {})

        if "data" not in body:
            console.print(f"  [yellow]Warning: no data in {custom_id}[/]")
            continue

        chunk_indices = request_map[custom_id]
        embeddings = [item["embedding"] for item in body["data"]]

        for idx, embedding in zip(chunk_indices, embeddings):
            chunk = chunks[idx]
            meta = chunk.metadata.to_pinecone_metadata()
            all_vectors.append((chunk.chunk_id, embedding, meta))

    console.print(f"  Parsed {len(all_vectors)} embeddings")

    store_start = time.time()
    stored = upsert_vectors(all_vectors)
    store_time = time.time() - store_start
    console.print(f"  Upserted [green]{stored}[/] vectors in {store_time:.1f}s")

    total_time = time.time() - start_time
    stats = get_index_stats()

    console.print(f"\n[bold green]Batch ingestion complete![/] {total_time:.1f}s total")
    console.print(f"  Index stats: {stats.get('total_vector_count', 'N/A')} total vectors")

    return {
        "chunks": len(chunks),
        "vectors": stored,
        "chunk_time": chunk_time,
        "embed_time": embed_time,
        "store_time": store_time,
        "total_time": total_time,
    }
