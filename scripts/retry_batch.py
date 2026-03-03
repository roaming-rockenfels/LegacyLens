"""Retry failed batch embeddings and upsert to Pinecone."""

import json
import pickle
import time

import httpx

from legacylens.config import VOYAGE_API_KEY
from legacylens.rag.embeddings import MODEL
from legacylens.rag.storage import upsert_vectors, get_index_stats

VOYAGE_BASE = "https://api.voyageai.com/v1"


def main():
    headers_auth = {"Authorization": f"Bearer {VOYAGE_API_KEY()}"}
    headers_json = {**headers_auth, "Content-Type": "application/json"}

    # Upload retry file
    with httpx.Client(timeout=60) as client:
        resp = client.post(
            f"{VOYAGE_BASE}/files",
            headers=headers_auth,
            files={"file": ("retry.jsonl", open("/tmp/legacylens_batch_retry.jsonl", "rb"), "application/jsonl")},
            data={"purpose": "batch"},
        )
        resp.raise_for_status()
        file_id = resp.json()["id"]
        print(f"Uploaded: {file_id}")

    # Create batch
    with httpx.Client(timeout=60) as client:
        resp = client.post(
            f"{VOYAGE_BASE}/batches",
            headers=headers_json,
            json={
                "endpoint": "/v1/embeddings",
                "completion_window": "12h",
                "request_params": {"model": MODEL, "input_type": "document"},
                "input_file_id": file_id,
            },
        )
        resp.raise_for_status()
        batch_id = resp.json()["id"]
        print(f"Batch created: {batch_id}")

    # Poll
    while True:
        time.sleep(15)
        with httpx.Client(timeout=30) as client:
            resp = client.get(f"{VOYAGE_BASE}/batches/{batch_id}", headers=headers_json)
            data = resp.json()
        status = data["status"]
        rc = data["request_counts"]
        print(f"  Status: {status} | Completed: {rc['completed']}/{rc['total']} | Failed: {rc['failed']}")
        if status in ("completed", "failed", "expired", "cancelled"):
            break

    if status != "completed":
        print(f"Batch {status}!")
        return

    # Download results
    output_file_id = data["output_file_id"]
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        resp = client.get(f"{VOYAGE_BASE}/files/{output_file_id}/content", headers=headers_auth)
        resp.raise_for_status()
        output_lines = resp.text.strip().split("\n")
    print(f"Downloaded {len(output_lines)} result lines")

    # Load state
    with open("/tmp/legacylens_retry_state.pkl", "rb") as f:
        state = pickle.load(f)
    retry_map = state["retry_map"]
    missing_chunks = state["missing_chunks"]

    # Parse and upsert
    all_vectors = []
    for line in output_lines:
        result = json.loads(line)
        custom_id = result["custom_id"]
        response = result.get("response", {})
        body = response.get("body", {})
        if "data" not in body:
            continue
        chunk_indices = retry_map[custom_id]
        embeddings = [item["embedding"] for item in body["data"]]
        for idx, embedding in zip(chunk_indices, embeddings):
            chunk = missing_chunks[idx]
            meta = chunk.metadata.to_pinecone_metadata()
            all_vectors.append((chunk.chunk_id, embedding, meta))

    print(f"Parsed {len(all_vectors)} vectors")
    stored = upsert_vectors(all_vectors)
    print(f"Upserted {stored} vectors")

    time.sleep(3)
    stats = get_index_stats()
    print(f"Total vectors: {stats.get('total_vector_count')}")


if __name__ == "__main__":
    main()
