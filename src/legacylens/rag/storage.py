"""Pinecone vector storage."""

from __future__ import annotations

from pinecone import Pinecone, ServerlessSpec

from legacylens.config import PINECONE_API_KEY, PINECONE_INDEX_NAME

_client: Pinecone | None = None
_index = None

DIMENSION = 1024  # Voyage Code 3


def _get_client() -> Pinecone:
    global _client
    if _client is None:
        _client = Pinecone(api_key=PINECONE_API_KEY())
    return _client


def get_index():
    """Get or create the Pinecone index."""
    global _index
    if _index is not None:
        return _index

    client = _get_client()
    index_name = PINECONE_INDEX_NAME()

    # Create index if it doesn't exist
    existing = [idx.name for idx in client.list_indexes()]
    if index_name not in existing:
        client.create_index(
            name=index_name,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    _index = client.Index(index_name)
    return _index


def upsert_vectors(
    vectors: list[tuple[str, list[float], dict]],
    namespace: str | None = None,
) -> int:
    """Upsert vectors to Pinecone.

    Args:
        vectors: List of (id, embedding, metadata) tuples.
        namespace: Pinecone namespace for source isolation.

    Returns:
        Number of vectors upserted.
    """
    index = get_index()
    batch_size = 100
    total = 0
    kwargs = {}
    if namespace:
        kwargs["namespace"] = namespace
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        index.upsert(vectors=batch, **kwargs)
        total += len(batch)
    return total


def query_vectors(
    embedding: list[float],
    top_k: int = 5,
    filter: dict | None = None,
    namespace: str | None = None,
) -> list[dict]:
    """Query Pinecone for similar vectors.

    Returns:
        List of matches with id, score, and metadata.
    """
    index = get_index()
    kwargs = {
        "vector": embedding,
        "top_k": top_k,
        "include_metadata": True,
    }
    if filter:
        kwargs["filter"] = filter
    if namespace:
        kwargs["namespace"] = namespace
    results = index.query(**kwargs)
    return [
        {
            "id": match.id,
            "score": match.score,
            "metadata": match.metadata,
        }
        for match in results.matches
    ]


def update_metadata(updates: list[tuple[str, dict]], batch_size: int = 100) -> int:
    """Update metadata on existing vectors without re-embedding.

    Args:
        updates: List of (vector_id, metadata_dict) tuples.
        batch_size: Number of updates per batch.

    Returns:
        Number of vectors updated.
    """
    index = get_index()
    total = 0
    for i in range(0, len(updates), batch_size):
        batch = updates[i : i + batch_size]
        for vector_id, metadata in batch:
            index.update(id=vector_id, set_metadata=metadata)
            total += 1
    return total


def list_vectors(prefix: str = "", limit: int = 100) -> list[str]:
    """List vector IDs in the index using pagination.

    Args:
        prefix: Optional ID prefix filter.
        limit: Max number of IDs to return.

    Returns:
        List of vector IDs.
    """
    index = get_index()
    ids: list[str] = []
    kwargs = {}
    if prefix:
        kwargs["prefix"] = prefix
    for id_list in index.list(**kwargs):
        ids.extend(id_list)
        if len(ids) >= limit:
            break
    return ids[:limit]


def fetch_vectors(ids: list[str], namespace: str | None = None) -> dict:
    """Fetch vectors by ID.

    Args:
        ids: List of vector IDs to fetch.
        namespace: Pinecone namespace.

    Returns:
        Dict of id -> vector data (with metadata).
    """
    index = get_index()
    kwargs = {"ids": ids}
    if namespace:
        kwargs["namespace"] = namespace
    result = index.fetch(**kwargs)
    return result.vectors


def delete_namespace(namespace: str) -> bool:
    """Delete all vectors in a namespace. Returns True if completed."""
    index = get_index()
    index.delete(delete_all=True, namespace=namespace)
    return True


def delete_index() -> bool:
    """Delete the Pinecone index. Returns True if deleted."""
    global _index
    client = _get_client()
    index_name = PINECONE_INDEX_NAME()
    existing = [idx.name for idx in client.list_indexes()]
    if index_name in existing:
        client.delete_index(index_name)
        _index = None
        return True
    return False


def get_index_stats() -> dict:
    """Get index statistics."""
    index = get_index()
    return index.describe_index_stats().to_dict()
