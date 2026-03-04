"""Tests for runtime retrieval metrics (TDD RED phase)."""

from unittest.mock import patch, MagicMock

import pytest


# --- Tests for dataclass constructors ---


def test_retrieval_metrics_defaults():
    """RetrievalMetrics should be constructable with all defaults."""
    from legacylens.rag.retrieve import RetrievalMetrics

    m = RetrievalMetrics()
    assert m.entity_matches == 0
    assert m.threshold_filtered == 0
    assert m.variants_collapsed == 0
    assert m.score_distribution == {"min": 0.0, "max": 0.0, "mean": 0.0}


def test_retrieval_result_structure():
    """RetrievalResult should have .results and .metrics attributes."""
    from legacylens.rag.retrieve import RetrievalResult, RetrievalMetrics

    r = RetrievalResult()
    assert r.results == []
    assert isinstance(r.metrics, RetrievalMetrics)


# --- Tests for retrieve_with_metrics() ---


@patch("legacylens.rag.retrieve._rerank_results", side_effect=lambda q, r, k: r[:k])
@patch("legacylens.rag.retrieve.query_vectors")
@patch("legacylens.rag.retrieve.embed_query", return_value=[0.1] * 1024)
def test_retrieve_with_metrics_returns_result(mock_embed, mock_query, mock_rerank):
    """retrieve_with_metrics() should return a RetrievalResult."""
    from legacylens.rag.retrieve import retrieve_with_metrics, RetrievalResult

    mock_query.side_effect = [
        [],  # driver
        [{"id": "1", "score": 0.80, "metadata": {"unit_name": "DGESV"}}],  # general
    ]

    result = retrieve_with_metrics("how does lapack solve equations?", top_k=5)
    assert isinstance(result, RetrievalResult)
    assert isinstance(result.results, list)
    assert len(result.results) == 1
    assert result.results[0]["id"] == "1"


@patch("legacylens.rag.retrieve._rerank_results", side_effect=lambda q, r, k: r[:k])
@patch("legacylens.rag.retrieve.query_vectors")
@patch("legacylens.rag.retrieve.embed_query", return_value=[0.1] * 1024)
def test_retrieve_backward_compat(mock_embed, mock_query, mock_rerank):
    """retrieve() should still return list[dict] for backward compatibility."""
    from legacylens.rag.retrieve import retrieve

    mock_query.side_effect = [
        [],  # driver
        [{"id": "1", "score": 0.80, "metadata": {"unit_name": "DGESV"}}],  # general
    ]

    results = retrieve("how does lapack solve equations?", top_k=5)
    assert isinstance(results, list)
    assert results[0]["id"] == "1"


# --- Tests for individual metric counters ---


@patch("legacylens.rag.retrieve._rerank_results", side_effect=lambda q, r, k: r[:k])
@patch("legacylens.rag.retrieve.query_vectors")
@patch("legacylens.rag.retrieve.embed_query", return_value=[0.1] * 1024)
def test_metrics_entity_match_count(mock_embed, mock_query, mock_rerank):
    """Entity-matched results should be counted in metrics.entity_matches."""
    from legacylens.rag.retrieve import retrieve_with_metrics

    entity_result = [
        {
            "id": "fortran:dgetrf:dgetrf",
            "score": 0.70,
            "metadata": {"unit_name": "DGETRF", "parameters": ["IPIV"]},
        }
    ]
    mock_query.side_effect = [
        entity_result,  # D-prefix param query
        [],             # all-precision param query
        [],             # driver
        [],             # general
    ]

    result = retrieve_with_metrics("What functions modify IPIV?", top_k=5)
    assert result.metrics.entity_matches == 1


@patch("legacylens.rag.retrieve._rerank_results", side_effect=lambda q, r, k: r[:k])
@patch("legacylens.rag.retrieve.query_vectors")
@patch("legacylens.rag.retrieve.embed_query", return_value=[0.1] * 1024)
def test_metrics_threshold_filtered(mock_embed, mock_query, mock_rerank):
    """Results below 0.45 that aren't entity-matched should be counted in threshold_filtered."""
    from legacylens.rag.retrieve import retrieve_with_metrics

    mock_query.side_effect = [
        [],  # driver
        [
            {"id": "1", "score": 0.80, "metadata": {"unit_name": "DGESV"}},
            {"id": "2", "score": 0.30, "metadata": {"unit_name": "DGEEV"}},  # below threshold
        ],
    ]

    result = retrieve_with_metrics("how does lapack solve equations?", top_k=5)
    assert result.metrics.threshold_filtered == 1
    assert len(result.results) == 1  # only the above-threshold result


@patch("legacylens.rag.retrieve._rerank_results", side_effect=lambda q, r, k: r[:k])
@patch("legacylens.rag.retrieve.query_vectors")
@patch("legacylens.rag.retrieve.embed_query", return_value=[0.1] * 1024)
def test_metrics_variants_collapsed(mock_embed, mock_query, mock_rerank):
    """Precision variants collapsed by final diversify should be counted."""
    from legacylens.rag.retrieve import retrieve_with_metrics

    mock_query.side_effect = [
        [],  # driver
        [
            {"id": "1", "score": 0.80, "metadata": {"unit_name": "SLAQZ0"}},
            {"id": "2", "score": 0.79, "metadata": {"unit_name": "DLAQZ0"}},
            {"id": "3", "score": 0.78, "metadata": {"unit_name": "CLAQZ0"}},
            {"id": "4", "score": 0.70, "metadata": {"unit_name": "DGESV"}},
        ],
    ]

    result = retrieve_with_metrics("eigenvalues", top_k=5)
    # 3 LAQZ0 variants → 1 kept (DLAQZ0) → 2 collapsed
    assert result.metrics.variants_collapsed >= 2


@patch("legacylens.rag.retrieve._rerank_results", side_effect=lambda q, r, k: r[:k])
@patch("legacylens.rag.retrieve.query_vectors")
@patch("legacylens.rag.retrieve.embed_query", return_value=[0.1] * 1024)
def test_metrics_score_distribution(mock_embed, mock_query, mock_rerank):
    """Score distribution should compute min, max, mean from final results."""
    from legacylens.rag.retrieve import retrieve_with_metrics

    mock_query.side_effect = [
        [],  # driver
        [
            {"id": "1", "score": 0.90, "metadata": {"unit_name": "DGESV"}},
            {"id": "2", "score": 0.60, "metadata": {"unit_name": "DGEEV"}},
        ],
    ]

    result = retrieve_with_metrics("solve equations", top_k=5)
    dist = result.metrics.score_distribution
    assert dist["min"] == pytest.approx(0.60)
    assert dist["max"] == pytest.approx(0.90)
    assert dist["mean"] == pytest.approx(0.75)
