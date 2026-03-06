"""Tests for retrieval pipeline."""

from unittest.mock import patch, MagicMock, call

import pytest


MOCK_RESULTS = [
    {
        "id": "fortran:dgesv:dgesv",
        "score": 0.85,
        "metadata": {
            "unit_name": "DGESV",
            "unit_type": "subroutine",
            "file_path": "dgesv.f",
            "language": "fortran",
            "purpose": "Computes the solution to a system of linear equations",
            "start_line": 1,
            "end_line": 178,
            "parameters": ["N", "NRHS", "A", "LDA", "IPIV", "B", "LDB", "INFO"],
            "calls": ["DGETRF", "DGETRS", "XERBLA"],
            "routine_role": "driver",
        },
    },
    {
        "id": "fortran:dgetrf:dgetrf",
        "score": 0.72,
        "metadata": {
            "unit_name": "DGETRF",
            "unit_type": "subroutine",
            "file_path": "dgetrf.f",
            "language": "fortran",
            "purpose": "Computes an LU factorization",
            "start_line": 1,
            "end_line": 226,
            "parameters": ["M", "N", "A", "LDA", "IPIV", "INFO"],
            "calls": ["DGEMM", "DGETRF2", "DLASWP", "DTRSM", "XERBLA"],
            "routine_role": "computational",
        },
    },
]


# --- Tests for _base_routine_name ---

def test_base_routine_name_strips_precision():
    from legacylens.rag.retrieve import _base_routine_name

    assert _base_routine_name("DGESV") == "GESV"
    assert _base_routine_name("SGESV") == "GESV"
    assert _base_routine_name("CGESV") == "GESV"
    assert _base_routine_name("ZGESV") == "GESV"


def test_base_routine_name_preserves_non_lapack():
    from legacylens.rag.retrieve import _base_routine_name

    assert _base_routine_name("XERBLA") == "XERBLA"  # X is not a precision prefix
    assert _base_routine_name("") == ""


def test_base_routine_name_short_names():
    from legacylens.rag.retrieve import _base_routine_name

    # Single character or names where second char is not alpha
    assert _base_routine_name("D") == "D"
    assert _base_routine_name("D1") == "D1"


# --- Tests for _diversify_results ---

def test_diversify_removes_precision_variants():
    from legacylens.rag.retrieve import _diversify_results

    results = [
        {"id": "1", "score": 0.65, "metadata": {"unit_name": "SLAQZ0"}},
        {"id": "2", "score": 0.64, "metadata": {"unit_name": "DLAQZ0"}},
        {"id": "3", "score": 0.63, "metadata": {"unit_name": "ZLAQZ0"}},
        {"id": "4", "score": 0.60, "metadata": {"unit_name": "DGESV"}},
        {"id": "5", "score": 0.55, "metadata": {"unit_name": "SGESV"}},
    ]
    diversified = _diversify_results(results, top_k=5)

    # Should prefer D-prefix: DLAQZ0 over SLAQZ0, keep DGESV, drop SGESV
    names = [r["metadata"]["unit_name"] for r in diversified]
    assert names == ["DLAQZ0", "DGESV"]


def test_diversify_prefers_d_prefix():
    """When S-prefix appears first, D-prefix should replace it."""
    from legacylens.rag.retrieve import _diversify_results

    results = [
        {"id": "1", "score": 0.70, "metadata": {"unit_name": "SGESV"}},
        {"id": "2", "score": 0.65, "metadata": {"unit_name": "DGESV"}},
        {"id": "3", "score": 0.60, "metadata": {"unit_name": "DGEEV"}},
    ]
    diversified = _diversify_results(results, top_k=5)
    names = [r["metadata"]["unit_name"] for r in diversified]
    # DGESV should replace SGESV; DGEEV is a different base
    assert names == ["DGESV", "DGEEV"]


def test_diversify_respects_top_k():
    from legacylens.rag.retrieve import _diversify_results

    results = [
        {"id": "1", "score": 0.9, "metadata": {"unit_name": "DGESV"}},
        {"id": "2", "score": 0.8, "metadata": {"unit_name": "DGEEV"}},
        {"id": "3", "score": 0.7, "metadata": {"unit_name": "DSYEV"}},
        {"id": "4", "score": 0.6, "metadata": {"unit_name": "DGESVD"}},
    ]
    diversified = _diversify_results(results, top_k=2)
    assert len(diversified) == 2


def test_diversify_empty():
    from legacylens.rag.retrieve import _diversify_results

    assert _diversify_results([], top_k=5) == []


# --- Tests for _sort_by_role ---

def test_sort_by_role_orders_correctly():
    from legacylens.rag.retrieve import _sort_by_role

    results = [
        {"id": "1", "metadata": {"unit_name": "DLAQZ0", "routine_role": "auxiliary"}},
        {"id": "2", "metadata": {"unit_name": "DGESV", "routine_role": "driver"}},
        {"id": "3", "metadata": {"unit_name": "DGETRF", "routine_role": "computational"}},
        {"id": "4", "metadata": {"unit_name": "DGEMM", "routine_role": "blas"}},
    ]
    sorted_results = _sort_by_role(results)
    roles = [r["metadata"]["routine_role"] for r in sorted_results]
    assert roles == ["driver", "computational", "auxiliary", "blas"]


def test_sort_by_role_missing_role_goes_last():
    from legacylens.rag.retrieve import _sort_by_role

    results = [
        {"id": "1", "metadata": {"unit_name": "UNKNOWN"}},  # No routine_role
        {"id": "2", "metadata": {"unit_name": "DGESV", "routine_role": "driver"}},
    ]
    sorted_results = _sort_by_role(results)
    assert sorted_results[0]["id"] == "2"  # Driver first
    assert sorted_results[1]["id"] == "1"  # Unknown last


# --- Tests for _extract_fortran_entities ---

def test_extract_entities_parameter():
    from legacylens.rag.retrieve import _extract_fortran_entities

    result = _extract_fortran_entities("modify pivot array IPIV")
    assert "IPIV" in result["parameters"]
    assert result["routines"] == []


def test_extract_entities_routine():
    from legacylens.rag.retrieve import _extract_fortran_entities

    result = _extract_fortran_entities("routines that call DGETRF")
    assert "DGETRF" in result["routines"]
    assert result["parameters"] == []


def test_extract_entities_stopwords():
    from legacylens.rag.retrieve import _extract_fortran_entities

    result = _extract_fortran_entities("WHAT DOES THIS DO")
    assert result["parameters"] == []
    assert result["routines"] == []


def test_extract_entities_mixed():
    from legacylens.rag.retrieve import _extract_fortran_entities

    result = _extract_fortran_entities("Does DGESV use IPIV?")
    assert "DGESV" in result["routines"]
    assert "IPIV" in result["parameters"]


def test_extract_entities_known_utility():
    """XERBLA should be classified as a routine, not a parameter."""
    from legacylens.rag.retrieve import _extract_fortran_entities

    result = _extract_fortran_entities("routines that call XERBLA")
    assert "XERBLA" in result["routines"]
    assert "XERBLA" not in result["parameters"]


def test_extract_entities_ilaenv():
    """ILAENV should be classified as a routine via known utilities."""
    from legacylens.rag.retrieve import _extract_fortran_entities

    result = _extract_fortran_entities("How does ILAENV work?")
    assert "ILAENV" in result["routines"]
    assert "ILAENV" not in result["parameters"]


# --- Tests for _rerank_results ---

@patch("legacylens.rag.retrieve._get_voyage_client")
def test_rerank_results_reorders(mock_client_fn):
    """Reranker should reorder results based on relevance scores."""
    from legacylens.rag.retrieve import _rerank_results

    results = [
        {"id": "1", "score": 0.9, "metadata": {"unit_name": "DGESDD", "unit_type": "subroutine", "purpose": "SVD"}},
        {"id": "2", "score": 0.8, "metadata": {"unit_name": "DSYEV", "unit_type": "subroutine", "purpose": "eigenvalues"}},
        {"id": "3", "score": 0.7, "metadata": {"unit_name": "DGEEV", "unit_type": "subroutine", "purpose": "eigenvalues"}},
    ]

    # Mock reranker to prefer eigenvalue routines
    mock_client = MagicMock()
    mock_client_fn.return_value = mock_client
    mock_rr0 = MagicMock(index=2)  # DGEEV first
    mock_rr1 = MagicMock(index=1)  # DSYEV second
    mock_rerank_result = MagicMock()
    mock_rerank_result.results = [mock_rr0, mock_rr1]
    mock_client.rerank.return_value = mock_rerank_result

    reranked = _rerank_results("eigenvalues", results, top_k=2)

    assert len(reranked) == 2
    assert reranked[0]["id"] == "3"  # DGEEV
    assert reranked[1]["id"] == "2"  # DSYEV
    mock_client.rerank.assert_called_once()


@patch("legacylens.rag.retrieve._get_voyage_client")
def test_rerank_results_fallback(mock_client_fn):
    """If reranking fails, original order should be preserved (truncated to top_k)."""
    from legacylens.rag.retrieve import _rerank_results

    results = [
        {"id": "1", "score": 0.9, "metadata": {}},
        {"id": "2", "score": 0.8, "metadata": {}},
        {"id": "3", "score": 0.7, "metadata": {}},
    ]

    mock_client = MagicMock()
    mock_client_fn.return_value = mock_client
    mock_client.rerank.side_effect = RuntimeError("API error")

    reranked = _rerank_results("test query", results, top_k=2)

    assert len(reranked) == 2
    assert reranked[0]["id"] == "1"
    assert reranked[1]["id"] == "2"


# --- Tests for tiered reranking ---

@patch("legacylens.rag.retrieve._rerank_results", side_effect=lambda q, r, k: list(reversed(r[:k])))
@patch("legacylens.rag.retrieve.query_vectors")
@patch("legacylens.rag.retrieve.embed_query", return_value=[0.1] * 1024)
def test_retrieve_entity_results_before_reranked(mock_embed, mock_query, mock_rerank):
    """Entity results should always appear before non-entity results, each tier reranked separately."""
    from legacylens.rag.retrieve import retrieve

    d_prefix_result = [
        {
            "id": "fortran:dgetrf:dgetrf",
            "score": 0.70,
            "metadata": {"unit_name": "DGETRF", "parameters": ["M", "N", "A", "LDA", "IPIV", "INFO"]},
        }
    ]
    all_precision_result = [
        {
            "id": "fortran:cgetrf:cgetrf",
            "score": 0.65,
            "metadata": {"unit_name": "CGETRF", "parameters": ["M", "N", "A", "LDA", "IPIV", "INFO"]},
        }
    ]
    driver_results = [
        {"id": "fortran:dgesv:dgesv", "score": 0.90, "metadata": {"unit_name": "DGESV", "routine_role": "driver"}},
    ]
    general_results = [
        {"id": "fortran:dsyev:dsyev", "score": 0.85, "metadata": {"unit_name": "DSYEV"}},
    ]

    # Entity pass (D-prefix + all-precision) + driver + semantic
    mock_query.side_effect = [d_prefix_result, all_precision_result, driver_results, general_results]

    results = retrieve("What functions modify IPIV?", top_k=5)

    # Entity result must be first — entity tier always precedes non-entity tier
    assert results[0]["id"] == "fortran:dgetrf:dgetrf"
    assert results[0].get("_entity_match") is True

    # Reranker is called twice: once for entity tier, once for non-entity tier
    assert mock_rerank.call_count == 2
    # First call: entity results
    entity_rerank_results = mock_rerank.call_args_list[0][0][1]
    assert all(r.get("_entity_match") for r in entity_rerank_results)
    # Second call: non-entity results
    non_entity_rerank_results = mock_rerank.call_args_list[1][0][1]
    assert all(not r.get("_entity_match") for r in non_entity_rerank_results)


@patch("legacylens.rag.retrieve._rerank_results", side_effect=lambda q, r, k: r[:k])
@patch("legacylens.rag.retrieve.query_vectors")
@patch("legacylens.rag.retrieve.embed_query", return_value=[0.1] * 1024)
def test_retrieve_entity_diversifies_precision_variants(mock_embed, mock_query, mock_rerank):
    """Entity results should be diversified to prefer D-prefix over C/Z variants."""
    from legacylens.rag.retrieve import retrieve

    d_prefix_results = [
        {"id": "3", "score": 0.60, "metadata": {"unit_name": "DGETRF", "parameters": ["IPIV"]}},
        {"id": "4", "score": 0.55, "metadata": {"unit_name": "DLAHEF_RK", "parameters": ["IPIV"]}},
        {"id": "5", "score": 0.50, "metadata": {"unit_name": "DLAHEF_ROOK", "parameters": ["IPIV"]}},
    ]
    all_precision_results = [
        {"id": "1", "score": 0.70, "metadata": {"unit_name": "CLAHEF_RK", "parameters": ["IPIV"]}},
        {"id": "2", "score": 0.65, "metadata": {"unit_name": "ZLAHEF_ROOK", "parameters": ["IPIV"]}},
    ]
    driver_results = []
    general_results = []

    # D-prefix query + all-precision query + driver + semantic
    mock_query.side_effect = [d_prefix_results, all_precision_results, driver_results, general_results]

    results = retrieve("What functions modify IPIV?", top_k=5)
    names = [r["metadata"]["unit_name"] for r in results]

    # D-prefix should replace C/Z variants via diversification
    assert "DLAHEF_RK" in names, "D-prefix DLAHEF_RK should replace CLAHEF_RK"
    assert "DLAHEF_ROOK" in names, "D-prefix DLAHEF_ROOK should replace ZLAHEF_ROOK"
    assert "CLAHEF_RK" not in names
    assert "ZLAHEF_ROOK" not in names


# --- Tests for score threshold ---

@patch("legacylens.rag.retrieve._rerank_results", side_effect=lambda q, r, k: r[:k])
@patch("legacylens.rag.retrieve.query_vectors")
@patch("legacylens.rag.retrieve.embed_query", return_value=[0.1] * 1024)
def test_retrieve_filters_low_scores(mock_embed, mock_query, mock_rerank):
    """Results below _MIN_SCORE_THRESHOLD should be filtered out."""
    from legacylens.rag.retrieve import retrieve

    driver_results = []
    general_results = [
        {"id": "1", "score": 0.60, "metadata": {"unit_name": "DGESV"}},
        {"id": "2", "score": 0.30, "metadata": {"unit_name": "DGEEV"}},  # Below threshold
    ]

    mock_query.side_effect = [driver_results, general_results]

    results = retrieve("how does lapack solve equations?", top_k=5)

    # Only result above threshold should survive
    assert len(results) == 1
    assert results[0]["id"] == "1"


@patch("legacylens.rag.retrieve._rerank_results", side_effect=lambda q, r, k: r[:k])
@patch("legacylens.rag.retrieve.query_vectors")
@patch("legacylens.rag.retrieve.embed_query", return_value=[0.1] * 1024)
def test_retrieve_entity_match_bypasses_threshold(mock_embed, mock_query, mock_rerank):
    """Entity-matched results should be kept even with low scores."""
    from legacylens.rag.retrieve import retrieve

    d_prefix_result = [
        {
            "id": "fortran:dgetrf:dgetrf",
            "score": 0.30,  # Below threshold, but entity-matched
            "metadata": {"unit_name": "DGETRF", "parameters": ["M", "N", "A", "LDA", "IPIV", "INFO"]},
        }
    ]
    all_precision_result = []
    driver_results = []
    general_results = [
        {"id": "fortran:dgesv:dgesv", "score": 0.60, "metadata": {"unit_name": "DGESV"}},
    ]

    # Entity pass (D-prefix + all-precision) + driver + semantic
    mock_query.side_effect = [d_prefix_result, all_precision_result, driver_results, general_results]

    results = retrieve("What functions modify IPIV?", top_k=5)

    # Entity result should survive despite low score
    ids = [r["id"] for r in results]
    assert "fortran:dgetrf:dgetrf" in ids


# --- Tests for retrieve() ---

@patch("legacylens.rag.retrieve._rerank_results", side_effect=lambda q, r, k: r[:k])
@patch("legacylens.rag.retrieve.query_vectors")
@patch("legacylens.rag.retrieve.embed_query", return_value=[0.1] * 1024)
def test_retrieve_two_pass(mock_embed, mock_query, mock_rerank):
    """retrieve() should do driver-filtered + unfiltered queries (no entities in lowercase)."""
    from legacylens.rag.retrieve import retrieve

    driver_results = [MOCK_RESULTS[0]]  # DGESV (driver)
    general_results = MOCK_RESULTS       # DGESV + DGETRF

    mock_query.side_effect = [driver_results, general_results]

    results = retrieve("How does LAPACK solve linear equations?", top_k=5)

    # Should have called query_vectors twice (no entities in lowercase query with LAPACK stopword)
    assert mock_query.call_count == 2

    # First call: driver-filtered
    first_call = mock_query.call_args_list[0]
    assert first_call.kwargs.get("filter") == {"routine_role": "driver"}

    # Second call: unfiltered with over-fetch
    second_call = mock_query.call_args_list[1]
    assert second_call.kwargs.get("top_k") == 15  # 5 * 3

    # Results should include both routines
    assert results[0]["id"] == "fortran:dgesv:dgesv"
    assert len(results) == 2


@patch("legacylens.rag.retrieve._rerank_results", side_effect=lambda q, r, k: r[:k])
@patch("legacylens.rag.retrieve.query_vectors")
@patch("legacylens.rag.retrieve.embed_query", return_value=[0.1] * 1024)
def test_retrieve_diversifies_precision_variants(mock_embed, mock_query, mock_rerank):
    """Precision variants should be deduplicated in results."""
    from legacylens.rag.retrieve import retrieve

    driver_results = []  # No drivers matched
    general_results = [
        {"id": "1", "score": 0.65, "metadata": {"unit_name": "SLAQZ0"}},
        {"id": "2", "score": 0.64, "metadata": {"unit_name": "DLAQZ0"}},
        {"id": "3", "score": 0.63, "metadata": {"unit_name": "CLAQZ0"}},
        {"id": "4", "score": 0.60, "metadata": {"unit_name": "DGESV"}},
        {"id": "5", "score": 0.55, "metadata": {"unit_name": "DGEEV"}},
    ]

    mock_query.side_effect = [driver_results, general_results]

    results = retrieve("eigenvalues", top_k=5)
    names = [r["metadata"]["unit_name"] for r in results]

    # Only one LAQZ0 variant should survive
    laqz_count = sum(1 for n in names if "LAQZ0" in n)
    assert laqz_count == 1


@patch("legacylens.rag.retrieve.query_vectors")
@patch("legacylens.rag.retrieve.embed_query", return_value=[0.1] * 1024)
def test_retrieve_pin_unit(mock_embed, mock_query):
    """pin_unit should do a filtered query first, then merge with semantic results."""
    from legacylens.rag.retrieve import retrieve

    pinned = [MOCK_RESULTS[0]]  # DGESV
    semantic = MOCK_RESULTS  # DGESV + DGETRF

    mock_query.side_effect = [pinned, semantic]

    results = retrieve("deps of DGESV", top_k=5, pin_unit="DGESV")

    # Should have called query_vectors twice: filtered + semantic
    assert mock_query.call_count == 2
    # First call should have filter
    first_call = mock_query.call_args_list[0]
    assert first_call.kwargs.get("filter") == {"unit_name": "DGESV"}
    # Results should have DGESV first, then DGETRF (deduped)
    assert results[0]["id"] == "fortran:dgesv:dgesv"
    assert len(results) == 2


@patch("legacylens.rag.retrieve._rerank_results", side_effect=lambda q, r, k: r[:k])
@patch("legacylens.rag.retrieve.query_vectors", return_value=[])
@patch("legacylens.rag.retrieve.embed_query", return_value=[0.1] * 1024)
def test_retrieve_empty_results(mock_embed, mock_query, mock_rerank):
    from legacylens.rag.retrieve import retrieve

    # Two-pass: both return empty
    mock_query.side_effect = [[], []]
    results = retrieve("nonexistent query")
    assert results == []


@patch("legacylens.rag.retrieve._rerank_results", side_effect=lambda q, r, k: r[:k])
@patch("legacylens.rag.retrieve.query_vectors")
@patch("legacylens.rag.retrieve.embed_query", return_value=[0.1] * 1024)
def test_retrieve_entity_filtered(mock_embed, mock_query, mock_rerank):
    """Entity-filtered pass should issue D-prefix + all-precision queries."""
    from legacylens.rag.retrieve import retrieve

    d_prefix_result = [
        {
            "id": "fortran:dgetrf:dgetrf",
            "score": 0.70,
            "metadata": {"unit_name": "DGETRF", "parameters": ["M", "N", "A", "LDA", "IPIV", "INFO"]},
        }
    ]
    all_precision_result = [MOCK_RESULTS[1]]  # DGETRF again (deduped)
    driver_results = [MOCK_RESULTS[0]]  # DGESV
    general_results = MOCK_RESULTS

    # Entity pass (D-prefix + all-precision) + driver + semantic
    mock_query.side_effect = [d_prefix_result, all_precision_result, driver_results, general_results]

    results = retrieve("What functions modify the pivot array IPIV?", top_k=5)

    # Should have called query_vectors 4 times: D-prefix + all-precision + driver + semantic
    assert mock_query.call_count == 4

    # First call: D-prefix filtered
    first_call = mock_query.call_args_list[0]
    assert first_call.kwargs.get("filter") == {"parameters": "IPIV", "precision": "double"}
    assert first_call.kwargs.get("top_k") == 5

    # Second call: all-precision
    second_call = mock_query.call_args_list[1]
    assert second_call.kwargs.get("filter") == {"parameters": "IPIV"}
    assert second_call.kwargs.get("top_k") == 5

    # Entity result should come first
    assert results[0]["id"] == "fortran:dgetrf:dgetrf"


@patch("legacylens.rag.retrieve._rerank_results", side_effect=lambda q, r, k: r[:k])
@patch("legacylens.rag.retrieve.query_vectors")
@patch("legacylens.rag.retrieve.embed_query", return_value=[0.1] * 1024)
def test_retrieve_routine_bidirectional(mock_embed, mock_query, mock_rerank):
    """Routine entity should issue unit_name + D-prefix callers + all callers queries."""
    from legacylens.rag.retrieve import retrieve

    # unit_name query returns the routine itself
    unit_name_result = [
        {
            "id": "fortran:dgetrf:dgetrf",
            "score": 0.90,
            "metadata": {"unit_name": "DGETRF", "unit_type": "subroutine", "purpose": "LU factorization"},
        }
    ]
    # D-prefix callers
    d_prefix_callers = [
        {
            "id": "fortran:dgesv:dgesv",
            "score": 0.80,
            "metadata": {"unit_name": "DGESV", "calls": ["DGETRF", "DGETRS", "XERBLA"]},
        }
    ]
    # All-precision callers
    all_callers = [
        {
            "id": "fortran:cgesv:cgesv",
            "score": 0.75,
            "metadata": {"unit_name": "CGESV", "calls": ["CGETRF", "CGETRS", "XERBLA"]},
        }
    ]
    driver_results = []
    general_results = []

    # Entity pass: unit_name + D-prefix callers + all callers, then driver + semantic
    mock_query.side_effect = [unit_name_result, d_prefix_callers, all_callers, driver_results, general_results]

    results = retrieve("what does DGETRF do?", top_k=5)

    # Should have called query_vectors 5 times: unit_name + D-callers + all-callers + driver + semantic
    assert mock_query.call_count == 5

    # First call: unit_name filter
    first_call = mock_query.call_args_list[0]
    assert first_call.kwargs.get("filter") == {"unit_name": "DGETRF"}
    assert first_call.kwargs.get("top_k") == 3

    # Second call: D-prefix callers
    second_call = mock_query.call_args_list[1]
    assert second_call.kwargs.get("filter") == {"calls": "DGETRF", "precision": "double"}
    assert second_call.kwargs.get("top_k") == 5

    # Third call: all-precision callers
    third_call = mock_query.call_args_list[2]
    assert third_call.kwargs.get("filter") == {"calls": "DGETRF"}
    assert third_call.kwargs.get("top_k") == 5

    # DGETRF should appear first (entity match)
    assert results[0]["id"] == "fortran:dgetrf:dgetrf"
    assert results[0].get("_entity_match") is True


@patch("legacylens.rag.retrieve._rerank_results", side_effect=lambda q, r, k: r[:k])
@patch("legacylens.rag.retrieve.query_vectors")
@patch("legacylens.rag.retrieve.embed_query", return_value=[0.1] * 1024)
def test_retrieve_no_entities_no_extra_calls(mock_embed, mock_query, mock_rerank):
    """Lowercase query with no Fortran identifiers should make exactly 2 query_vectors calls."""
    from legacylens.rag.retrieve import retrieve

    mock_query.side_effect = [[], []]

    retrieve("how does lapack solve linear equations?", top_k=5)

    # Only driver + semantic passes, no entity pass
    assert mock_query.call_count == 2


# --- Tests for _extract_python_entities ---

def test_extract_python_entities_snake_case():
    from legacylens.rag.retrieve import _extract_python_entities

    result = _extract_python_entities("how does load_config work?")
    assert "load_config" in result["functions"]


def test_extract_python_entities_camel_case():
    from legacylens.rag.retrieve import _extract_python_entities

    result = _extract_python_entities("what does MyClass do?")
    assert "MyClass" in result["classes"]


def test_extract_python_entities_dotted_path():
    from legacylens.rag.retrieve import _extract_python_entities

    result = _extract_python_entities("how is os.path.join used?")
    assert "os.path.join" in result["modules"]


def test_extract_python_entities_filters_stopwords():
    from legacylens.rag.retrieve import _extract_python_entities

    result = _extract_python_entities("what does this class method return?")
    assert "class" not in result["functions"]
    assert "method" not in result["functions"]
    assert "return" not in result["functions"]


def test_extract_python_entities_mixed():
    from legacylens.rag.retrieve import _extract_python_entities

    result = _extract_python_entities("does DatabaseManager use save_record?")
    assert "DatabaseManager" in result["classes"]
    assert "save_record" in result["functions"]


# --- Tests for language-aware diversification ---

def test_diversify_python_no_dedup():
    """Python chunks should not be deduplicated by SDCZ prefix."""
    from legacylens.rag.retrieve import _diversify_results

    results = [
        {"id": "1", "score": 0.9, "metadata": {"unit_name": "save_data", "language": "python"}},
        {"id": "2", "score": 0.8, "metadata": {"unit_name": "send_email", "language": "python"}},
        {"id": "3", "score": 0.7, "metadata": {"unit_name": "sort_items", "language": "python"}},
    ]
    diversified = _diversify_results(results, top_k=5)
    # All 3 should survive — no SDCZ dedup for Python
    assert len(diversified) == 3


def test_diversify_mixed_languages():
    """Fortran dedup should only apply to Fortran, not Python chunks."""
    from legacylens.rag.retrieve import _diversify_results

    results = [
        {"id": "1", "score": 0.9, "metadata": {"unit_name": "DGESV", "language": "fortran"}},
        {"id": "2", "score": 0.8, "metadata": {"unit_name": "SGESV", "language": "fortran"}},
        {"id": "3", "score": 0.7, "metadata": {"unit_name": "save_data", "language": "python"}},
    ]
    diversified = _diversify_results(results, top_k=5)
    names = [r["metadata"]["unit_name"] for r in diversified]
    # DGESV replaces SGESV (Fortran dedup), save_data stays
    assert "DGESV" in names
    assert "save_data" in names
    assert "SGESV" not in names


def test_format_results_no_crash(capsys):
    """format_results should not crash on empty or populated results."""
    from legacylens.rag.retrieve import format_results

    format_results([])
    format_results(MOCK_RESULTS)


@patch("legacylens.rag.source_reader.read_source_snippet", return_value="      SUBROUTINE DGESV( N, NRHS )")
def test_format_results_show_code(mock_snippet, capsys):
    """format_results with show_code=True renders source snippets."""
    from legacylens.rag.retrieve import format_results

    format_results(MOCK_RESULTS, show_code=True)
    mock_snippet.assert_called()
    # read_source_snippet should be called for each result with valid start/end lines
    assert mock_snippet.call_count >= 1


@patch("legacylens.rag.source_reader.read_source_snippet", return_value="")
def test_format_results_show_code_no_snippet(mock_snippet, capsys):
    """format_results with show_code=True gracefully handles missing source."""
    from legacylens.rag.retrieve import format_results

    # Should not crash even if snippet returns empty
    format_results(MOCK_RESULTS, show_code=True)


def test_format_results_show_code_false(capsys):
    """format_results with show_code=False does not attempt to read source."""
    from legacylens.rag.retrieve import format_results

    # Patch should NOT be called — we're passing show_code=False
    with patch("legacylens.rag.source_reader.read_source_snippet") as mock_snippet:
        format_results(MOCK_RESULTS, show_code=False)
        mock_snippet.assert_not_called()


# --- Tests for classify_module_tier ---

def test_classify_module_tier():
    from legacylens.chunkers.base import classify_module_tier

    assert classify_module_tier("v1/fields.py") == "legacy"
    assert classify_module_tier("pydantic/v1/validators.py") == "legacy"
    assert classify_module_tier("deprecated/old_api.py") == "deprecated"
    assert classify_module_tier("src/deprecated/compat.py") == "deprecated"
    assert classify_module_tier("_internal/core.py") == "internal"
    assert classify_module_tier("pydantic/_internal/fields.py") == "internal"
    assert classify_module_tier("main.py") == "current"
    assert classify_module_tier("src/models/user.py") == "current"
    # Windows path separators
    assert classify_module_tier("pydantic\\v1\\fields.py") == "legacy"


# --- Tests for tier-aware diversification ---

def test_diversify_python_tier_dedup():
    """Same-named Python unit across tiers should collapse to current."""
    from legacylens.rag.retrieve import _diversify_results

    results = [
        {"id": "1", "score": 0.9, "metadata": {"unit_name": "FieldInfo", "language": "python", "module_tier": "legacy"}},
        {"id": "2", "score": 0.8, "metadata": {"unit_name": "FieldInfo", "language": "python", "module_tier": "current"}},
        {"id": "3", "score": 0.7, "metadata": {"unit_name": "BaseModel", "language": "python", "module_tier": "current"}},
    ]
    diversified = _diversify_results(results, top_k=5)
    names = [r["metadata"]["unit_name"] for r in diversified]
    assert len(diversified) == 2
    # Current FieldInfo should win over legacy
    field_info = [r for r in diversified if r["metadata"]["unit_name"] == "FieldInfo"][0]
    assert field_info["metadata"]["module_tier"] == "current"
    assert "BaseModel" in names


def test_diversify_python_unique_legacy_survives():
    """Legacy unit with no current counterpart should survive."""
    from legacylens.rag.retrieve import _diversify_results

    results = [
        {"id": "1", "score": 0.9, "metadata": {"unit_name": "FieldInfo", "language": "python", "module_tier": "current"}},
        {"id": "2", "score": 0.8, "metadata": {"unit_name": "V1Validator", "language": "python", "module_tier": "legacy"}},
    ]
    diversified = _diversify_results(results, top_k=5)
    names = [r["metadata"]["unit_name"] for r in diversified]
    assert "FieldInfo" in names
    assert "V1Validator" in names


# --- Tests for Python two-pass retrieval ---

@patch("legacylens.rag.retrieve._rerank_results", side_effect=lambda q, r, k: r[:k])
@patch("legacylens.rag.retrieve.query_vectors", return_value=[])
@patch("legacylens.rag.retrieve.embed_query", return_value=[0.1] * 1024)
def test_retrieve_python_two_pass(mock_embed, mock_query, mock_rerank):
    """Python strategy should include a current-first filtered query."""
    from legacylens.rag.retrieve import retrieve

    # Use return_value=[] for all calls; we just verify the filter patterns
    retrieve("How does it work?", top_k=5, languages=["python"])

    # Find the call with module_tier filter (current-first pass)
    tier_filter_calls = [
        c for c in mock_query.call_args_list
        if c.kwargs.get("filter") == {"module_tier": {"$ne": "legacy"}}
    ]
    assert len(tier_filter_calls) == 1, "Should have one current-first filtered query"

    # Find the unfiltered call (general pass with over-fetch)
    unfiltered_calls = [
        c for c in mock_query.call_args_list
        if "filter" not in c.kwargs and c.kwargs.get("top_k") == 15
    ]
    assert len(unfiltered_calls) == 1, "Should have one unfiltered over-fetch query"

    # Entity queries should also have the tier filter
    entity_calls = [
        c for c in mock_query.call_args_list
        if c.kwargs.get("filter") and "module_tier" in c.kwargs["filter"]
        and c.kwargs.get("top_k") != 15  # exclude the current-first pass
    ]
    # All entity queries should include the tier filter
    assert len(entity_calls) >= 1, "Entity queries should include module_tier filter"


# --- Tests for _query_wants_legacy ---

def test_query_wants_legacy_true():
    from legacylens.rag.retrieve import _query_wants_legacy

    assert _query_wants_legacy("How do I migrate v1 validators to v2?")
    assert _query_wants_legacy("What does the legacy compat layer do?")
    assert _query_wants_legacy("How to upgrade from v1?")
    assert _query_wants_legacy("differences in the old api")


def test_query_wants_legacy_false():
    from legacylens.rag.retrieve import _query_wants_legacy

    assert not _query_wants_legacy("How does Pydantic validate field types?")
    assert not _query_wants_legacy("What does BaseModel.__init__ do?")


@patch("legacylens.rag.retrieve._rerank_results", side_effect=lambda q, r, k: r[:k])
@patch("legacylens.rag.retrieve.query_vectors", return_value=[])
@patch("legacylens.rag.retrieve.embed_query", return_value=[0.1] * 1024)
def test_retrieve_python_legacy_query_skips_filter(mock_embed, mock_query, mock_rerank):
    """When query mentions v1/legacy, tier filter should be skipped."""
    from legacylens.rag.retrieve import retrieve

    retrieve("How do I migrate from v1?", top_k=5, languages=["python"])

    # No call should have the module_tier filter
    tier_filter_calls = [
        c for c in mock_query.call_args_list
        if c.kwargs.get("filter") and "module_tier" in c.kwargs.get("filter", {})
    ]
    assert len(tier_filter_calls) == 0, "Legacy-intent query should skip tier filter"


# --- Tests for base_class filtered queries ---

@patch("legacylens.rag.retrieve._rerank_results", side_effect=lambda q, r, k: r[:k])
@patch("legacylens.rag.retrieve.query_vectors", return_value=[])
@patch("legacylens.rag.retrieve.embed_query", return_value=[0.1] * 1024)
def test_retrieve_python_base_class_filter(mock_embed, mock_query, mock_rerank):
    """When a CamelCase class name is found, should also query by base_classes."""
    from legacylens.rag.retrieve import retrieve

    retrieve("How does BaseModel validate?", top_k=5, languages=["python"])

    # Should have a query filtering by base_classes=BaseModel
    base_class_calls = [
        c for c in mock_query.call_args_list
        if c.kwargs.get("filter") and "base_classes" in c.kwargs.get("filter", {})
    ]
    assert len(base_class_calls) >= 1, "Should query by base_classes for CamelCase entity"
    assert base_class_calls[0].kwargs["filter"]["base_classes"] == "BaseModel"


# --- Tests for decorator-aware queries ---

@patch("legacylens.rag.retrieve._rerank_results", side_effect=lambda q, r, k: r[:k])
@patch("legacylens.rag.retrieve.query_vectors", return_value=[])
@patch("legacylens.rag.retrieve.embed_query", return_value=[0.1] * 1024)
def test_retrieve_python_decorator_concept(mock_embed, mock_query, mock_rerank):
    """Query containing 'validate' without identifiers should trigger decorator queries."""
    from legacylens.rag.retrieve import retrieve

    retrieve("how do i validate field types?", top_k=5, languages=["python"])

    # Should have decorator-filtered queries for validation decorators
    decorator_calls = [
        c for c in mock_query.call_args_list
        if c.kwargs.get("filter") and "decorators" in c.kwargs.get("filter", {})
    ]
    assert len(decorator_calls) >= 1, "Should query by decorators for concept keyword"
    decorator_names = [c.kwargs["filter"]["decorators"] for c in decorator_calls]
    assert "field_validator" in decorator_names or "validator" in decorator_names


# --- Tests for enhanced rerank document ---

@patch("legacylens.rag.retrieve._get_voyage_client")
def test_rerank_includes_base_classes(mock_client_fn):
    """Rerank document should include base_classes and decorators."""
    from legacylens.rag.retrieve import _rerank_results

    results = [
        {
            "id": "1",
            "score": 0.9,
            "metadata": {
                "unit_name": "MyModel",
                "unit_type": "class",
                "purpose": "A model",
                "base_classes": ["BaseModel"],
                "decorators": ["dataclass"],
            },
        },
    ]

    mock_client = MagicMock()
    mock_client_fn.return_value = mock_client
    mock_rr = MagicMock(index=0)
    mock_rerank_result = MagicMock()
    mock_rerank_result.results = [mock_rr]
    mock_client.rerank.return_value = mock_rerank_result

    _rerank_results("model", results, top_k=1)

    # Check that the document passed to rerank includes structural info
    call_args = mock_client.rerank.call_args
    documents = call_args[0][1]
    assert "inherits BaseModel" in documents[0]
    assert "@dataclass" in documents[0]


# --- Tests for keyword match threshold bypass ---

@patch("legacylens.rag.retrieve._rerank_results", side_effect=lambda q, r, k: r[:k])
@patch("legacylens.rag.retrieve.query_vectors")
@patch("legacylens.rag.retrieve.embed_query", return_value=[0.1] * 1024)
def test_keyword_match_bypasses_threshold(mock_embed, mock_query, mock_rerank):
    """Keyword-matched results should survive score threshold filtering."""
    from legacylens.rag.retrieve import retrieve_with_metrics

    driver_results = []
    general_results = [
        {"id": "1", "score": 0.60, "metadata": {"unit_name": "DGESV"}},
        {"id": "2", "score": 0.30, "metadata": {"unit_name": "DGEEV"}, "_keyword_match": True},
    ]

    mock_query.side_effect = [driver_results, general_results]

    result = retrieve_with_metrics("eigenvalues", top_k=5)

    # keyword_match result should survive despite low score
    ids = [r["id"] for r in result.results]
    assert "2" in ids
