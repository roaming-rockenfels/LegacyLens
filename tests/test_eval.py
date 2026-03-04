"""Tests for retrieval evaluation library (TDD RED phase)."""

import json
import tempfile
from pathlib import Path

import pytest


# --- Tests for precision_at_k ---


def test_precision_at_k():
    from legacylens.rag.eval import precision_at_k

    # 2 hits out of k=5 → 0.4
    assert precision_at_k({"A", "B"}, ["A", "B", "C", "D", "E"], k=5) == pytest.approx(0.4)
    # 0 hits
    assert precision_at_k({"X"}, ["A", "B", "C"], k=3) == pytest.approx(0.0)
    # All hits
    assert precision_at_k({"A", "B", "C"}, ["A", "B", "C"], k=3) == pytest.approx(1.0)


# --- Tests for recall_at_k ---


def test_recall_at_k():
    from legacylens.rag.eval import recall_at_k

    # 1 of 2 expected found → 0.5
    assert recall_at_k({"A", "B"}, ["A", "C", "D"], k=3) == pytest.approx(0.5)
    # All expected found
    assert recall_at_k({"A"}, ["A", "B", "C"], k=3) == pytest.approx(1.0)
    # None found
    assert recall_at_k({"X", "Y"}, ["A", "B", "C"], k=3) == pytest.approx(0.0)


# --- Tests for MRR ---


def test_mrr_first_position():
    from legacylens.rag.eval import mrr

    assert mrr({"A"}, ["A", "B", "C"]) == pytest.approx(1.0)


def test_mrr_second_position():
    from legacylens.rag.eval import mrr

    assert mrr({"A"}, ["B", "A", "C"]) == pytest.approx(0.5)


def test_mrr_not_found():
    from legacylens.rag.eval import mrr

    assert mrr({"X"}, ["A", "B", "C"]) == pytest.approx(0.0)


# --- Tests for hit_rate ---


def test_hit_rate():
    from legacylens.rag.eval import hit_rate

    assert hit_rate({"A"}, ["A", "B", "C"], k=3) == pytest.approx(1.0)
    assert hit_rate({"X"}, ["A", "B", "C"], k=3) == pytest.approx(0.0)


# --- Tests for golden set loading ---


def test_load_golden_set_valid():
    from legacylens.rag.eval import load_golden_set

    data = [
        {"query": "What does DGESV do?", "expected_units": ["DGESV"]},
        {"query": "Who calls DGETRF?", "expected_units": ["DGESV", "DGETRF"], "top_k": 10},
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        f.flush()
        loaded = load_golden_set(f.name)

    assert len(loaded) == 2
    assert loaded[0]["query"] == "What does DGESV do?"
    assert loaded[0]["top_k"] == 5  # default
    assert loaded[1]["top_k"] == 10  # explicit


def test_load_golden_set_validates_schema():
    from legacylens.rag.eval import load_golden_set

    data = [{"expected_units": ["DGESV"]}]  # missing 'query'
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        f.flush()
        with pytest.raises(ValueError, match="query"):
            load_golden_set(f.name)


# --- Tests for aggregate_metrics ---


def test_aggregate_metrics():
    from legacylens.rag.eval import aggregate_metrics

    per_query = [
        {"precision": 0.4, "recall": 1.0, "mrr": 1.0, "hit_rate": 1.0},
        {"precision": 0.2, "recall": 0.5, "mrr": 0.5, "hit_rate": 1.0},
    ]
    agg = aggregate_metrics(per_query)
    assert agg["precision"] == pytest.approx(0.3)
    assert agg["recall"] == pytest.approx(0.75)
    assert agg["mrr"] == pytest.approx(0.75)
    assert agg["hit_rate"] == pytest.approx(1.0)


# --- Tests for format_report ---


def test_format_report_contains_headers():
    from legacylens.rag.eval import format_report

    per_query = [
        {"query": "test?", "precision": 0.4, "recall": 1.0, "mrr": 1.0, "hit_rate": 1.0},
    ]
    agg = {"precision": 0.4, "recall": 1.0, "mrr": 1.0, "hit_rate": 1.0}
    report = format_report(per_query, agg, threshold=0.7)
    assert "Query" in report
    assert "Recall" in report
    assert "MRR" in report


def test_format_report_pass_fail():
    from legacylens.rag.eval import format_report

    passing = [{"query": "q1", "precision": 0.4, "recall": 0.8, "mrr": 1.0, "hit_rate": 1.0}]
    agg_pass = {"precision": 0.4, "recall": 0.8, "mrr": 1.0, "hit_rate": 1.0}
    assert "PASS" in format_report(passing, agg_pass, threshold=0.7)

    failing = [{"query": "q1", "precision": 0.2, "recall": 0.5, "mrr": 0.5, "hit_rate": 1.0}]
    agg_fail = {"precision": 0.2, "recall": 0.5, "mrr": 0.5, "hit_rate": 1.0}
    assert "FAIL" in format_report(failing, agg_fail, threshold=0.7)
