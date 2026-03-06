"""Tests for the shared source reader utility."""

from pathlib import Path
from unittest.mock import patch

import pytest

from legacylens.rag.source_reader import resolve_source_path, read_source_snippet


@pytest.fixture()
def source_tree(tmp_path):
    """Create a minimal source tree with a direct file and a SRC/ fallback."""
    (tmp_path / "dgesv.f").write_text("line1\nline2\nline3\nline4\nline5\n")
    src_dir = tmp_path / "SRC"
    src_dir.mkdir()
    (src_dir / "dgetrf.f").write_text("src_line1\nsrc_line2\nsrc_line3\n")
    return tmp_path


# --- resolve_source_path ---

def test_resolve_direct(source_tree):
    result = resolve_source_path("dgesv.f", base_dir=source_tree)
    assert result is not None
    assert result.name == "dgesv.f"


def test_resolve_src_fallback(source_tree):
    result = resolve_source_path("dgetrf.f", base_dir=source_tree)
    assert result is not None
    assert "SRC" in str(result)


def test_resolve_not_found(source_tree):
    result = resolve_source_path("nonexistent.f", base_dir=source_tree)
    assert result is None


def test_resolve_missing_config():
    """When no base_dir and CODEBASE_DATA_DIR is missing, returns None."""
    with patch("legacylens.rag.source_reader.CODEBASE_DATA_DIR", side_effect=RuntimeError):
        result = resolve_source_path("dgesv.f")
    assert result is None


# --- read_source_snippet ---

def test_read_line_range(source_tree):
    snippet = read_source_snippet("dgesv.f", 2, 4, base_dir=source_tree)
    assert snippet == "line2\nline3\nline4"


def test_read_full_file(source_tree):
    snippet = read_source_snippet("dgesv.f", 1, 5, base_dir=source_tree)
    assert "line1" in snippet
    assert "line5" in snippet


def test_read_not_found(source_tree):
    snippet = read_source_snippet("nonexistent.f", 1, 10, base_dir=source_tree)
    assert snippet == ""


def test_read_truncation(source_tree):
    snippet = read_source_snippet("dgesv.f", 1, 5, base_dir=source_tree, max_chars=10)
    assert "[truncated]" in snippet


def test_read_src_fallback(source_tree):
    snippet = read_source_snippet("dgetrf.f", 1, 2, base_dir=source_tree)
    assert snippet == "src_line1\nsrc_line2"
