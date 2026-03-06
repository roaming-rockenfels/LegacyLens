"""Tests for source management."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from legacylens.sources import (
    SourceInfo,
    _slugify,
    scan_directory,
    detect_languages,
    register_source,
    list_sources,
    get_source,
    remove_source,
    get_source_languages,
)


# --- _slugify ---


def test_slugify_basic():
    assert _slugify("my-project") == "my-project"
    assert _slugify("MyProject") == "myproject"
    assert _slugify("my project!") == "my-project"
    assert _slugify("foo/bar/baz") == "foo-bar-baz"


def test_slugify_empty():
    assert _slugify("") == "default"
    assert _slugify("!!!") == "default"


# --- scan_directory ---


def test_scan_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "main.py").write_text("print('hi')")
        (Path(tmpdir) / "utils.py").write_text("x = 1")
        (Path(tmpdir) / "readme.md").write_text("# Readme")
        sub = Path(tmpdir) / "sub"
        sub.mkdir()
        (sub / "helper.py").write_text("pass")

        result = scan_directory(tmpdir)
        assert result.get(".py") == 3
        assert result.get(".md") == 1


def test_scan_directory_with_filter():
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "main.py").write_text("print('hi')")
        (Path(tmpdir) / "readme.md").write_text("# Readme")

        result = scan_directory(tmpdir, known_extensions=[".py"])
        assert result.get(".py") == 1
        assert ".md" not in result


# --- detect_languages ---


def test_detect_languages():
    assert detect_languages({".py": 10, ".pyi": 2}) == ["python"]
    assert detect_languages({".f": 5, ".f90": 3}) == ["fortran"]
    assert detect_languages({".py": 10, ".f": 5}) == ["fortran", "python"]
    assert detect_languages({".txt": 1}) == []


# --- Source registry (with temp file) ---


@pytest.fixture
def temp_sources_file(tmp_path):
    sources_file = tmp_path / "sources.json"
    with patch("legacylens.sources._SOURCES_FILE", sources_file), \
         patch("legacylens.sources._SOURCES_DIR", tmp_path):
        yield sources_file


def test_register_and_get(temp_sources_file):
    info = register_source("test-project", "/tmp/test", ["python"], {".py": 5}, 42)
    assert info.name == "test-project"
    assert info.namespace == "test-project"
    assert info.languages == ["python"]
    assert info.chunk_count == 42

    retrieved = get_source("test-project")
    assert retrieved is not None
    assert retrieved.name == "test-project"
    assert retrieved.chunk_count == 42


def test_list_sources(temp_sources_file):
    register_source("proj1", "/tmp/p1", ["python"], {".py": 5}, 10)
    register_source("proj2", "/tmp/p2", ["fortran"], {".f": 3}, 20)

    sources = list_sources()
    assert len(sources) == 2
    names = {s.name for s in sources}
    assert names == {"proj1", "proj2"}


def test_remove_source(temp_sources_file):
    register_source("to-remove", "/tmp/rm", ["python"], {".py": 1}, 5)
    assert get_source("to-remove") is not None

    result = remove_source("to-remove")
    assert result is True
    assert get_source("to-remove") is None


def test_remove_nonexistent(temp_sources_file):
    assert remove_source("nonexistent") is False


def test_get_source_languages(temp_sources_file):
    register_source("lang-test", "/tmp/lt", ["python", "fortran"], {".py": 5, ".f": 3}, 30)
    langs = get_source_languages("lang-test")
    assert langs == ["python", "fortran"]


def test_get_source_languages_missing(temp_sources_file):
    assert get_source_languages("missing") == []
