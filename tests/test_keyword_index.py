"""Tests for the BM25 keyword index."""

from unittest.mock import patch

import pytest

from legacylens.chunkers.base import Chunk, ChunkMetadata
from legacylens.rag.keyword_index import KeywordIndex, _tokenize


# --- Tokenization ---

def test_tokenize_snake_case():
    tokens = _tokenize("validate_field_types")
    assert "validate" in tokens
    assert "field" in tokens
    assert "types" in tokens


def test_tokenize_camel_case():
    tokens = _tokenize("BaseModel")
    assert "base" in tokens
    assert "model" in tokens


def test_tokenize_mixed():
    tokens = _tokenize("field_validator BaseModel pydantic.fields")
    assert "field" in tokens
    assert "validator" in tokens
    assert "base" in tokens
    assert "model" in tokens
    assert "pydantic" in tokens
    assert "fields" in tokens


def test_tokenize_removes_stopwords():
    tokens = _tokenize("the class for this import")
    assert "the" not in tokens
    assert "class" not in tokens
    assert "import" not in tokens


def test_tokenize_short_tokens_removed():
    tokens = _tokenize("a b cd ef")
    assert "a" not in tokens
    assert "b" not in tokens
    assert "cd" in tokens
    assert "ef" in tokens


# --- Index build and search ---

def _make_chunk(name: str, purpose: str, calls: list[str] = None,
                base_classes: list[str] = None, decorators: list[str] = None) -> Chunk:
    meta = ChunkMetadata(
        file_path="test.py",
        start_line=1,
        end_line=10,
        unit_name=name,
        unit_type="class" if name[0].isupper() else "function",
        language="python",
        calls=calls or [],
        purpose=purpose,
        base_classes=base_classes or [],
        decorators=decorators or [],
    )
    return Chunk(content="pass", enriched_content="pass", metadata=meta)


def test_build_and_search():
    chunks = [
        _make_chunk("BaseModel", "Base class for all models", base_classes=["ABC"]),
        _make_chunk("validate_python", "Validate data against a model", calls=["run_validators"]),
        _make_chunk("FieldInfo", "Field information for validation", base_classes=["BaseModel"]),
    ]
    idx = KeywordIndex()
    idx.build(chunks)

    results = idx.search("validate model", top_k=3)
    assert len(results) >= 1
    # validate_python should score highly for "validate"
    ids = [cid for cid, _ in results]
    assert any("validate_python" in cid for cid in ids)


def test_search_empty_index():
    idx = KeywordIndex()
    results = idx.search("anything")
    assert results == []


def test_search_empty_query():
    chunks = [_make_chunk("foo", "a function")]
    idx = KeywordIndex()
    idx.build(chunks)
    results = idx.search("")
    assert results == []


def test_search_respects_top_k():
    chunks = [
        _make_chunk("alpha", "alpha function"),
        _make_chunk("beta", "beta function"),
        _make_chunk("gamma", "gamma function"),
    ]
    idx = KeywordIndex()
    idx.build(chunks)
    results = idx.search("function", top_k=2)
    assert len(results) <= 2


def test_search_decorator_match():
    chunks = [
        _make_chunk("check_name", "Check name field", decorators=["field_validator"]),
        _make_chunk("save_data", "Save data to disk"),
    ]
    idx = KeywordIndex()
    idx.build(chunks)

    results = idx.search("field_validator", top_k=3)
    ids = [cid for cid, _ in results]
    assert any("check_name" in cid for cid in ids)


def test_search_base_class_match():
    chunks = [
        _make_chunk("MyModel", "A custom model", base_classes=["BaseModel"]),
        _make_chunk("helper", "Helper function"),
    ]
    idx = KeywordIndex()
    idx.build(chunks)

    results = idx.search("BaseModel", top_k=3)
    ids = [cid for cid, _ in results]
    assert any("mymodel" in cid.lower() for cid in ids)


# --- Persistence ---

def test_save_and_load(tmp_path):
    chunks = [
        _make_chunk("validate_python", "Validate data"),
        _make_chunk("BaseModel", "Base model class"),
    ]
    idx = KeywordIndex()
    idx.build(chunks)

    with patch("legacylens.rag.keyword_index._INDEX_DIR", tmp_path):
        idx.save("test_source")

        idx2 = KeywordIndex()
        assert idx2.load("test_source")

        results = idx2.search("validate", top_k=3)
        assert len(results) >= 1


def test_load_nonexistent(tmp_path):
    idx = KeywordIndex()
    with patch("legacylens.rag.keyword_index._INDEX_DIR", tmp_path):
        assert not idx.load("nonexistent_source")
