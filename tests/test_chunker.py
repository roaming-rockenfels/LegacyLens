"""Tests for the Fortran chunker against real LAPACK files."""

import os
from pathlib import Path

import pytest

from legacylens.chunkers.base import ChunkerRegistry
from legacylens.chunkers.fortran import FortranChunker

LAPACK_DIR = Path(__file__).parent.parent / "data" / "lapack"
SRC_DIR = LAPACK_DIR / "SRC"
BLAS_DIR = LAPACK_DIR / "BLAS" / "SRC"


@pytest.fixture
def chunker():
    return FortranChunker()


@pytest.fixture
def registry(chunker):
    reg = ChunkerRegistry()
    reg.register(chunker)
    return reg


def _read_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


# ─── Registry tests ───


def test_registry_selects_fortran(registry):
    assert registry.get_chunker("foo.f") is not None
    assert registry.get_chunker("foo.f90") is not None
    assert registry.get_chunker("foo.py") is None


def test_registry_lists_extensions(registry):
    exts = registry.supported_extensions()
    assert ".f" in exts
    assert ".f90" in exts


# ─── Basic chunking tests ───


@pytest.mark.skipif(not SRC_DIR.exists(), reason="LAPACK not downloaded")
class TestDGESV:
    """Test against SRC/dgesv.f — small utility routine, ~177 lines."""

    def test_produces_one_chunk(self, chunker):
        content = _read_file(SRC_DIR / "dgesv.f")
        chunks = chunker.chunk_file("SRC/dgesv.f", content)
        assert len(chunks) == 1

    def test_extracts_name(self, chunker):
        content = _read_file(SRC_DIR / "dgesv.f")
        chunk = chunker.chunk_file("SRC/dgesv.f", content)[0]
        assert chunk.metadata.unit_name == "DGESV"
        assert chunk.metadata.unit_type == "subroutine"

    def test_extracts_parameters(self, chunker):
        content = _read_file(SRC_DIR / "dgesv.f")
        chunk = chunker.chunk_file("SRC/dgesv.f", content)[0]
        assert "N" in chunk.metadata.parameters
        assert "NRHS" in chunk.metadata.parameters
        assert "INFO" in chunk.metadata.parameters

    def test_extracts_calls(self, chunker):
        content = _read_file(SRC_DIR / "dgesv.f")
        chunk = chunker.chunk_file("SRC/dgesv.f", content)[0]
        assert "DGETRF" in chunk.metadata.calls
        assert "DGETRS" in chunk.metadata.calls
        assert "XERBLA" in chunk.metadata.calls

    def test_extracts_externals(self, chunker):
        content = _read_file(SRC_DIR / "dgesv.f")
        chunk = chunker.chunk_file("SRC/dgesv.f", content)[0]
        assert "DGETRF" in chunk.metadata.external_deps
        assert "DGETRS" in chunk.metadata.external_deps

    def test_extracts_purpose(self, chunker):
        content = _read_file(SRC_DIR / "dgesv.f")
        chunk = chunker.chunk_file("SRC/dgesv.f", content)[0]
        assert "solution" in chunk.metadata.purpose.lower()
        assert "linear" in chunk.metadata.purpose.lower()

    def test_classifies_precision(self, chunker):
        content = _read_file(SRC_DIR / "dgesv.f")
        chunk = chunker.chunk_file("SRC/dgesv.f", content)[0]
        assert chunk.metadata.precision == "double"

    def test_classifies_category(self, chunker):
        content = _read_file(SRC_DIR / "dgesv.f")
        chunk = chunker.chunk_file("SRC/dgesv.f", content)[0]
        assert chunk.metadata.category == "general"

    def test_enriched_content_has_summary(self, chunker):
        content = _read_file(SRC_DIR / "dgesv.f")
        chunk = chunker.chunk_file("SRC/dgesv.f", content)[0]
        assert chunk.enriched_content.startswith("# DGESV")
        assert "Purpose:" in chunk.enriched_content
        assert "File: SRC/dgesv.f" in chunk.enriched_content

    def test_chunk_id_format(self, chunker):
        content = _read_file(SRC_DIR / "dgesv.f")
        chunk = chunker.chunk_file("SRC/dgesv.f", content)[0]
        assert chunk.chunk_id == "fortran:src:dgesv:dgesv"

    def test_metadata_line_numbers(self, chunker):
        content = _read_file(SRC_DIR / "dgesv.f")
        chunk = chunker.chunk_file("SRC/dgesv.f", content)[0]
        assert chunk.metadata.start_line == 1
        assert chunk.metadata.end_line > 100
        assert chunk.metadata.file_path == "SRC/dgesv.f"

    def test_pinecone_metadata_is_flat(self, chunker):
        content = _read_file(SRC_DIR / "dgesv.f")
        chunk = chunker.chunk_file("SRC/dgesv.f", content)[0]
        meta = chunk.metadata.to_pinecone_metadata()
        assert isinstance(meta, dict)
        assert meta["unit_name"] == "DGESV"
        assert isinstance(meta["calls"], list)
        # Pinecone requires flat types
        for v in meta.values():
            assert isinstance(v, (str, int, float, bool, list))


@pytest.mark.skipif(not SRC_DIR.exists(), reason="LAPACK not downloaded")
class TestDGETRF:
    """Test against SRC/dgetrf.f — medium routine."""

    def test_produces_one_chunk(self, chunker):
        content = _read_file(SRC_DIR / "dgetrf.f")
        chunks = chunker.chunk_file("SRC/dgetrf.f", content)
        assert len(chunks) >= 1

    def test_extracts_name(self, chunker):
        content = _read_file(SRC_DIR / "dgetrf.f")
        chunk = chunker.chunk_file("SRC/dgetrf.f", content)[0]
        assert chunk.metadata.unit_name == "DGETRF"


@pytest.mark.skipif(not BLAS_DIR.exists(), reason="LAPACK not downloaded")
class TestDGEMM:
    """Test against BLAS/SRC/dgemm.f — BLAS routine."""

    def test_produces_one_chunk(self, chunker):
        content = _read_file(BLAS_DIR / "dgemm.f")
        chunks = chunker.chunk_file("BLAS/SRC/dgemm.f", content)
        assert len(chunks) >= 1

    def test_extracts_name(self, chunker):
        content = _read_file(BLAS_DIR / "dgemm.f")
        chunk = chunker.chunk_file("BLAS/SRC/dgemm.f", content)[0]
        assert chunk.metadata.unit_name == "DGEMM"

    def test_extracts_purpose(self, chunker):
        content = _read_file(BLAS_DIR / "dgemm.f")
        chunk = chunker.chunk_file("BLAS/SRC/dgemm.f", content)[0]
        assert "matrix" in chunk.metadata.purpose.lower()


@pytest.mark.skipif(not SRC_DIR.exists(), reason="LAPACK not downloaded")
class TestSinglePrecision:
    """Test single-precision variant."""

    def test_sgesv_precision(self, chunker):
        path = SRC_DIR / "sgesv.f"
        if not path.exists():
            pytest.skip("sgesv.f not found")
        content = _read_file(path)
        chunk = chunker.chunk_file("SRC/sgesv.f", content)[0]
        assert chunk.metadata.precision == "single"
        assert chunk.metadata.unit_name == "SGESV"


@pytest.mark.skipif(not SRC_DIR.exists(), reason="LAPACK not downloaded")
class TestComplexPrecision:
    """Test complex-precision variant."""

    def test_zgesv_precision(self, chunker):
        path = SRC_DIR / "zgesv.f"
        if not path.exists():
            pytest.skip("zgesv.f not found")
        content = _read_file(path)
        chunk = chunker.chunk_file("SRC/zgesv.f", content)[0]
        assert chunk.metadata.precision == "double_complex"


@pytest.mark.skipif(not SRC_DIR.exists(), reason="LAPACK not downloaded")
class TestDirectoryChunking:
    """Test chunking an entire directory."""

    def test_chunks_blas_directory(self, chunker):
        chunks = chunker.chunk_directory(str(BLAS_DIR))
        assert len(chunks) > 50  # BLAS has ~169 files
        # Every chunk should have a name
        for chunk in chunks:
            assert chunk.metadata.unit_name != ""
            assert chunk.metadata.language == "fortran"

    def test_all_chunks_have_file_paths(self, chunker):
        chunks = chunker.chunk_directory(str(BLAS_DIR))
        for chunk in chunks:
            assert chunk.metadata.file_path != ""
            assert chunk.metadata.start_line >= 1


@pytest.mark.skipif(not SRC_DIR.exists(), reason="LAPACK not downloaded")
class TestEmptyAndEdgeCases:
    """Edge case handling."""

    def test_empty_content(self, chunker):
        chunks = chunker.chunk_file("empty.f", "")
        assert chunks == []

    def test_whitespace_only(self, chunker):
        chunks = chunker.chunk_file("blank.f", "   \n  \n  ")
        assert chunks == []
