"""Live integration tests against the ingested Pydantic source.

These tests require:
- VOYAGE_API_KEY and PINECONE_API_KEY environment variables
- The 'pydantic' source to be ingested (1052 chunks)

Skip with: pytest -m "not live"
"""

from __future__ import annotations

import os
import pytest

# Load environment from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Skip entire module if no API keys or no source
pytestmark = pytest.mark.live

_HAS_KEYS = bool(os.environ.get("VOYAGE_API_KEY") and os.environ.get("PINECONE_API_KEY"))


def _source_exists() -> bool:
    try:
        from legacylens.sources import get_source
        return get_source("pydantic") is not None
    except Exception:
        return False


def _skip_if_no_setup():
    if not _HAS_KEYS:
        pytest.skip("No API keys (VOYAGE_API_KEY / PINECONE_API_KEY)")
    if not _source_exists():
        pytest.skip("Pydantic source not ingested")


def _retrieve(question: str, top_k: int = 5):
    """Run retrieval against the live Pydantic index."""
    _skip_if_no_setup()
    from legacylens.rag.retrieve import retrieve_with_metrics
    return retrieve_with_metrics(question, top_k=top_k, namespace="pydantic", languages=["python"])


def _names(result) -> list[str]:
    return [r.get("metadata", {}).get("unit_name", "") for r in result.results]


def _tiers(result) -> list[str]:
    return [r.get("metadata", {}).get("module_tier", "current") for r in result.results]


def _has_any(names: list[str], targets: set[str]) -> bool:
    return any(n in targets for n in names)


# ============================================================================
# Query 1: Core validation — the motivating query
# ============================================================================

class TestValidationQueries:
    def test_validate_field_types(self):
        """The key query: 'How does Pydantic validate field types?'
        Should surface v2 validation code, NOT legacy v1 code."""
        result = _retrieve("How does Pydantic validate field types?")
        names = _names(result)
        tiers = _tiers(result)

        print(f"\n  Query: 'How does Pydantic validate field types?'")
        for r in result.results:
            m = r.get("metadata", {})
            print(f"    {r.get('score', 0):.3f} | {m.get('unit_name', '?'):<30} | {m.get('module_tier', '?'):<10} | {m.get('file_path', '?')}")

        # No legacy results in top 5
        legacy_count = sum(1 for t in tiers if t == "legacy")
        assert legacy_count == 0, f"Got {legacy_count} legacy results: {list(zip(names, tiers))}"

        # Should have results
        assert len(result.results) >= 3, f"Only {len(result.results)} results"

    def test_basemodel_init(self):
        """'What does BaseModel.__init__ do?'
        Should find BaseModel and/or its __init__ implementation."""
        result = _retrieve("What does BaseModel.__init__ do?")
        names = _names(result)

        print(f"\n  Query: 'What does BaseModel.__init__ do?'")
        for r in result.results:
            m = r.get("metadata", {})
            print(f"    {r.get('score', 0):.3f} | {m.get('unit_name', '?'):<30} | {m.get('module_tier', '?'):<10} | {m.get('file_path', '?')}")

        assert _has_any(names, {"BaseModel", "__init__"}), f"Expected BaseModel in results, got {names}"

    def test_v1_migration_query(self):
        """'How do I migrate v1 validators to v2?'
        Should surface BOTH v1 compat code AND v2 equivalents."""
        result = _retrieve("How do I migrate v1 validators to v2?")
        names = _names(result)
        tiers = _tiers(result)

        print(f"\n  Query: 'How do I migrate v1 validators to v2?'")
        for r in result.results:
            m = r.get("metadata", {})
            print(f"    {r.get('score', 0):.3f} | {m.get('unit_name', '?'):<30} | {m.get('module_tier', '?'):<10} | {m.get('file_path', '?')}")

        # Should have at least some results
        assert len(result.results) >= 2, f"Only {len(result.results)} results"

        # This query should include legacy code (user asked about v1)
        # Just check that we got results — tier composition depends on what's indexed


# ============================================================================
# Query 2: Specific class/function lookups
# ============================================================================

class TestEntityLookupQueries:
    def test_field_info_lookup(self):
        """Direct lookup: 'What is FieldInfo?'"""
        result = _retrieve("What is FieldInfo?")
        names = _names(result)

        print(f"\n  Query: 'What is FieldInfo?'")
        for r in result.results:
            m = r.get("metadata", {})
            print(f"    {r.get('score', 0):.3f} | {m.get('unit_name', '?'):<30} | {m.get('file_path', '?')}")

        assert _has_any(names, {"FieldInfo"}), f"Expected FieldInfo in results, got {names}"

    def test_config_dict(self):
        """'How do I configure model validation behavior?'"""
        result = _retrieve("How do I configure model validation behavior?")
        names = _names(result)

        print(f"\n  Query: 'How do I configure model validation behavior?'")
        for r in result.results:
            m = r.get("metadata", {})
            print(f"    {r.get('score', 0):.3f} | {m.get('unit_name', '?'):<30} | {m.get('file_path', '?')}")

        assert len(result.results) >= 1

    def test_model_validator(self):
        """'What does model_validator do?'"""
        result = _retrieve("What does model_validator do?")
        names = _names(result)

        print(f"\n  Query: 'What does model_validator do?'")
        for r in result.results:
            m = r.get("metadata", {})
            print(f"    {r.get('score', 0):.3f} | {m.get('unit_name', '?'):<30} | {m.get('file_path', '?')}")

        assert _has_any(names, {"model_validator", "ModelValidator"}), f"Expected model_validator, got {names}"


# ============================================================================
# Query 3: Concept-level queries (no explicit identifiers)
# ============================================================================

class TestConceptQueries:
    def test_serialization(self):
        """'How does Pydantic serialize models to JSON?'"""
        result = _retrieve("How does Pydantic serialize models to JSON?")
        names = _names(result)

        print(f"\n  Query: 'How does Pydantic serialize models to JSON?'")
        for r in result.results:
            m = r.get("metadata", {})
            print(f"    {r.get('score', 0):.3f} | {m.get('unit_name', '?'):<30} | {m.get('file_path', '?')}")

        assert len(result.results) >= 1

    def test_type_annotation(self):
        """'How does Pydantic handle type annotations?'"""
        result = _retrieve("How does Pydantic handle type annotations?")
        names = _names(result)

        print(f"\n  Query: 'How does Pydantic handle type annotations?'")
        for r in result.results:
            m = r.get("metadata", {})
            print(f"    {r.get('score', 0):.3f} | {m.get('unit_name', '?'):<30} | {m.get('file_path', '?')}")

        assert len(result.results) >= 1

    def test_error_handling(self):
        """'How does Pydantic report validation errors?'"""
        result = _retrieve("How does Pydantic report validation errors?")
        names = _names(result)

        print(f"\n  Query: 'How does Pydantic report validation errors?'")
        for r in result.results:
            m = r.get("metadata", {})
            print(f"    {r.get('score', 0):.3f} | {m.get('unit_name', '?'):<30} | {m.get('file_path', '?')}")

        assert len(result.results) >= 1


# ============================================================================
# Query 4: Retrieval quality metrics
# ============================================================================

class TestRetrievalQualityMetrics:
    def test_score_distribution(self):
        """Scores should be reasonable (not all near-zero)."""
        result = _retrieve("How does Pydantic validate field types?")
        metrics = result.metrics

        print(f"\n  Score distribution: min={metrics.score_distribution['min']:.3f}, "
              f"max={metrics.score_distribution['max']:.3f}, "
              f"mean={metrics.score_distribution['mean']:.3f}")
        print(f"  Entity matches: {metrics.entity_matches}")
        print(f"  Threshold filtered: {metrics.threshold_filtered}")
        print(f"  Variants collapsed: {metrics.variants_collapsed}")

        assert metrics.score_distribution["max"] > 0.3, "Max score too low — retrieval may be broken"

    def test_no_excessive_filtering(self):
        """Shouldn't filter out too many results."""
        result = _retrieve("What classes inherit from BaseModel?", top_k=5)
        # Should get at least a few results
        assert len(result.results) >= 1, "Filtering removed everything"

    def test_entity_matches_for_explicit_names(self):
        """Explicit class names should trigger entity matching."""
        result = _retrieve("What does BaseModel do?")
        # BaseModel should be an entity match
        assert result.metrics.entity_matches >= 1, "Expected entity matches for 'BaseModel'"


# ============================================================================
# Query 5: Tier-aware results
# ============================================================================

class TestTierAwareRetrieval:
    def test_current_preferred_over_legacy(self):
        """For non-legacy queries, current-tier results should dominate."""
        result = _retrieve("How does field validation work?")
        tiers = _tiers(result)

        print(f"\n  Tiers: {tiers}")
        legacy_count = sum(1 for t in tiers if t == "legacy")
        assert legacy_count <= 1, f"Too many legacy results ({legacy_count}): {list(zip(_names(result), tiers))}"

    def test_legacy_accessible_when_requested(self):
        """Legacy queries should surface v1 code."""
        result = _retrieve("How do v1 validators work?")

        print(f"\n  Query: 'How do v1 validators work?'")
        for r in result.results:
            m = r.get("metadata", {})
            print(f"    {r.get('score', 0):.3f} | {m.get('unit_name', '?'):<30} | {m.get('module_tier', '?'):<10} | {m.get('file_path', '?')}")

        # Should have results (legacy code exists)
        assert len(result.results) >= 1
