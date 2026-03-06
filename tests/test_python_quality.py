"""Comprehensive quality tests for Python code retrieval pipeline.

Tests the full stack: chunking → metadata extraction → enriched content →
retrieval logic → BM25 keyword search → reranking — all using synthetic
Pydantic-like code that exercises the thin-wrapper / delegation problem.
"""

import ast

import pytest

from legacylens.chunkers.python import (
    PythonChunker,
    _extract_base_classes,
    _extract_decorators,
    _extract_return_type,
    _extract_dunder_methods,
    _build_enriched_content,
)
from legacylens.chunkers.base import ChunkMetadata, classify_visibility
from legacylens.rag.keyword_index import KeywordIndex, _tokenize


# ============================================================================
# Part 1: AST Extraction — Realistic Pydantic-Style Code
# ============================================================================

PYDANTIC_BASE_MODEL = '''
from typing import Any, ClassVar, Generic, TypeVar
from abc import ABCMeta

T = TypeVar("T")

class BaseModel(metaclass=ABCMeta):
    """Base class for all Pydantic models.

    Delegates validation to pydantic-core (Rust).
    """
    model_config: ClassVar[dict] = {}
    __pydantic_validator__: ClassVar[Any] = None

    def __init__(self, /, **data: Any) -> None:
        """Create a new model by parsing and validating input data."""
        self.__pydantic_validator__.validate_python(data)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

    def model_validate(cls, obj: Any) -> "BaseModel":
        """Validate a Python object against the model."""
        return cls.__pydantic_validator__.validate_python(obj)

    def model_dump(self) -> dict[str, Any]:
        """Serialize the model to a dictionary."""
        return self.__pydantic_serializer__.to_python(self)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, self.__class__)
'''

PYDANTIC_FIELD_INFO = '''
from typing import Any, Optional

class FieldInfo:
    """Stores metadata about a model field.

    Contains validation constraints, default values,
    and serialization settings for a single field.
    """
    def __init__(
        self,
        default: Any = ...,
        *,
        alias: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        gt: Optional[float] = None,
        lt: Optional[float] = None,
        ge: Optional[float] = None,
        le: Optional[float] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
    ) -> None:
        self.default = default
        self.alias = alias
        self.title = title
        self.description = description
        self.metadata = {}
        self._validate_constraints(gt, lt, ge, le, min_length, max_length)

    def _validate_constraints(self, gt, lt, ge, le, min_length, max_length):
        if gt is not None and ge is not None:
            raise ValueError("Cannot specify both gt and ge")
        if lt is not None and le is not None:
            raise ValueError("Cannot specify both lt and le")
'''

PYDANTIC_VALIDATORS = '''
from typing import Any, Callable
from functools import wraps

def field_validator(*fields: str, mode: str = "after"):
    """Decorator for field-level validators.

    Validators run after (or before) Pydantic's built-in validation.
    """
    def decorator(func: Callable) -> Callable:
        func.__pydantic_field_validator__ = {
            "fields": fields,
            "mode": mode,
        }
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)
        return wrapper
    return decorator


def model_validator(*, mode: str = "after"):
    """Decorator for model-level validators.

    Runs validation on the entire model after all fields.
    """
    def decorator(func: Callable) -> Callable:
        func.__pydantic_model_validator__ = {"mode": mode}
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)
        return wrapper
    return decorator
'''

PYDANTIC_USER_MODEL = '''
from typing import Optional
from pydantic import BaseModel, field_validator, EmailStr

class UserModel(BaseModel):
    """A user account model with validation.

    Validates email format, name length, and age constraints.
    """
    name: str
    email: EmailStr
    age: Optional[int] = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Name must be at least 2 characters."""
        if len(v) < 2:
            raise ValueError("Name too short")
        return v.strip()

    @field_validator("age")
    @classmethod
    def validate_age(cls, v: Optional[int]) -> Optional[int]:
        """Age must be between 0 and 150 if provided."""
        if v is not None and (v < 0 or v > 150):
            raise ValueError("Invalid age")
        return v

    def __repr__(self) -> str:
        return f"UserModel(name={self.name!r})"
'''

PYDANTIC_CONFIG = '''
from typing import Any

class ConfigDict:
    """Configuration dictionary for model settings.

    Controls validation behavior, serialization, and other model-wide settings.
    """
    def __init__(
        self,
        *,
        strict: bool = False,
        populate_by_name: bool = False,
        validate_default: bool = False,
        arbitrary_types_allowed: bool = False,
        json_schema_extra: dict[str, Any] | None = None,
    ) -> None:
        self.strict = strict
        self.populate_by_name = populate_by_name
        self.validate_default = validate_default
        self.arbitrary_types_allowed = arbitrary_types_allowed
        self.json_schema_extra = json_schema_extra
'''

INTERNAL_VALIDATION = '''
from typing import Any

def _validate_field_default(field_info: Any, value: Any) -> Any:
    """Internal: validate a field's default value against constraints."""
    return value

def _collect_known_metadata(annotation: Any) -> dict:
    """Internal: collect metadata from type annotations."""
    return {}

def __private_helper():
    """This is double-underscore private."""
    pass
'''


@pytest.fixture
def chunker():
    return PythonChunker()


# --- 1a. Base class extraction ---

class TestBaseClassExtraction:
    def test_single_base(self, chunker):
        chunks = chunker.chunk_file("models.py", PYDANTIC_USER_MODEL)
        cls = [c for c in chunks if c.metadata.unit_name == "UserModel"][0]
        assert "BaseModel" in cls.metadata.base_classes

    def test_metaclass_base(self, chunker):
        chunks = chunker.chunk_file("base.py", PYDANTIC_BASE_MODEL)
        cls = [c for c in chunks if c.metadata.unit_name == "BaseModel"][0]
        # metaclass= is a keyword arg, not a base class
        # BaseModel has no positional bases (just metaclass kwarg)
        # The ast should not include metaclass as a base
        assert cls.metadata.base_classes == [] or "ABCMeta" not in cls.metadata.base_classes

    def test_no_bases(self, chunker):
        chunks = chunker.chunk_file("config.py", PYDANTIC_CONFIG)
        cls = [c for c in chunks if c.metadata.unit_name == "ConfigDict"][0]
        assert cls.metadata.base_classes == []


# --- 1b. Decorator extraction ---

class TestDecoratorExtraction:
    def test_decorated_methods(self, chunker):
        chunks = chunker.chunk_file("user.py", PYDANTIC_USER_MODEL)
        cls = [c for c in chunks if c.metadata.unit_name == "UserModel"][0]
        # Class itself has no decorators
        assert cls.metadata.decorators == []

    def test_top_level_function_no_decorators(self, chunker):
        chunks = chunker.chunk_file("validators.py", PYDANTIC_VALIDATORS)
        funcs = [c for c in chunks if c.metadata.unit_type == "function"]
        for f in funcs:
            assert f.metadata.decorators == []  # These are decorator factories, not decorated

    def test_decorated_standalone(self, chunker):
        code = '''
@classmethod
@field_validator("name")
def check_name(cls, v):
    return v
'''
        chunks = chunker.chunk_file("deco.py", code)
        func = [c for c in chunks if c.metadata.unit_name == "check_name"][0]
        assert "classmethod" in func.metadata.decorators


# --- 1c. Return type extraction ---

class TestReturnTypeExtraction:
    def test_function_return_type(self, chunker):
        chunks = chunker.chunk_file("validators.py", PYDANTIC_VALIDATORS)
        # field_validator returns a decorator (Callable)
        funcs = {c.metadata.unit_name: c for c in chunks if c.metadata.unit_type == "function"}
        # These return bare, no annotation in the source
        # Actually check the source...
        assert funcs["field_validator"].metadata.return_type == ""

    def test_class_methods_no_standalone_return(self, chunker):
        # Class-level chunks don't have a return type
        chunks = chunker.chunk_file("base.py", PYDANTIC_BASE_MODEL)
        cls = [c for c in chunks if c.metadata.unit_type == "class"][0]
        assert cls.metadata.return_type == ""


# --- 1d. Dunder method extraction ---

class TestDunderMethodExtraction:
    def test_basemodel_dunders(self, chunker):
        chunks = chunker.chunk_file("base.py", PYDANTIC_BASE_MODEL)
        cls = [c for c in chunks if c.metadata.unit_name == "BaseModel"][0]
        assert "__init__" in cls.metadata.dunder_methods
        assert "__repr__" in cls.metadata.dunder_methods
        assert "__eq__" in cls.metadata.dunder_methods
        assert "__init_subclass__" in cls.metadata.dunder_methods
        # Non-dunder methods should NOT be included
        assert "model_validate" not in cls.metadata.dunder_methods
        assert "model_dump" not in cls.metadata.dunder_methods

    def test_user_model_dunders(self, chunker):
        chunks = chunker.chunk_file("user.py", PYDANTIC_USER_MODEL)
        cls = [c for c in chunks if c.metadata.unit_name == "UserModel"][0]
        assert "__repr__" in cls.metadata.dunder_methods

    def test_no_dunders(self, chunker):
        chunks = chunker.chunk_file("config.py", PYDANTIC_CONFIG)
        cls = [c for c in chunks if c.metadata.unit_name == "ConfigDict"][0]
        assert "__init__" in cls.metadata.dunder_methods  # __init__ is a dunder


# --- 1e. Visibility classification ---

class TestVisibilityClassification:
    def test_public(self):
        assert classify_visibility("BaseModel") == "public"
        assert classify_visibility("validate") == "public"

    def test_protected(self):
        assert classify_visibility("_validate_constraints") == "protected"
        assert classify_visibility("_internal") == "protected"

    def test_private(self):
        assert classify_visibility("__private_helper") == "private"

    def test_dunder(self):
        assert classify_visibility("__init__") == "dunder"
        assert classify_visibility("__repr__") == "dunder"

    def test_internal_module_visibility(self, chunker):
        chunks = chunker.chunk_file("_internal/core.py", INTERNAL_VALIDATION)
        funcs = {c.metadata.unit_name: c for c in chunks if "function" in c.metadata.unit_type}
        assert funcs["_validate_field_default"].metadata.visibility == "protected"
        assert funcs["_collect_known_metadata"].metadata.visibility == "protected"
        assert funcs["__private_helper"].metadata.visibility == "private"


# ============================================================================
# Part 2: Enriched Content Quality — Semantic Signal
# ============================================================================

class TestEnrichedContentQuality:
    """Verify that enriched content carries strong semantic signal for embeddings."""

    def test_basemodel_enriched_has_structure(self, chunker):
        chunks = chunker.chunk_file("base.py", PYDANTIC_BASE_MODEL)
        cls = [c for c in chunks if c.metadata.unit_name == "BaseModel"][0]
        enriched = cls.enriched_content
        assert "# BaseModel" in enriched
        assert "Python Class" in enriched
        assert "# Implements:" in enriched
        assert "__init__" in enriched

    def test_user_model_enriched_has_inheritance(self, chunker):
        chunks = chunker.chunk_file("user.py", PYDANTIC_USER_MODEL)
        cls = [c for c in chunks if c.metadata.unit_name == "UserModel"][0]
        enriched = cls.enriched_content
        assert "# Inherits: BaseModel" in enriched
        assert "# Purpose:" in enriched
        assert "validation" in enriched.lower() or "validate" in enriched.lower()

    def test_field_info_enriched_has_purpose(self, chunker):
        chunks = chunker.chunk_file("fields.py", PYDANTIC_FIELD_INFO)
        cls = [c for c in chunks if c.metadata.unit_name == "FieldInfo"][0]
        enriched = cls.enriched_content
        assert "metadata" in enriched.lower() or "field" in enriched.lower()

    def test_legacy_prefix_applied(self, chunker):
        chunks = chunker.chunk_file("v1/validators.py", PYDANTIC_VALIDATORS)
        for c in chunks:
            if c.metadata.unit_type == "function":
                assert "[LEGACY]" in c.enriched_content
                assert c.metadata.module_tier == "legacy"

    def test_internal_prefix_applied(self, chunker):
        chunks = chunker.chunk_file("_internal/core.py", INTERNAL_VALIDATION)
        for c in chunks:
            if "function" in c.metadata.unit_type:
                assert c.metadata.module_tier == "internal"
                assert "[INTERNAL]" in c.enriched_content

    def test_enriched_content_includes_calls(self, chunker):
        chunks = chunker.chunk_file("base.py", PYDANTIC_BASE_MODEL)
        cls = [c for c in chunks if c.metadata.unit_name == "BaseModel"][0]
        # BaseModel calls validate_python, to_python, etc.
        assert "validate_python" in cls.metadata.calls


# ============================================================================
# Part 3: Metadata → Pinecone Serialization Round-Trip
# ============================================================================

class TestPineconeMetadataSerialization:
    def test_all_new_fields_serialize(self, chunker):
        chunks = chunker.chunk_file("user.py", PYDANTIC_USER_MODEL)
        cls = [c for c in chunks if c.metadata.unit_name == "UserModel"][0]
        meta = cls.metadata.to_pinecone_metadata()

        # Verify flat structure
        for k, v in meta.items():
            assert isinstance(v, (str, int, float, bool, list)), f"Key {k} has type {type(v)}"

        # Verify new fields present
        assert "base_classes" in meta
        assert meta["base_classes"] == ["BaseModel"]
        assert "dunder_methods" in meta
        assert "__repr__" in meta["dunder_methods"]

    def test_empty_optional_fields_omitted(self, chunker):
        chunks = chunker.chunk_file("validators.py", PYDANTIC_VALIDATORS)
        func = [c for c in chunks if c.metadata.unit_name == "field_validator"][0]
        meta = func.metadata.to_pinecone_metadata()

        # Empty lists/strings should not appear
        assert "base_classes" not in meta
        assert "decorators" not in meta
        assert "return_type" not in meta
        assert "dunder_methods" not in meta

    def test_visibility_only_when_non_public(self, chunker):
        chunks = chunker.chunk_file("_internal/core.py", INTERNAL_VALIDATION)
        funcs = {c.metadata.unit_name: c for c in chunks if "function" in c.metadata.unit_type}

        # Protected function should have visibility in metadata
        meta_prot = funcs["_validate_field_default"].metadata.to_pinecone_metadata()
        assert meta_prot.get("visibility") == "protected"

        # Private function should have visibility
        meta_priv = funcs["__private_helper"].metadata.to_pinecone_metadata()
        assert meta_priv.get("visibility") == "private"

    def test_public_visibility_omitted(self, chunker):
        chunks = chunker.chunk_file("validators.py", PYDANTIC_VALIDATORS)
        func = [c for c in chunks if c.metadata.unit_name == "field_validator"][0]
        meta = func.metadata.to_pinecone_metadata()
        assert "visibility" not in meta  # public is the default, omitted


# ============================================================================
# Part 4: BM25 Keyword Index — Pydantic-Style Corpus
# ============================================================================

class TestBM25PydanticCorpus:
    """Test BM25 keyword search against a realistic Pydantic-like chunk corpus."""

    @pytest.fixture
    def pydantic_index(self, chunker):
        """Build a BM25 index from our Pydantic-like test corpus."""
        all_chunks = []
        files = [
            ("base.py", PYDANTIC_BASE_MODEL),
            ("fields.py", PYDANTIC_FIELD_INFO),
            ("validators.py", PYDANTIC_VALIDATORS),
            ("user.py", PYDANTIC_USER_MODEL),
            ("config.py", PYDANTIC_CONFIG),
            ("_internal/core.py", INTERNAL_VALIDATION),
        ]
        for path, code in files:
            all_chunks.extend(chunker.chunk_file(path, code))

        idx = KeywordIndex()
        idx.build(all_chunks)
        return idx, all_chunks

    def test_validate_query_finds_validators(self, pydantic_index):
        idx, chunks = pydantic_index
        results = idx.search("validate field types", top_k=5)
        ids = [cid for cid, _ in results]
        names = self._ids_to_names(ids, chunks)
        # Should surface validation-related code
        assert any("validator" in n.lower() or "validate" in n.lower() or "UserModel" in n
                    for n in names), f"Expected validation code, got {names}"

    def test_basemodel_query_finds_basemodel(self, pydantic_index):
        idx, chunks = pydantic_index
        results = idx.search("BaseModel", top_k=5)
        ids = [cid for cid, _ in results]
        names = self._ids_to_names(ids, chunks)
        assert any("BaseModel" in n or "UserModel" in n for n in names), f"Got {names}"

    def test_field_info_query(self, pydantic_index):
        idx, chunks = pydantic_index
        results = idx.search("field information constraints", top_k=5)
        ids = [cid for cid, _ in results]
        names = self._ids_to_names(ids, chunks)
        assert any("FieldInfo" in n for n in names), f"Got {names}"

    def test_config_query(self, pydantic_index):
        idx, chunks = pydantic_index
        results = idx.search("configuration strict mode", top_k=5)
        ids = [cid for cid, _ in results]
        names = self._ids_to_names(ids, chunks)
        assert any("ConfigDict" in n or "config" in n.lower() for n in names), f"Got {names}"

    def test_dunder_init_query(self, pydantic_index):
        idx, chunks = pydantic_index
        results = idx.search("__init__ constructor", top_k=5)
        ids = [cid for cid, _ in results]
        names = self._ids_to_names(ids, chunks)
        # Classes with __init__ in their dunder_methods should rank high
        assert len(results) > 0, "Should find classes with __init__"

    def test_serialization_query(self, pydantic_index):
        idx, chunks = pydantic_index
        results = idx.search("serialize model dump", top_k=5)
        ids = [cid for cid, _ in results]
        names = self._ids_to_names(ids, chunks)
        # BaseModel has model_dump in its calls
        assert any("BaseModel" in n for n in names), f"Got {names}"

    def test_internal_code_is_searchable(self, pydantic_index):
        idx, chunks = pydantic_index
        results = idx.search("internal validate default", top_k=5)
        ids = [cid for cid, _ in results]
        names = self._ids_to_names(ids, chunks)
        assert any("_validate_field_default" in n for n in names), f"Got {names}"

    def _ids_to_names(self, ids: list[str], chunks):
        id_to_name = {c.chunk_id: c.metadata.unit_name for c in chunks}
        return [id_to_name.get(cid, cid) for cid in ids]


# ============================================================================
# Part 5: Tokenizer Quality
# ============================================================================

class TestTokenizerQuality:
    def test_pydantic_identifiers(self):
        tokens = _tokenize("BaseModel field_validator model_validate")
        assert "base" in tokens
        assert "model" in tokens
        assert "field" in tokens
        assert "validator" in tokens
        assert "validate" in tokens

    def test_dunder_methods(self):
        tokens = _tokenize("__init__ __repr__ __eq__")
        assert "init" in tokens
        assert "repr" in tokens
        assert "eq" in tokens

    def test_type_annotations(self):
        tokens = _tokenize("Optional[str] dict[str, Any] list[int]")
        assert "optional" in tokens
        assert "any" in tokens

    def test_dotted_paths(self):
        tokens = _tokenize("pydantic.fields.FieldInfo")
        assert "pydantic" in tokens
        assert "fields" in tokens
        # FieldInfo gets split by camelCase
        assert "field" in tokens
        assert "info" in tokens

    def test_mixed_query(self):
        tokens = _tokenize("How does Pydantic validate field types?")
        assert "pydantic" in tokens
        assert "validate" in tokens
        assert "field" in tokens
        assert "types" in tokens
        # Stopwords removed
        assert "how" not in tokens
        assert "does" not in tokens


# ============================================================================
# Part 6: Retrieval Logic — Entity Extraction for Python
# ============================================================================

class TestPythonEntityExtraction:
    def test_camel_case_extraction(self):
        from legacylens.rag.retrieve import _extract_python_entities
        result = _extract_python_entities("What does BaseModel.__init__ do?")
        assert "BaseModel" in result["classes"]

    def test_snake_case_extraction(self):
        from legacylens.rag.retrieve import _extract_python_entities
        result = _extract_python_entities("How does field_validator work?")
        assert "field_validator" in result["functions"]

    def test_multiple_entities(self):
        from legacylens.rag.retrieve import _extract_python_entities
        result = _extract_python_entities("Does UserModel use field_validator for name?")
        assert "UserModel" in result["classes"]
        assert "field_validator" in result["functions"]

    def test_dotted_path(self):
        from legacylens.rag.retrieve import _extract_python_entities
        result = _extract_python_entities("How is pydantic.fields.FieldInfo used?")
        assert any("pydantic" in m for m in result["modules"])


# ============================================================================
# Part 7: Decorator Concept Mapping
# ============================================================================

class TestDecoratorConceptMapping:
    def test_validate_concept(self):
        from legacylens.rag.retrieve import _DECORATOR_CONCEPTS
        assert "validate" in _DECORATOR_CONCEPTS
        decorators = _DECORATOR_CONCEPTS["validate"]
        assert "field_validator" in decorators
        assert "model_validator" in decorators
        assert "validator" in decorators

    def test_serialize_concept(self):
        from legacylens.rag.retrieve import _DECORATOR_CONCEPTS
        assert "serialize" in _DECORATOR_CONCEPTS
        decorators = _DECORATOR_CONCEPTS["serialize"]
        assert "field_serializer" in decorators

    def test_config_concept(self):
        from legacylens.rag.retrieve import _DECORATOR_CONCEPTS
        assert "config" in _DECORATOR_CONCEPTS
        decorators = _DECORATOR_CONCEPTS["config"]
        assert "dataclass" in decorators


# ============================================================================
# Part 8: Legacy Intent Detection
# ============================================================================

class TestLegacyIntentDetection:
    def test_v1_migration(self):
        from legacylens.rag.retrieve import _query_wants_legacy
        assert _query_wants_legacy("How do I migrate v1 validators to v2?")

    def test_legacy_compat(self):
        from legacylens.rag.retrieve import _query_wants_legacy
        assert _query_wants_legacy("What does the legacy compat layer do?")

    def test_upgrade(self):
        from legacylens.rag.retrieve import _query_wants_legacy
        assert _query_wants_legacy("How to upgrade from the old api?")

    def test_normal_query_no_legacy(self):
        from legacylens.rag.retrieve import _query_wants_legacy
        assert not _query_wants_legacy("How does Pydantic validate field types?")
        assert not _query_wants_legacy("What does BaseModel.__init__ do?")
        assert not _query_wants_legacy("How do I use field_validator?")


# ============================================================================
# Part 9: Enriched Content Builder — Conditional Lines
# ============================================================================

class TestEnrichedContentBuilder:
    def test_all_fields_present(self):
        enriched = _build_enriched_content(
            source="class Foo: pass",
            name="Foo",
            unit_type="class",
            purpose="A foo class",
            calls=["bar", "baz"],
            file_path="foo.py",
            start_line=1,
            end_line=10,
            module_tier="current",
            base_classes=["BaseModel", "Generic[T]"],
            decorators=["dataclass"],
            return_type="",
            dunder_methods=["__init__", "__repr__"],
        )
        assert "# Inherits: BaseModel, Generic[T]" in enriched
        assert "# Decorators: @dataclass" in enriched
        assert "# Purpose: A foo class" in enriched
        assert "# Implements: __init__, __repr__" in enriched
        assert "File: foo.py" in enriched
        assert "Calls: bar, baz" in enriched

    def test_empty_optional_fields_omitted(self):
        enriched = _build_enriched_content(
            source="def foo(): pass",
            name="foo",
            unit_type="function",
            purpose="A function",
            calls=[],
            file_path="foo.py",
            start_line=1,
            end_line=1,
        )
        assert "Inherits" not in enriched
        assert "Decorators" not in enriched
        assert "Returns" not in enriched
        assert "Implements" not in enriched

    def test_return_type_shown(self):
        enriched = _build_enriched_content(
            source="def foo(): pass",
            name="foo",
            unit_type="function",
            purpose="",
            calls=[],
            file_path="foo.py",
            start_line=1,
            end_line=1,
            return_type="Optional[str]",
        )
        assert "# Returns: Optional[str]" in enriched

    def test_legacy_prefix(self):
        enriched = _build_enriched_content(
            source="class Old: pass",
            name="Old",
            unit_type="class",
            purpose="Legacy class",
            calls=[],
            file_path="v1/old.py",
            start_line=1,
            end_line=1,
            module_tier="legacy",
        )
        assert "# [LEGACY] Old" in enriched


# ============================================================================
# Part 10: End-to-End Chunking Coverage
# ============================================================================

class TestEndToEndChunkingCoverage:
    """Verify that the full set of Pydantic-like files chunks correctly."""

    @pytest.fixture
    def all_chunks(self, chunker):
        files = [
            ("base.py", PYDANTIC_BASE_MODEL),
            ("fields.py", PYDANTIC_FIELD_INFO),
            ("validators.py", PYDANTIC_VALIDATORS),
            ("user.py", PYDANTIC_USER_MODEL),
            ("config.py", PYDANTIC_CONFIG),
            ("_internal/core.py", INTERNAL_VALIDATION),
            ("v1/validators.py", PYDANTIC_VALIDATORS),  # Legacy copy
        ]
        chunks = []
        for path, code in files:
            chunks.extend(chunker.chunk_file(path, code))
        return chunks

    def test_total_chunks(self, all_chunks):
        # Should have a reasonable number of chunks
        assert len(all_chunks) >= 10, f"Only {len(all_chunks)} chunks, expected 10+"

    def test_all_unit_types_present(self, all_chunks):
        types = {c.metadata.unit_type for c in all_chunks}
        assert "class" in types
        assert "function" in types
        assert "module" in types  # preamble chunks

    def test_all_tiers_present(self, all_chunks):
        tiers = {c.metadata.module_tier for c in all_chunks}
        assert "current" in tiers
        assert "legacy" in tiers
        assert "internal" in tiers

    def test_no_empty_enriched_content(self, all_chunks):
        for c in all_chunks:
            assert c.enriched_content.strip(), f"Empty enriched content for {c.metadata.unit_name}"

    def test_no_empty_unit_names(self, all_chunks):
        for c in all_chunks:
            assert c.metadata.unit_name, f"Empty unit_name at {c.metadata.file_path}:{c.metadata.start_line}"

    def test_chunk_ids_unique(self, all_chunks):
        ids = [c.chunk_id for c in all_chunks]
        assert len(ids) == len(set(ids)), f"Duplicate chunk IDs: {[x for x in ids if ids.count(x) > 1]}"

    def test_classes_have_inheritance_info(self, all_chunks):
        """Classes that inherit should have base_classes populated."""
        user_model = [c for c in all_chunks if c.metadata.unit_name == "UserModel"][0]
        assert user_model.metadata.base_classes == ["BaseModel"]

    def test_legacy_chunks_marked(self, all_chunks):
        legacy = [c for c in all_chunks if c.metadata.module_tier == "legacy"]
        assert len(legacy) >= 2, "Should have legacy chunks from v1/"
        for c in legacy:
            assert "[LEGACY]" in c.enriched_content

    def test_bm25_index_buildable(self, all_chunks):
        """All chunks should be indexable by BM25."""
        idx = KeywordIndex()
        idx.build(all_chunks)
        results = idx.search("validate", top_k=10)
        assert len(results) > 0, "BM25 search should find results"
