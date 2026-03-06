"""Tests for the Python chunker — all using synthetic snippets (no external data)."""

import pytest

from legacylens.chunkers.python import PythonChunker
from legacylens.chunkers.base import ChunkerRegistry
from legacylens.chunkers.fortran import FortranChunker


@pytest.fixture
def chunker():
    return PythonChunker()


@pytest.fixture
def registry():
    reg = ChunkerRegistry()
    reg.register(FortranChunker())
    reg.register(PythonChunker())
    return reg


# --- Registry integration ---


def test_registry_selects_python(registry):
    assert registry.get_chunker("foo.py") is not None
    assert registry.get_chunker("foo.pyi") is not None


def test_registry_selects_fortran(registry):
    assert registry.get_chunker("foo.f") is not None


def test_registry_lists_all_extensions(registry):
    exts = registry.supported_extensions()
    assert ".py" in exts
    assert ".f" in exts


# --- Simple function ---


def test_simple_function(chunker):
    code = '''
def greet(name: str) -> str:
    """Say hello to someone."""
    return f"Hello, {name}!"
'''
    chunks = chunker.chunk_file("hello.py", code)
    assert len(chunks) >= 1
    func_chunks = [c for c in chunks if c.metadata.unit_type == "function"]
    assert len(func_chunks) == 1
    chunk = func_chunks[0]
    assert chunk.metadata.unit_name == "greet"
    assert chunk.metadata.unit_type == "function"
    assert chunk.metadata.language == "python"
    assert "name: str" in chunk.metadata.parameters
    assert "Say hello" in chunk.metadata.purpose


def test_simple_function_calls(chunker):
    code = '''
def process():
    data = fetch_data()
    result = transform(data)
    save(result)
'''
    chunks = chunker.chunk_file("proc.py", code)
    func_chunks = [c for c in chunks if c.metadata.unit_type == "function"]
    assert len(func_chunks) == 1
    assert "fetch_data" in func_chunks[0].metadata.calls
    assert "transform" in func_chunks[0].metadata.calls
    assert "save" in func_chunks[0].metadata.calls


# --- Async function ---


def test_async_function(chunker):
    code = '''
async def fetch(url: str) -> bytes:
    """Fetch a URL."""
    async with aiohttp.get(url) as resp:
        return await resp.read()
'''
    chunks = chunker.chunk_file("fetch.py", code)
    func_chunks = [c for c in chunks if "async" in c.metadata.unit_type]
    assert len(func_chunks) == 1
    assert func_chunks[0].metadata.unit_type == "async_function"
    assert func_chunks[0].metadata.unit_name == "fetch"


# --- Class with methods ---


def test_class_with_methods(chunker):
    code = '''
class Calculator:
    """A simple calculator."""

    def __init__(self, value: int = 0):
        self.value = value

    def add(self, n: int) -> int:
        self.value += n
        return self.value

    def reset(self):
        self.value = 0
'''
    chunks = chunker.chunk_file("calc.py", code)
    class_chunks = [c for c in chunks if c.metadata.unit_type == "class"]
    assert len(class_chunks) == 1
    chunk = class_chunks[0]
    assert chunk.metadata.unit_name == "Calculator"
    assert "A simple calculator" in chunk.metadata.purpose


# --- Multiple top-level units ---


def test_multiple_functions(chunker):
    code = '''
import os

def foo():
    pass

def bar():
    pass

def baz():
    pass
'''
    chunks = chunker.chunk_file("multi.py", code)
    func_chunks = [c for c in chunks if c.metadata.unit_type == "function"]
    assert len(func_chunks) == 3
    names = {c.metadata.unit_name for c in func_chunks}
    assert names == {"foo", "bar", "baz"}


def test_preamble_chunk(chunker):
    code = '''
"""Module docstring."""

import os
import sys

CONSTANT = 42
'''
    chunks = chunker.chunk_file("preamble.py", code)
    module_chunks = [c for c in chunks if c.metadata.unit_type == "module"]
    assert len(module_chunks) == 1
    assert module_chunks[0].metadata.unit_name == "preamble"
    assert "os" in module_chunks[0].metadata.uses
    assert "sys" in module_chunks[0].metadata.uses


# --- Module preamble + functions ---


def test_preamble_plus_functions(chunker):
    code = '''
"""A utility module."""

import json
from pathlib import Path

CONFIG_PATH = Path("config.json")

def load_config():
    return json.loads(CONFIG_PATH.read_text())

def save_config(data):
    CONFIG_PATH.write_text(json.dumps(data))
'''
    chunks = chunker.chunk_file("utils.py", code)
    # Should have preamble + 2 functions
    assert len(chunks) >= 3
    module_chunks = [c for c in chunks if c.metadata.unit_type == "module"]
    func_chunks = [c for c in chunks if c.metadata.unit_type == "function"]
    assert len(module_chunks) == 1
    assert len(func_chunks) == 2


# --- Edge cases ---


def test_empty_file(chunker):
    chunks = chunker.chunk_file("empty.py", "")
    assert chunks == []


def test_whitespace_only(chunker):
    chunks = chunker.chunk_file("blank.py", "   \n  \n  ")
    assert chunks == []


def test_syntax_error_fallback(chunker):
    code = '''
def broken(
    # missing closing paren
    print("oops"
'''
    chunks = chunker.chunk_file("broken.py", code)
    assert len(chunks) == 1
    assert chunks[0].metadata.unit_type == "module"
    assert chunks[0].metadata.language == "python"


def test_init_file(chunker):
    code = '''
"""Package init."""

from .module import MyClass
from .utils import helper
'''
    chunks = chunker.chunk_file("__init__.py", code)
    assert len(chunks) >= 1
    # Should have a module preamble
    module_chunks = [c for c in chunks if c.metadata.unit_type == "module"]
    assert len(module_chunks) >= 1


def test_decorated_function(chunker):
    code = '''
import functools

def decorator(f):
    return f

@decorator
@functools.lru_cache
def cached_compute(x: int) -> int:
    """Compute something expensive."""
    return x * x
'''
    chunks = chunker.chunk_file("deco.py", code)
    func_chunks = [c for c in chunks if c.metadata.unit_name == "cached_compute"]
    assert len(func_chunks) == 1
    # Decorators should be included in the source
    assert "@decorator" in func_chunks[0].content


# --- chunk_id format ---


def test_chunk_id_format(chunker):
    code = '''
def my_func():
    pass
'''
    chunks = chunker.chunk_file("src/utils.py", code)
    func_chunks = [c for c in chunks if c.metadata.unit_type == "function"]
    assert len(func_chunks) == 1
    assert func_chunks[0].chunk_id.startswith("python:")
    assert "my_func" in func_chunks[0].chunk_id


# --- Large class splitting ---


def test_large_class_splitting(chunker):
    # Generate a class > 60K chars
    methods = []
    for i in range(300):
        # Each method ~250 chars; 300 * 250 = 75K > 60K threshold
        methods.append(f'''
    def method_{i}(self):
        """Method {i} docstring that adds some length to the content for splitting test purposes."""
        x_{i} = {i}
        y_{i} = x_{i} * 2 + {i} * 3 + {i} * 4 + {i} * 5
        z_{i} = y_{i} + x_{i} + {i} * 100 + {i} * 200 + {i} * 300
        w_{i} = z_{i} * x_{i} + y_{i} * z_{i} + {i} * 400 + {i} * 500
        return w_{i} + z_{i} + y_{i} + x_{i} + {i} * 600 + {i} * 700
''')
    code = f'''
class HugeClass:
    """A very large class."""

    def __init__(self):
        self.value = 0
{''.join(methods)}
'''
    chunks = chunker.chunk_file("huge.py", code)
    class_chunks = [c for c in chunks if c.metadata.unit_name == "HugeClass"]
    # Should be split into multiple chunks
    assert len(class_chunks) > 1
    # First chunk should be the preamble
    assert class_chunks[0].metadata.chunk_index == 1
    assert class_chunks[0].metadata.chunk_total == len(class_chunks)


# --- Script with no functions ---


def test_script_no_functions(chunker):
    code = '''
import sys

for arg in sys.argv[1:]:
    print(f"Processing {arg}")

if __name__ == "__main__":
    print("Done")
'''
    chunks = chunker.chunk_file("script.py", code)
    assert len(chunks) >= 1


# --- Enriched content ---


def test_enriched_content_format(chunker):
    code = '''
def compute(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y
'''
    chunks = chunker.chunk_file("math.py", code)
    func_chunks = [c for c in chunks if c.metadata.unit_type == "function"]
    enriched = func_chunks[0].enriched_content
    assert enriched.startswith("# compute")
    assert "Python Function" in enriched
    assert "Purpose: Add two numbers" in enriched
    assert "File: math.py" in enriched


# --- Metadata to_pinecone_metadata ---


def test_pinecone_metadata_is_flat(chunker):
    code = '''
def example(a: int, b: str) -> bool:
    """Check something."""
    return len(b) > a
'''
    chunks = chunker.chunk_file("ex.py", code)
    func_chunks = [c for c in chunks if c.metadata.unit_type == "function"]
    meta = func_chunks[0].metadata.to_pinecone_metadata()
    assert isinstance(meta, dict)
    assert meta["unit_name"] == "example"
    assert meta["language"] == "python"
    assert meta["unit_type"] == "function"
    for v in meta.values():
        assert isinstance(v, (str, int, float, bool, list))


# --- Module tier tests ---


def test_module_tier_legacy(chunker):
    code = '''
def validate(v):
    """Legacy validator."""
    return v
'''
    chunks = chunker.chunk_file("v1/fields.py", code)
    func_chunks = [c for c in chunks if c.metadata.unit_type == "function"]
    assert len(func_chunks) == 1
    assert func_chunks[0].metadata.module_tier == "legacy"


def test_module_tier_current(chunker):
    code = '''
def validate(v):
    """Current validator."""
    return v
'''
    chunks = chunker.chunk_file("main.py", code)
    func_chunks = [c for c in chunks if c.metadata.unit_type == "function"]
    assert len(func_chunks) == 1
    assert func_chunks[0].metadata.module_tier == "current"


def test_module_tier_in_pinecone_metadata(chunker):
    code = '''
def example():
    pass
'''
    chunks = chunker.chunk_file("v1/compat.py", code)
    func_chunks = [c for c in chunks if c.metadata.unit_type == "function"]
    meta = func_chunks[0].metadata.to_pinecone_metadata()
    assert "module_tier" in meta
    assert meta["module_tier"] == "legacy"


def test_enriched_content_legacy_prefix(chunker):
    code = '''
def validate(v):
    """Legacy validator."""
    return v
'''
    chunks = chunker.chunk_file("v1/fields.py", code)
    func_chunks = [c for c in chunks if c.metadata.unit_type == "function"]
    assert "[LEGACY]" in func_chunks[0].enriched_content


def test_enriched_content_current_no_prefix(chunker):
    code = '''
def validate(v):
    """Current validator."""
    return v
'''
    chunks = chunker.chunk_file("fields.py", code)
    func_chunks = [c for c in chunks if c.metadata.unit_type == "function"]
    assert "[LEGACY]" not in func_chunks[0].enriched_content
    assert "[DEPRECATED]" not in func_chunks[0].enriched_content


# --- AST extraction helpers ---


def test_extract_base_classes(chunker):
    code = '''
class MyModel(BaseModel, Generic[T]):
    """A model."""
    pass
'''
    chunks = chunker.chunk_file("model.py", code)
    cls = [c for c in chunks if c.metadata.unit_type == "class"][0]
    assert "BaseModel" in cls.metadata.base_classes
    assert "Generic[T]" in cls.metadata.base_classes


def test_extract_decorators_function(chunker):
    code = '''
@field_validator("name")
@classmethod
def validate_name(cls, v):
    """Validate name."""
    return v
'''
    chunks = chunker.chunk_file("validators.py", code)
    func = [c for c in chunks if c.metadata.unit_name == "validate_name"][0]
    assert "classmethod" in func.metadata.decorators
    assert any("field_validator" in d for d in func.metadata.decorators)


def test_extract_decorators_class(chunker):
    code = '''
@dataclass
class Config:
    """Config."""
    name: str = "default"
'''
    chunks = chunker.chunk_file("config.py", code)
    cls = [c for c in chunks if c.metadata.unit_type == "class"][0]
    assert "dataclass" in cls.metadata.decorators


def test_extract_return_type(chunker):
    code = '''
def process(x: int) -> Optional[str]:
    """Process something."""
    return str(x)
'''
    chunks = chunker.chunk_file("proc.py", code)
    func = [c for c in chunks if c.metadata.unit_type == "function"][0]
    assert func.metadata.return_type == "Optional[str]"


def test_extract_dunder_methods(chunker):
    code = '''
class Foo:
    """A class with dunders."""
    def __init__(self):
        pass
    def __eq__(self, other):
        return True
    def __repr__(self):
        return "Foo()"
    def normal_method(self):
        pass
'''
    chunks = chunker.chunk_file("foo.py", code)
    cls = [c for c in chunks if c.metadata.unit_type == "class"][0]
    assert "__init__" in cls.metadata.dunder_methods
    assert "__eq__" in cls.metadata.dunder_methods
    assert "__repr__" in cls.metadata.dunder_methods
    assert "normal_method" not in cls.metadata.dunder_methods


def test_visibility_classification(chunker):
    code = '''
def public_func():
    pass

def _protected_func():
    pass

def __private_func():
    pass

def __dunder_func__():
    pass
'''
    chunks = chunker.chunk_file("vis.py", code)
    by_name = {c.metadata.unit_name: c for c in chunks if c.metadata.unit_type == "function"}
    assert by_name["public_func"].metadata.visibility == "public"
    assert by_name["_protected_func"].metadata.visibility == "protected"
    assert by_name["__private_func"].metadata.visibility == "private"
    assert by_name["__dunder_func__"].metadata.visibility == "dunder"


def test_enriched_content_includes_inherits(chunker):
    code = '''
class MyModel(BaseModel):
    """A model."""
    pass
'''
    chunks = chunker.chunk_file("model.py", code)
    cls = [c for c in chunks if c.metadata.unit_type == "class"][0]
    assert "# Inherits: BaseModel" in cls.enriched_content


def test_enriched_content_includes_decorators(chunker):
    code = '''
@dataclass
class Config:
    """Config."""
    name: str = "default"
'''
    chunks = chunker.chunk_file("config.py", code)
    cls = [c for c in chunks if c.metadata.unit_type == "class"][0]
    assert "# Decorators: @dataclass" in cls.enriched_content


def test_enriched_content_includes_return_type(chunker):
    code = '''
def compute(x: int) -> int:
    """Compute."""
    return x * 2
'''
    chunks = chunker.chunk_file("comp.py", code)
    func = [c for c in chunks if c.metadata.unit_type == "function"][0]
    assert "# Returns: int" in func.enriched_content


def test_enriched_content_includes_implements(chunker):
    code = '''
class Foo:
    """A class."""
    def __init__(self):
        pass
    def __eq__(self, other):
        return True
'''
    chunks = chunker.chunk_file("foo.py", code)
    cls = [c for c in chunks if c.metadata.unit_type == "class"][0]
    assert "# Implements: __init__, __eq__" in cls.enriched_content


def test_new_metadata_in_pinecone(chunker):
    code = '''
class MyModel(BaseModel):
    """A model."""
    def __init__(self):
        pass
'''
    chunks = chunker.chunk_file("model.py", code)
    cls = [c for c in chunks if c.metadata.unit_type == "class"][0]
    meta = cls.metadata.to_pinecone_metadata()
    assert meta["base_classes"] == ["BaseModel"]
    assert meta["dunder_methods"] == ["__init__"]


def test_split_large_class_propagates_metadata(chunker):
    """Ensure base_classes/decorators propagate to sub-chunks of split classes."""
    methods = []
    for i in range(300):
        methods.append(f'''
    def method_{i}(self):
        """Method {i} docstring that adds some length for splitting."""
        x_{i} = {i}
        y_{i} = x_{i} * 2 + {i} * 3 + {i} * 4 + {i} * 5
        z_{i} = y_{i} + x_{i} + {i} * 100 + {i} * 200 + {i} * 300
        w_{i} = z_{i} * x_{i} + y_{i} * z_{i} + {i} * 400 + {i} * 500
        return w_{i} + z_{i} + y_{i} + x_{i} + {i} * 600 + {i} * 700
''')
    code = f'''
class HugeModel(BaseModel):
    """A very large model."""

    def __init__(self):
        self.value = 0
{''.join(methods)}
'''
    chunks = chunker.chunk_file("huge.py", code)
    class_chunks = [c for c in chunks if c.metadata.unit_name == "HugeModel"]
    assert len(class_chunks) > 1
    # All sub-chunks should have base_classes propagated
    for chunk in class_chunks:
        assert "BaseModel" in chunk.metadata.base_classes
