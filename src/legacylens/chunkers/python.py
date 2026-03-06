"""Python-specific chunker using the ast module for reliable parsing.

Chunks Python files at top-level unit boundaries (functions, classes).
Falls back to whole-file chunking for files with syntax errors.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from legacylens.chunkers.base import BaseChunker, Chunk, ChunkMetadata, classify_module_tier, classify_visibility

_CHARS_PER_TOKEN = 4
_MAX_CHUNK_TOKENS = 15_000
_MAX_CHUNK_CHARS = _MAX_CHUNK_TOKENS * _CHARS_PER_TOKEN
_PREAMBLE_MAX_CHARS = 4000 * _CHARS_PER_TOKEN


def _get_docstring_summary(node: ast.AST) -> str:
    """Extract first 1-2 sentences of a node's docstring."""
    doc = ast.get_docstring(node)
    if not doc:
        return ""
    # Take first 1-2 sentences
    sentences = re.split(r"(?<=[.!?])\s+", doc.strip())
    return " ".join(sentences[:2]).strip()


def _get_start_line(node: ast.AST) -> int:
    """Get the start line of a node, including decorators."""
    if hasattr(node, "decorator_list") and node.decorator_list:
        return node.decorator_list[0].lineno
    return node.lineno


def _extract_calls(node: ast.AST) -> list[str]:
    """Extract function/method call names from an AST node's body."""
    calls: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            if isinstance(child.func, ast.Name):
                calls.add(child.func.id)
            elif isinstance(child.func, ast.Attribute):
                calls.add(child.func.attr)
    return sorted(calls)


def _extract_imports(tree: ast.AST) -> list[str]:
    """Extract imported module names from import statements."""
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.add(node.module.split(".")[0])
    return sorted(modules)


def _format_params(args: ast.arguments) -> list[str]:
    """Extract parameter names with type annotations as strings."""
    params: list[str] = []
    all_args = args.posonlyargs + args.args + args.kwonlyargs
    if args.vararg:
        all_args_names = [a.arg for a in all_args] + [f"*{args.vararg.arg}"]
    else:
        all_args_names = [a.arg for a in all_args]
    if args.kwarg:
        all_args_names.append(f"**{args.kwarg.arg}")

    for arg in all_args:
        if arg.arg == "self" or arg.arg == "cls":
            continue
        if arg.annotation:
            try:
                params.append(f"{arg.arg}: {ast.unparse(arg.annotation)}")
            except Exception:
                params.append(arg.arg)
        else:
            params.append(arg.arg)
    return params


def _extract_base_classes(node: ast.ClassDef) -> list[str]:
    """Extract base class names from a ClassDef node."""
    bases: list[str] = []
    for base in node.bases:
        try:
            bases.append(ast.unparse(base))
        except Exception:
            pass
    return bases


def _extract_decorators(node: ast.AST) -> list[str]:
    """Extract decorator names from a decorated node."""
    decorators: list[str] = []
    if hasattr(node, "decorator_list"):
        for dec in node.decorator_list:
            try:
                decorators.append(ast.unparse(dec))
            except Exception:
                pass
    return decorators


def _extract_return_type(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Extract return type annotation if present."""
    if node.returns:
        try:
            return ast.unparse(node.returns)
        except Exception:
            pass
    return ""


def _extract_dunder_methods(node: ast.ClassDef) -> list[str]:
    """List dunder method names in a class."""
    dunders: list[str] = []
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if child.name.startswith("__") and child.name.endswith("__"):
                dunders.append(child.name)
    return dunders


def _classify_unit_type(node: ast.AST, is_method: bool = False) -> str:
    """Determine unit_type string from an AST node."""
    if isinstance(node, ast.ClassDef):
        return "class"
    if isinstance(node, ast.AsyncFunctionDef):
        return "async_method" if is_method else "async_function"
    if isinstance(node, ast.FunctionDef):
        return "method" if is_method else "function"
    return "module"


def _build_enriched_content(
    source: str,
    name: str,
    unit_type: str,
    purpose: str,
    calls: list[str],
    file_path: str,
    start_line: int,
    end_line: int,
    module_tier: str = "current",
    base_classes: list[str] | None = None,
    decorators: list[str] | None = None,
    return_type: str = "",
    dunder_methods: list[str] | None = None,
) -> str:
    """Build enriched content with NL summary prefix."""
    type_label = unit_type.replace("_", " ").title()
    tier_prefix = ""
    if module_tier == "legacy":
        tier_prefix = "[LEGACY] "
    elif module_tier == "deprecated":
        tier_prefix = "[DEPRECATED] "
    elif module_tier == "internal":
        tier_prefix = "[INTERNAL] "
    header_parts = [f"# {tier_prefix}{name}", f"- Python {type_label}"]

    if base_classes:
        header_parts.append(f"# Inherits: {', '.join(base_classes)}")
    if decorators:
        header_parts.append(f"# Decorators: @{', @'.join(decorators)}")
    if purpose:
        header_parts.append(f"# Purpose: {purpose}")
    if return_type:
        header_parts.append(f"# Returns: {return_type}")
    if dunder_methods:
        header_parts.append(f"# Implements: {', '.join(dunder_methods)}")

    meta_parts = [f"File: {file_path}", f"Lines: {start_line}-{end_line}"]
    if calls:
        meta_parts.append(f"Calls: {', '.join(calls[:10])}")
    header_parts.append(f"# {' | '.join(meta_parts)}")

    header = "\n".join(header_parts)
    return f"{header}\n\n{source}"


class PythonChunker(BaseChunker):
    """Chunker for Python source files using the ast module."""

    def supported_extensions(self) -> list[str]:
        return [".py", ".pyi"]

    def chunk_file(self, file_path: str, content: str) -> list[Chunk]:
        if not content.strip():
            return []

        lines = content.splitlines()
        total_lines = len(lines)

        try:
            tree = ast.parse(content, filename=file_path)
        except SyntaxError:
            return self._fallback_chunk(file_path, content, lines, total_lines)

        imports = _extract_imports(tree)
        chunks: list[Chunk] = []

        # Separate top-level nodes into preamble vs units
        preamble_nodes: list[ast.AST] = []
        unit_nodes: list[ast.AST] = []

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                unit_nodes.append(node)
            else:
                preamble_nodes.append(node)

        # Module preamble chunk (imports, constants, module docstring)
        if preamble_nodes:
            preamble_chunk = self._make_preamble_chunk(
                file_path, content, lines, total_lines, tree, preamble_nodes, imports,
            )
            if preamble_chunk:
                chunks.append(preamble_chunk)

        # If no top-level units but there IS code outside functions/classes
        if not unit_nodes:
            # Check if there's actual code (not just imports/constants)
            has_code = any(
                isinstance(n, (ast.Expr, ast.For, ast.While, ast.If, ast.With, ast.Try, ast.Assert))
                and not (isinstance(n, ast.Expr) and isinstance(n.value, (ast.Constant,)))
                for n in preamble_nodes
            )
            if has_code and not chunks:
                # Script with no functions — single module_code chunk
                return [self._make_module_code_chunk(file_path, content, lines, total_lines, imports)]
            return chunks

        # One chunk per top-level unit
        for node in unit_nodes:
            start = _get_start_line(node) - 1  # 0-based
            end = node.end_lineno  # 1-based inclusive, so lines[start:end]
            source = "\n".join(lines[start:end])
            name = node.name
            unit_type = _classify_unit_type(node)
            purpose = _get_docstring_summary(node)
            calls = _extract_calls(node)
            decos = _extract_decorators(node)
            visibility = classify_visibility(name)

            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                params = _format_params(node.args)
                ret_type = _extract_return_type(node)
                base_cls: list[str] = []
                dunders: list[str] = []
            else:
                params = []
                ret_type = ""
                base_cls = []
                dunders = []

            if isinstance(node, ast.ClassDef):
                base_cls = _extract_base_classes(node)
                dunders = _extract_dunder_methods(node)

            # Large class splitting
            if isinstance(node, ast.ClassDef) and len(source) > _MAX_CHUNK_CHARS:
                chunks.extend(
                    self._split_large_class(file_path, lines, node, imports)
                )
                continue

            tier = classify_module_tier(file_path)
            enriched = _build_enriched_content(
                source, name, unit_type, purpose, calls,
                file_path, start + 1, end, module_tier=tier,
                base_classes=base_cls, decorators=decos,
                return_type=ret_type, dunder_methods=dunders,
            )
            metadata = ChunkMetadata(
                file_path=file_path,
                start_line=start + 1,
                end_line=end,
                unit_name=name,
                unit_type=unit_type,
                language="python",
                parameters=params,
                calls=calls,
                external_deps=imports,
                uses=imports,
                purpose=purpose,
                module_tier=tier,
                base_classes=base_cls,
                decorators=decos,
                return_type=ret_type,
                visibility=visibility,
                dunder_methods=dunders,
            )
            chunks.append(Chunk(content=source, enriched_content=enriched, metadata=metadata))

        return chunks

    def _make_preamble_chunk(
        self,
        file_path: str,
        content: str,
        lines: list[str],
        total_lines: int,
        tree: ast.Module,
        preamble_nodes: list[ast.AST],
        imports: list[str],
    ) -> Chunk | None:
        """Create a preamble chunk from module-level non-unit nodes."""
        # Find the line range of preamble nodes
        preamble_lines: list[int] = []
        for node in preamble_nodes:
            if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
                for ln in range(node.lineno, node.end_lineno + 1):
                    preamble_lines.append(ln)

        if not preamble_lines:
            return None

        start = min(preamble_lines)
        end = max(preamble_lines)
        source = "\n".join(lines[start - 1 : end])

        if not source.strip():
            return None

        name = Path(file_path).stem
        purpose = _get_docstring_summary(tree)
        tier = classify_module_tier(file_path)

        enriched = _build_enriched_content(
            source, name, "module", purpose, [],
            file_path, start, end, module_tier=tier,
        )
        metadata = ChunkMetadata(
            file_path=file_path,
            start_line=start,
            end_line=end,
            unit_name=name,
            unit_type="module",
            language="python",
            parameters=[],
            calls=[],
            external_deps=imports,
            uses=imports,
            purpose=purpose,
            module_tier=tier,
        )
        return Chunk(content=source, enriched_content=enriched, metadata=metadata)

    def _make_module_code_chunk(
        self,
        file_path: str,
        content: str,
        lines: list[str],
        total_lines: int,
        imports: list[str],
    ) -> Chunk:
        """Create a single chunk for script-style files with no functions/classes."""
        name = Path(file_path).stem
        tier = classify_module_tier(file_path)
        enriched = _build_enriched_content(
            content, name, "module_code", "", [],
            file_path, 1, total_lines, module_tier=tier,
        )
        metadata = ChunkMetadata(
            file_path=file_path,
            start_line=1,
            end_line=total_lines,
            unit_name=name,
            unit_type="module_code",
            language="python",
            parameters=[],
            calls=_extract_calls(ast.parse(content)),
            external_deps=imports,
            uses=imports,
            purpose="",
            module_tier=tier,
        )
        return Chunk(content=content, enriched_content=enriched, metadata=metadata)

    def _split_large_class(
        self,
        file_path: str,
        lines: list[str],
        node: ast.ClassDef,
        imports: list[str],
    ) -> list[Chunk]:
        """Split a large class into multiple chunks at method boundaries."""
        chunks: list[Chunk] = []
        class_start = _get_start_line(node) - 1  # 0-based
        class_end = node.end_lineno  # 1-based
        purpose = _get_docstring_summary(node)
        base_cls = _extract_base_classes(node)
        decos = _extract_decorators(node)
        dunders = _extract_dunder_methods(node)
        visibility = classify_visibility(node.name)

        # Collect methods
        methods: list[ast.AST] = []
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(child)

        tier = classify_module_tier(file_path)

        if not methods:
            # No methods to split on — just return one chunk
            source = "\n".join(lines[class_start:class_end])
            enriched = _build_enriched_content(
                source, node.name, "class", purpose, _extract_calls(node),
                file_path, class_start + 1, class_end, module_tier=tier,
                base_classes=base_cls, decorators=decos, dunder_methods=dunders,
            )
            metadata = ChunkMetadata(
                file_path=file_path,
                start_line=class_start + 1,
                end_line=class_end,
                unit_name=node.name,
                unit_type="class",
                language="python",
                parameters=[],
                calls=_extract_calls(node),
                external_deps=imports,
                uses=imports,
                purpose=purpose,
                module_tier=tier,
                base_classes=base_cls,
                decorators=decos,
                visibility=visibility,
                dunder_methods=dunders,
            )
            return [Chunk(content=source, enriched_content=enriched, metadata=metadata)]

        # Chunk 0: class preamble (up to first method)
        first_method_start = _get_start_line(methods[0]) - 1
        preamble_source = "\n".join(lines[class_start:first_method_start])
        # Class signature for prefixing later chunks
        class_sig = lines[class_start].rstrip()

        # Group methods into chunks that fit within budget
        available = _MAX_CHUNK_CHARS - len(preamble_source) - 500
        if available < 1000:
            available = _MAX_CHUNK_CHARS // 2

        method_groups: list[list[ast.AST]] = [[]]
        current_size = 0

        for method in methods:
            m_start = _get_start_line(method) - 1
            m_end = method.end_lineno
            m_source = "\n".join(lines[m_start:m_end])
            if current_size + len(m_source) > available and method_groups[-1]:
                method_groups.append([])
                current_size = 0
            method_groups[-1].append(method)
            current_size += len(m_source)

        # Build preamble chunk (index 0)
        chunk_total = len(method_groups) + 1
        preamble_enriched = _build_enriched_content(
            preamble_source, node.name, "class", purpose, [],
            file_path, class_start + 1, first_method_start, module_tier=tier,
            base_classes=base_cls, decorators=decos, dunder_methods=dunders,
        )
        preamble_meta = ChunkMetadata(
            file_path=file_path,
            start_line=class_start + 1,
            end_line=first_method_start,
            unit_name=node.name,
            unit_type="class",
            language="python",
            parameters=[],
            calls=[],
            external_deps=imports,
            uses=imports,
            purpose=purpose,
            module_tier=tier,
            base_classes=base_cls,
            decorators=decos,
            visibility=visibility,
            dunder_methods=dunders,
            chunk_index=1,
            chunk_total=chunk_total,
        )
        chunks.append(Chunk(content=preamble_source, enriched_content=preamble_enriched, metadata=preamble_meta))

        # Build method group chunks
        for idx, group in enumerate(method_groups):
            g_start = _get_start_line(group[0]) - 1
            g_end = group[-1].end_lineno
            group_source = "\n".join(lines[g_start:g_end])
            # Prefix with truncated class signature for context
            prefixed = f"{class_sig}\n    ...\n{group_source}"
            method_names = [m.name for m in group if hasattr(m, "name")]
            group_calls = []
            for m in group:
                group_calls.extend(_extract_calls(m))

            enriched = _build_enriched_content(
                prefixed, node.name, "class", purpose, sorted(set(group_calls)),
                file_path, g_start + 1, g_end, module_tier=tier,
                base_classes=base_cls, decorators=decos, dunder_methods=dunders,
            )
            meta = ChunkMetadata(
                file_path=file_path,
                start_line=g_start + 1,
                end_line=g_end,
                unit_name=node.name,
                unit_type="class",
                language="python",
                parameters=method_names,
                calls=sorted(set(group_calls)),
                external_deps=imports,
                uses=imports,
                purpose=purpose,
                module_tier=tier,
                base_classes=base_cls,
                decorators=decos,
                visibility=visibility,
                dunder_methods=dunders,
                chunk_index=idx + 2,
                chunk_total=chunk_total,
            )
            chunks.append(Chunk(content=group_source, enriched_content=enriched, metadata=meta))

        return chunks

    def _fallback_chunk(
        self,
        file_path: str,
        content: str,
        lines: list[str],
        total_lines: int,
    ) -> list[Chunk]:
        """Fallback for files with syntax errors — treat whole file as one chunk."""
        name = Path(file_path).stem
        # Try to extract a name via regex
        match = re.search(r"(?:class|def)\s+(\w+)", content)
        if match:
            name = match.group(1)

        tier = classify_module_tier(file_path)
        enriched = _build_enriched_content(
            content, name, "module", "", [],
            file_path, 1, total_lines, module_tier=tier,
        )
        metadata = ChunkMetadata(
            file_path=file_path,
            start_line=1,
            end_line=total_lines,
            unit_name=name,
            unit_type="module",
            language="python",
            parameters=[],
            calls=[],
            external_deps=[],
            uses=[],
            purpose="",
            module_tier=tier,
        )
        return [Chunk(content=content, enriched_content=enriched, metadata=metadata)]
