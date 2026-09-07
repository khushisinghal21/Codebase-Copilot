"""
Phase 2 chunking: AST-aware chunking using tree-sitter.

Replaces the fixed 300-line windows in chunk.py with chunks that
respect syntactic boundaries — functions, classes, and methods.

Interface is identical to chunk.py (chunk_file -> list[dict]) so
cli.py and everything downstream needs zero changes when you swap it in.

Each chunk gets extra metadata vs Phase 1:
  - symbol_name: the function/class/method name
  - symbol_type: 'function' | 'class' | 'method'
  - parent_class: set for methods, None otherwise

Falls back to fixed-size chunking for files that can't be parsed.
"""

from pathlib import Path
from typing import Iterator

import tree_sitter_python as tspython
from tree_sitter import Language, Parser

# ── parser setup ──────────────────────────────────────────────────────────────
PY_LANGUAGE = Language(tspython.language())
_parser = Parser(PY_LANGUAGE)


# ── helpers ───────────────────────────────────────────────────────────────────

def _node_text(node, lines: list[str]) -> str:
    """Extract the source text for a node using its line range."""
    start = node.start_point[0]   # 0-indexed
    end = node.end_point[0]       # 0-indexed, inclusive
    return "".join(lines[start : end + 1])


def _get_name(node) -> str:
    """Return the identifier name for a function_definition or class_definition."""
    name_node = node.child_by_field_name("name")
    if name_node:
        return name_node.text.decode("utf-8", errors="ignore")
    return "unknown"


def _make_chunk(
    file_path: Path,
    node,
    lines: list[str],
    symbol_name: str,
    symbol_type: str,
    parent_class: str | None = None,
) -> dict:
    start_line = node.start_point[0]  # 0-indexed
    end_line = node.end_point[0]      # 0-indexed, inclusive
    return {
        "file_path": str(file_path),
        "start_line": start_line + 1,          # 1-indexed for citations
        "end_line": end_line + 1,
        "text": _node_text(node, lines),
        "symbol_name": symbol_name,
        "symbol_type": symbol_type,
        "parent_class": parent_class,
    }


# ── core chunking logic ───────────────────────────────────────────────────────

def _chunk_class(file_path: Path, class_node, lines: list[str]) -> list[dict]:
    """
    For a class, emit:
      1. One class-level chunk (the whole class body) - gives context about the class.
      2. One chunk per method - gives fine-grained access.
    """
    class_name = _get_name(class_node)
    chunks = []

    # 1. Whole class chunk
    chunks.append(
        _make_chunk(file_path, class_node, lines, class_name, "class")
    )

    # 2. Method chunks (function_definition nodes inside the class body)
    body = class_node.child_by_field_name("body")
    if body:
        for child in body.children:
            if child.type == "function_definition":
                method_name = _get_name(child)
                chunks.append(
                    _make_chunk(
                        file_path, child, lines,
                        symbol_name=method_name,
                        symbol_type="method",
                        parent_class=class_name,
                    )
                )
    return chunks


def chunk_file(file_path: Path) -> list[dict]:
    """
    Parse a Python file with tree-sitter and chunk at function/class boundaries.
    Falls back to fixed-size chunking on parse failure.
    """
    try:
        source = file_path.read_bytes()
    except OSError:
        return []

    if not source.strip():
        return []

    lines = source.decode("utf-8", errors="ignore").splitlines(keepends=True)

    try:
        tree = _parser.parse(source)
    except Exception:
        from ingest.chunk import chunk_file as fixed_chunk_file
        return fixed_chunk_file(file_path)

    chunks: list[dict] = []
    root = tree.root_node

    # Walk top-level nodes only
    for node in root.children:
        if node.type == "function_definition":
            name = _get_name(node)
            chunks.append(_make_chunk(file_path, node, lines, name, "function"))

        elif node.type == "class_definition":
            chunks.extend(_chunk_class(file_path, node, lines))

        elif node.type == "decorated_definition":
            inner = node.child_by_field_name("definition")
            if inner is None:
                for child in reversed(node.children):
                    if child.type in ("function_definition", "class_definition"):
                        inner = child
                        break

            if inner and inner.type == "function_definition":
                name = _get_name(inner)
                chunks.append(_make_chunk(file_path, node, lines, name, "function"))
            elif inner and inner.type == "class_definition":
                chunks.extend(_chunk_class(file_path, inner, lines))

    # If tree-sitter found nothing, fall back to fixed chunking
    if not chunks:
        from ingest.chunk import chunk_file as fixed_chunk_file
        return fixed_chunk_file(file_path)

    return chunks


def chunk_repo(file_paths: Iterator[Path]) -> Iterator[dict]:
    """Chunk every file; yields chunk dicts one at a time."""
    for file_path in file_paths:
        for chunk in chunk_file(file_path):
            yield chunk


# ── quick smoke test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from ingest.walk_repo import walk_repo

    target = sys.argv[1] if len(sys.argv) > 1 else "."
    total = 0
    for c in chunk_repo(walk_repo(target)):
        total += 1
        if total <= 5:
            label = f"[{c['symbol_type']}] {c.get('parent_class', '')}.{c['symbol_name']}".strip(".")
            print(f"{c['file_path']}:{c['start_line']}-{c['end_line']}  {label}")
    print(f"\n{total} AST chunks created", file=sys.stderr)
