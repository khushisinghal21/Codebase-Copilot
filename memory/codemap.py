"""
Read the current state of the code: files, symbols, docstrings, call edges.

`dream.py` uses this as the environment to verify memory against. Symbols come
from the same tree-sitter chunker the index uses, so a note's (file, symbol,
line range) always lines up with a real chunk. Docstrings and signatures come
from Python's `ast` module.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

from ingest.call_graph import extract_calls
from ingest.chunk_ast import chunk_file
from ingest.walk_repo import walk_repo


@dataclass
class SymbolInfo:
    name: str
    symbol_type: str            # function | class | method
    parent_class: str | None
    start_line: int
    end_line: int
    signature: str = ""
    docstring: str = ""
    calls: list[str] = field(default_factory=list)


@dataclass
class FileInfo:
    abs_path: str
    rel_path: str
    module_doc: str
    n_bytes: int
    symbols: dict[str, SymbolInfo]   # keyed by symbol name


def _first_paragraph(text: str, max_chars: int = 400) -> str:
    para = text.strip().split("\n\n")[0]
    para = " ".join(para.split())
    return para[:max_chars]


def _ast_details(source: str) -> tuple[str, dict[tuple[int, str], tuple[str, str]]]:
    """Return (module docstring, {(lineno, name): (signature, docstring)})."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return "", {}
    details: dict[tuple[int, str], tuple[str, str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            try:
                sig = f"def {node.name}({ast.unparse(node.args)})"
            except Exception:
                sig = f"def {node.name}(...)"
        elif isinstance(node, ast.ClassDef):
            bases = ", ".join(ast.unparse(b) for b in node.bases)
            sig = f"class {node.name}({bases})" if bases else f"class {node.name}"
        else:
            continue
        details[(node.lineno, node.name)] = (sig, ast.get_docstring(node) or "")
    return ast.get_docstring(tree) or "", details


def load_codemap(repo_path: str) -> dict[str, FileInfo]:
    """Map rel_path -> FileInfo for every indexable file in the repo."""
    root = Path(repo_path).resolve()
    files: dict[str, FileInfo] = {}

    for path in walk_repo(str(root)):
        try:
            source = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        rel = str(path.relative_to(root))
        module_doc, details = _ast_details(source)

        calls_by_symbol: dict[str, list[str]] = {}
        for _, caller, callee in extract_calls(path):
            calls_by_symbol.setdefault(caller, [])
            if callee not in calls_by_symbol[caller]:
                calls_by_symbol[caller].append(callee)

        symbols: dict[str, SymbolInfo] = {}
        for c in chunk_file(path):
            name = c.get("symbol_name")
            if not name or name in symbols:
                continue
            # decorated definitions start on the decorator line; ast uses the def line
            sig, doc = "", ""
            for (lineno, n), (s, d) in details.items():
                if n == name and c["start_line"] <= lineno <= c["end_line"]:
                    sig, doc = s, d
                    break
            symbols[name] = SymbolInfo(
                name=name,
                symbol_type=c.get("symbol_type") or "chunk",
                parent_class=c.get("parent_class"),
                start_line=c["start_line"],
                end_line=c["end_line"],
                signature=sig,
                docstring=doc,
                calls=calls_by_symbol.get(name, []),
            )

        files[rel] = FileInfo(
            abs_path=str(path),
            rel_path=rel,
            module_doc=module_doc,
            n_bytes=len(source.encode("utf-8")),
            symbols=symbols,
        )
    return files


def to_rel(file_path: str, codemap: dict[str, FileInfo]) -> str | None:
    """Resolve an absolute or relative path to a rel_path known to the codemap."""
    if file_path in codemap:
        return file_path
    for rel, info in codemap.items():
        if info.abs_path == file_path or file_path.endswith("/" + rel):
            return rel
    return None
