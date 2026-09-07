"""
Phase 4: Extract function call edges from Python source using tree-sitter.

For every function/method in a file, finds all the functions it calls
and emits (caller_file, caller_symbol, callee_name) edges.

Handles:
  foo()           → callee = "foo"
  self.bar()      → callee = "bar"
  obj.baz()       → callee = "baz"
  module.func()   → callee = "func"

We record the bare callee name (not the module) because we don't have
import resolution — we match callees against the symbol table by name.
"""

from pathlib import Path
from typing import Iterator

import tree_sitter_python as tspython
from tree_sitter import Language, Parser

PY_LANGUAGE = Language(tspython.language())
_parser = Parser(PY_LANGUAGE)


# ── call extraction ───────────────────────────────────────────────────────────

def _collect_calls(node) -> list[str]:
    """
    Recursively walk a subtree and collect the bare name of every call target.
    Returns deduplicated list.
    """
    names: set[str] = set()

    def _walk(n):
        if n.type == "call":
            func = n.child_by_field_name("function")
            if func is not None:
                if func.type == "identifier":
                    names.add(func.text.decode("utf-8", errors="ignore"))
                elif func.type == "attribute":
                    attr = func.child_by_field_name("attribute")
                    if attr:
                        names.add(attr.text.decode("utf-8", errors="ignore"))
        for child in n.children:
            _walk(child)

    _walk(node)
    return list(names)


def _process_function(file_path: str, node, symbol_name: str) -> list[tuple]:
    """Emit call edges for a single function/method node."""
    callees = _collect_calls(node)
    edges = []
    for callee in callees:
        if callee != symbol_name:   # skip trivial self-recursion
            edges.append((file_path, symbol_name, callee))
    return edges


def extract_calls(file_path: Path) -> list[tuple]:
    """
    Parse a Python file and return all call edges as
    (caller_file, caller_symbol, callee_name) tuples.
    """
    try:
        source = file_path.read_bytes()
    except OSError:
        return []

    if not source.strip():
        return []

    try:
        tree = _parser.parse(source)
    except Exception:
        return []

    edges: list[tuple] = []
    file_str = str(file_path)

    def walk(node, parent_class: str | None = None):
        if node.type == "function_definition":
            name_node = node.child_by_field_name("name")
            if name_node:
                raw_name = name_node.text.decode("utf-8", errors="ignore")
                # Use bare name for matching (parent_class stored separately in symbols table)
                edges.extend(_process_function(file_str, node, raw_name))

        elif node.type == "class_definition":
            name_node = node.child_by_field_name("name")
            class_name = (
                name_node.text.decode("utf-8", errors="ignore")
                if name_node
                else None
            )
            body = node.child_by_field_name("body")
            if body:
                for child in body.children:
                    walk(child, parent_class=class_name)
            return  # body already walked above

        for child in node.children:
            walk(child, parent_class)

    walk(tree.root_node)
    return edges


def build_call_graph(file_paths: Iterator[Path]) -> list[tuple]:
    """Extract call edges from every file in the iterator."""
    all_edges: list[tuple] = []
    for fp in file_paths:
        all_edges.extend(extract_calls(fp))
    return all_edges


# ── smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from ingest.walk_repo import walk_repo

    target = sys.argv[1] if len(sys.argv) > 1 else "."
    edges = build_call_graph(walk_repo(target))
    print(f"{len(edges)} call edges extracted\n")
    for caller_file, caller, callee in edges[:20]:
        short = caller_file.split("/")[-1]
        print(f"  {short} :: {caller}  →  {callee}")
