"""
Phase 6: Parse Python import statements to detect cross-repo dependencies.

Strategy:
  `import X.Y.Z`        →  top_module = "X"
  `from X.Y import Z`   →  top_module = "X"

A cross-repo edge is emitted when top_module matches a registered repo_id.

Example:
  Repo "backend" has:   from shared_lib.auth import verify_token
  Repo "shared_lib" registered → cross-repo edge:
    (backend, verify_token)  →  (shared_lib, verify_token)

This doesn't require import resolution — just name matching.
"""

from pathlib import Path

import tree_sitter_python as tspython
from tree_sitter import Language, Parser

PY_LANGUAGE = Language(tspython.language())
_parser = Parser(PY_LANGUAGE)


def _extract_top_modules(tree_root) -> list[str]:
    """Walk AST and extract top-level module names from all import statements."""
    modules: set[str] = set()

    def walk(node):
        # import X.Y.Z  →  module_name = "X.Y.Z"
        if node.type == "import_statement":
            for child in node.children:
                if child.type == "dotted_name":
                    top = child.children[0].text.decode("utf-8", errors="ignore")
                    modules.add(top)
                elif child.type == "aliased_import":
                    name_node = child.child_by_field_name("name")
                    if name_node and name_node.type == "dotted_name":
                        top = name_node.children[0].text.decode("utf-8", errors="ignore")
                        modules.add(top)

        # from X.Y import Z  →  module = "X.Y", top = "X"
        elif node.type == "import_from_statement":
            mod = node.child_by_field_name("module_name")
            if mod:
                # module_name can be dotted_name or relative_import
                if mod.type == "dotted_name":
                    top = mod.children[0].text.decode("utf-8", errors="ignore")
                    modules.add(top)
                elif mod.type == "relative_import":
                    # relative imports (from . import X) — skip
                    pass

        for child in node.children:
            walk(child)

    walk(tree_root)
    return list(modules)


def parse_imports(file_path: Path) -> list[str]:
    """
    Return the top-level module names imported by this file.
    Returns [] on parse failure.
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
    return _extract_top_modules(tree.root_node)


def find_cross_repo_imports(
    file_path: Path,
    registered_repo_ids: list[str],
) -> list[str]:
    """
    Return repo_ids that this file imports from.

    e.g. file has `from shared_lib import X` and "shared_lib" is a
    registered repo → returns ["shared_lib"]
    """
    top_modules = parse_imports(file_path)
    repo_id_set = set(registered_repo_ids)
    return [m for m in top_modules if m in repo_id_set or m.replace("-", "_") in repo_id_set]


# ── smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    fp = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("cli.py")
    mods = parse_imports(fp)
    print(f"Imports in {fp.name}:")
    for m in sorted(mods):
        print(f"  {m}")
