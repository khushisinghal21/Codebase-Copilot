"""
Walk a repository and yield the paths of files worth indexing.

Respects .gitignore, filters by extension, and skips anything too
large to be useful as retrieval context (huge lockfiles, binaries, etc).
"""

import os
from pathlib import Path
from typing import Iterator, Optional

import pathspec

DEFAULT_EXTENSIONS = {".py"}  # Phase 1: start with one language
DEFAULT_MAX_FILE_SIZE = 1_000_000  # 1 MB
ALWAYS_IGNORE_DIRS = {
    ".git", "node_modules", "venv", ".venv", "__pycache__",
    "dist", "build", ".next", "target", ".mypy_cache", ".pytest_cache",
}


def _load_gitignore(repo_path: Path) -> Optional[pathspec.PathSpec]:
    gitignore_path = repo_path / ".gitignore"
    if not gitignore_path.exists():
        return None
    with open(gitignore_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    return pathspec.PathSpec.from_lines("gitwildmatch", lines)


def walk_repo(
    repo_path: str,
    extensions: Optional[set] = None,
    max_file_size: int = DEFAULT_MAX_FILE_SIZE,
) -> Iterator[Path]:
    """
    Yield Path objects for every file in repo_path worth indexing.
    """
    repo_path = Path(repo_path).resolve()
    extensions = extensions or DEFAULT_EXTENSIONS
    spec = _load_gitignore(repo_path)

    for root, dirs, files in os.walk(repo_path):
        # prune ignored directories in-place so os.walk doesn't descend into them
        dirs[:] = [d for d in dirs if d not in ALWAYS_IGNORE_DIRS]

        for filename in files:
            file_path = Path(root) / filename

            if file_path.suffix not in extensions:
                continue

            rel_path = file_path.relative_to(repo_path)

            if spec and spec.match_file(str(rel_path)):
                continue

            try:
                if file_path.stat().st_size > max_file_size:
                    continue
            except OSError:
                continue

            yield file_path


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "."
    count = 0
    for p in walk_repo(target):
        print(p)
        count += 1
    print(f"\n{count} files found", file=sys.stderr)
