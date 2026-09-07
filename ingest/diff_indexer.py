"""
Phase 5: Incremental re-indexing via git diff.

Instead of a full rebuild on every code change, this module:
  1. Reads the last indexed commit SHA from the meta store.
  2. Runs `git diff <last_commit> HEAD --name-status` to find changes.
  3. Returns which files to delete (M/D) and which to (re-)index (M/A).

This means a push that touches 3 files re-indexes 3 files, not 10,000.
"""

import subprocess
from pathlib import Path
from dataclasses import dataclass

DEFAULT_EXTENSIONS = {".py"}


@dataclass
class DiffResult:
    """Files that need action after a git diff."""
    to_delete: list[Path]    # files with old chunks to remove (modified + deleted)
    to_index: list[Path]     # files to (re-)chunk and embed (modified + added)
    old_commit: str | None   # the commit we diffed from
    new_commit: str | None   # HEAD (the commit we're now indexing to)


def get_current_commit(repo_path: str) -> str | None:
    """Return the current HEAD commit SHA, or None if not a git repo."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_path,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return result.stdout.strip()
    return None


def get_changed_files(
    repo_path: str,
    since_commit: str,
    extensions: set | None = None,
) -> DiffResult:
    """
    Return files changed between since_commit and HEAD.

    Uses `git diff --name-status` which gives:
      M  path/to/file.py   (modified)
      A  path/to/file.py   (added)
      D  path/to/file.py   (deleted)
    """
    extensions = extensions or DEFAULT_EXTENSIONS
    new_commit = get_current_commit(repo_path)

    result = subprocess.run(
        ["git", "diff", "--name-status", since_commit, "HEAD"],
        cwd=repo_path,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        # git diff failed (e.g. commit no longer exists) — signal full rebuild
        return DiffResult([], [], since_commit, new_commit)

    to_delete: list[Path] = []
    to_index: list[Path] = []

    for line in result.stdout.strip().splitlines():
        if not line:
            continue

        parts = line.split("\t", 1)
        if len(parts) < 2:
            continue

        status, rel_path = parts[0].strip(), parts[1].strip()
        path = Path(repo_path) / rel_path

        if path.suffix not in extensions:
            continue

        if status.startswith("M"):        # modified
            to_delete.append(path)
            if path.exists():
                to_index.append(path)

        elif status.startswith("A"):      # added
            if path.exists():
                to_index.append(path)

        elif status.startswith("D"):      # deleted
            to_delete.append(path)

        elif status.startswith("R"):      # renamed: "R\told\tnew"
            # parts[1] might be "old\tnew"
            sub = rel_path.split("\t")
            old_path = Path(repo_path) / sub[0]
            new_path = Path(repo_path) / sub[-1]
            if old_path.suffix in extensions:
                to_delete.append(old_path)
            if new_path.suffix in extensions and new_path.exists():
                to_index.append(new_path)

    return DiffResult(
        to_delete=to_delete,
        to_index=to_index,
        old_commit=since_commit,
        new_commit=new_commit,
    )


def is_git_repo(repo_path: str) -> bool:
    """Check whether the given path is inside a git repository."""
    result = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=repo_path,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


# ── smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    repo = sys.argv[1] if len(sys.argv) > 1 else "."
    commit = get_current_commit(repo)
    print(f"Current HEAD: {commit}")
    print(f"Is git repo: {is_git_repo(repo)}")
