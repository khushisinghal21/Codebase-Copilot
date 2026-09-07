"""
Phase 1 chunking: fixed-size line windows with overlap.

Every chunk carries {file_path, start_line, end_line, text} so retrieval
results can always be cited as path:line. This gets replaced by
AST-based (tree-sitter) chunking in Phase 2 - the interface below
(chunk_file -> list[dict]) is designed to stay the same so nothing
downstream has to change when you swap it out.
"""

from pathlib import Path
from typing import Iterator

CHUNK_LINES = 300
OVERLAP_LINES = 50


def chunk_file(
    file_path: Path,
    chunk_lines: int = CHUNK_LINES,
    overlap_lines: int = OVERLAP_LINES,
) -> list[dict]:
    """
    Split a single file into overlapping fixed-size line-window chunks.
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except OSError:
        return []

    if not lines:
        return []

    chunks = []
    step = max(chunk_lines - overlap_lines, 1)
    start = 0
    while start < len(lines):
        end = min(start + chunk_lines, len(lines))
        text = "".join(lines[start:end])
        if text.strip():
            chunks.append(
                {
                    "file_path": str(file_path),
                    "start_line": start + 1,  # 1-indexed for human-readable citations
                    "end_line": end,
                    "text": text,
                }
            )
        if end == len(lines):
            break
        start += step

    return chunks


def chunk_repo(file_paths: Iterator[Path], **kwargs) -> Iterator[dict]:
    """
    Chunk every file yielded by walk_repo(). Yields chunk dicts one at a time
    so you never have to hold the whole repo's chunks in memory at once.
    """
    for file_path in file_paths:
        for chunk in chunk_file(file_path, **kwargs):
            yield chunk


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from ingest.walk_repo import walk_repo

    target = sys.argv[1] if len(sys.argv) > 1 else "."
    total = 0
    for c in chunk_repo(walk_repo(target)):
        total += 1
        if total <= 3:
            print(f"{c['file_path']}:{c['start_line']}-{c['end_line']}")
    print(f"\n{total} chunks created", file=sys.stderr)
