"""
Phase 4: Post-generation citation validator.

After the LLM generates an answer, this module:
  1. Parses all path:line-line citations from the answer text.
  2. Checks each one against the chunks that were actually provided.
  3. Returns a validation report and a cleaned answer.

Why this matters: LLMs occasionally hallucinate citations — they invent
plausible-looking file paths or line numbers that don't exist in the
provided context. A hallucinated citation is worse than no citation
because it actively misleads the developer.
"""

import re
from pathlib import Path


# Matches patterns like:
#   /path/to/file.py:10-20
#   path/to/file.py:10-20
#   (path/to/file.py:10-20)
#   【path/to/file.py:10-20】   ← some models use Unicode brackets
_CITATION_RE = re.compile(
    r"[\(/【]?"
    r"([\w/.\-_]+\.py)"     # file path ending in .py
    r":(\d+)-(\d+)"         # :start_line-end_line
    r"[\)/】]?"
)


def _normalize_path(path_str: str, chunk_paths: list[str]) -> str | None:
    """
    Try to match a cited path (possibly relative) against the full paths
    of the provided chunks.
    """
    for chunk_path in chunk_paths:
        if chunk_path.endswith(path_str) or path_str in chunk_path:
            return chunk_path
    return None


def _ranges_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start <= b_end and b_start <= a_end


def validate_citations(answer: str, chunks: list[dict]) -> dict:
    """
    Validate all citations in an LLM-generated answer.

    Returns:
      {
        "valid":       list of (path, start, end) that matched a chunk,
        "hallucinated": list of (path, start, end) that didn't match,
        "flagged_answer": answer text with [?] appended to bad citations,
      }
    """
    chunk_paths = [c["file_path"] for c in chunks]
    citations = _CITATION_RE.findall(answer)

    valid = []
    hallucinated = []

    for path_str, start_str, end_str in citations:
        start = int(start_str)
        end = int(end_str)

        matched_path = _normalize_path(path_str, chunk_paths)
        if matched_path is None:
            hallucinated.append((path_str, start, end))
            continue

        # Check if line range overlaps with the matched chunk
        found = False
        for chunk in chunks:
            if chunk["file_path"] == matched_path:
                if _ranges_overlap(start, end, chunk["start_line"], chunk["end_line"]):
                    found = True
                    break

        if found:
            valid.append((path_str, start, end))
        else:
            hallucinated.append((path_str, start, end))

    # Append [?] after hallucinated citations in the answer text
    flagged = answer
    for path_str, start, end in hallucinated:
        pattern = f"{path_str}:{start}-{end}"
        flagged = flagged.replace(pattern, f"{pattern}[?]")

    return {
        "valid": valid,
        "hallucinated": hallucinated,
        "flagged_answer": flagged,
    }


def format_validation_report(report: dict) -> str:
    lines = []
    if report["hallucinated"]:
        lines.append(f"⚠️  {len(report['hallucinated'])} hallucinated citation(s) detected and flagged with [?]:")
        for path, start, end in report["hallucinated"]:
            lines.append(f"   • {path}:{start}-{end}")
    else:
        lines.append(f"✅ All {len(report['valid'])} citation(s) verified against provided chunks.")
    return "\n".join(lines)
