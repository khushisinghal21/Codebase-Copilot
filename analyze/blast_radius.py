"""
Phase 7a: Blast Radius Analysis.

Given a symbol name, estimates the impact of changing it by performing a
breadth-first traversal of the call graph — finding every symbol that
(transitively) calls the target.

Output:
  - Direct callers (depth 1)
  - Transitive callers (depth 2+)
  - Affected files and repos
  - Impact score (% of indexed codebase affected)
  - LLM-generated summary of the change impact
"""

import os
from collections import deque

from groq import Groq

from index.graph_store import GraphStore


def compute_blast_radius(symbol_name: str, graph: GraphStore) -> dict:
    """
    BFS through callers of symbol_name to find the full blast radius.

    Returns:
      {
        "symbol":             target symbol name,
        "direct_callers":     list of {file_path, symbol_name} at depth 1,
        "all_affected":       all transitively affected symbol names,
        "affected_files":     unique files that contain affected symbols,
        "depth":              max traversal depth reached,
        "impact_pct":         percentage of indexed codebase affected,
        "total_symbols":      total symbols in the index,
        "by_depth":           {depth: [symbol_names]} for layered display,
      }
    """
    visited: set[str] = set()
    by_depth: dict[int, list[str]] = {}
    affected_files: set[str] = set()
    queue: deque[tuple[str, int]] = deque()

    # Seed with the direct callers of the target symbol
    direct_callers = graph.get_callers(symbol_name)
    for caller in direct_callers:
        queue.append((caller["symbol_name"], 1))
        affected_files.add(caller["file_path"])

    max_depth = 0

    while queue:
        current_sym, depth = queue.popleft()
        if current_sym in visited:
            continue
        visited.add(current_sym)
        max_depth = max(max_depth, depth)

        by_depth.setdefault(depth, []).append(current_sym)

        # Find callers of current_sym (transitive)
        for caller in graph.get_callers(current_sym):
            if caller["symbol_name"] not in visited:
                queue.append((caller["symbol_name"], depth + 1))
                affected_files.add(caller["file_path"])

    total_symbols = graph.count_symbols()
    impact_pct = len(visited) / max(total_symbols, 1) * 100

    return {
        "symbol":         symbol_name,
        "direct_callers": direct_callers,
        "all_affected":   sorted(visited),
        "affected_files": sorted(affected_files),
        "depth":          max_depth,
        "impact_pct":     impact_pct,
        "total_symbols":  total_symbols,
        "by_depth":       by_depth,
    }


def format_blast_radius(result: dict, verbose: bool = False) -> str:
    """Format blast radius as a human-readable report."""
    sym = result["symbol"]
    n = len(result["all_affected"])
    pct = result["impact_pct"]
    depth = result["depth"]

    lines = [
        f"💥 Blast Radius: `{sym}`",
        f"   {n} symbol(s) affected across {len(result['affected_files'])} file(s)  "
        f"({pct:.1f}% of codebase)  |  max depth: {depth}",
        "",
    ]

    # Show layer by layer
    for d in sorted(result["by_depth"]):
        label = "direct callers" if d == 1 else f"depth-{d} callers"
        syms = ", ".join(result["by_depth"][d][:8])
        if len(result["by_depth"][d]) > 8:
            syms += f" ... (+{len(result['by_depth'][d]) - 8} more)"
        lines.append(f"  [{label}]  {syms}")

    lines.append("")
    lines.append("Affected files:")
    for f in result["affected_files"]:
        lines.append(f"  • {f}")

    return "\n".join(lines)


def llm_impact_summary(symbol: str, result: dict, chunk_text: str = "", api_key: str | None = None) -> str:
    """
    Ask the LLM to write a plain-English summary of what changing this
    symbol would break and why.
    """
    affected_list = ", ".join(result["all_affected"][:20])
    if len(result["all_affected"]) > 20:
        affected_list += f" ... and {len(result['all_affected']) - 20} more"

    prompt = f"""\
You are a senior engineer reviewing a potential code change.

The developer wants to modify the function/class: `{symbol}`

Here is its source code:
```
{chunk_text or "(source not available)"}
```

Symbols that would be directly or transitively affected ({len(result["all_affected"])} total):
{affected_list}

Affected files:
{chr(10).join("• " + f for f in result["affected_files"][:15])}

Write a concise (5-8 bullet point) change-impact summary:
1. What does `{symbol}` do?
2. Which callers are most likely to break if the signature or behaviour changes?
3. What tests or callsites should be reviewed before merging?
4. Rate the risk: LOW / MEDIUM / HIGH and explain why.
"""

    try:
        client = Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"(LLM summary unavailable: {e})"


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from dotenv import load_dotenv
    load_dotenv()
    from index.graph_store import GraphStore, DEFAULT_GRAPH_DB

    symbol = sys.argv[1] if len(sys.argv) > 1 else "walk_repo"
    graph = GraphStore(DEFAULT_GRAPH_DB)
    result = compute_blast_radius(symbol, graph)
    print(format_blast_radius(result))
