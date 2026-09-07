"""
Phase 4: Graph-based context expansion.

Given the top-k retrieved chunks, expand the context by adding:
  - Callees: functions that the retrieved symbols call
  - Callers: functions that call the retrieved symbols

This is the key differentiator for code Q&A — code questions are
graph questions, not document questions.

Example:
  User asks: "what calls chunk_file?"
  Retrieved: chunk_file definition
  Expanded:  + chunk_repo (calls chunk_file)
             + cmd_index in cli.py (calls chunk_repo which calls chunk_file)
"""

from index.graph_store import GraphStore
from index.store import VectorStore
from ingest.embed import Embedder

# How many callers/callees to pull per symbol — keep it tight to stay
# within token budget.
MAX_CALLERS_PER_SYMBOL = 3
MAX_CALLEES_PER_SYMBOL = 3


def _chunk_key(chunk: dict) -> tuple:
    return (chunk["file_path"], chunk["start_line"])


def _fetch_chunk_for_symbol(
    symbol_name: str,
    file_path: str,
    embedder: Embedder,
    store: VectorStore,
) -> dict | None:
    """
    Fetch the stored chunk for a given symbol by doing a targeted
    embedding search. Returns None if not found.
    """
    query = f"def {symbol_name}"
    vec = embedder.embed_query(query)
    hits = store.query(vec, k=5)
    for hit in hits:
        if (
            hit.get("symbol_name") == symbol_name
            or symbol_name in hit.get("text", "")
        ) and (file_path == "" or hit["file_path"] == file_path):
            return hit
    return None


def expand_with_graph(
    query: str,
    chunks: list[dict],
    graph: GraphStore,
    embedder: Embedder,
    store: VectorStore,
    max_extra: int = 4,
) -> list[dict]:
    """
    Given retrieved chunks, find and append related chunks via the call graph.

    Returns the original chunks + up to max_extra graph-expanded chunks,
    deduped by (file_path, start_line). Original order is preserved;
    expanded chunks are appended at the end tagged with 'graph_expanded=True'.
    """
    seen = {_chunk_key(c) for c in chunks}
    extra: list[dict] = []

    for chunk in chunks:
        symbol = chunk.get("symbol_name")
        if not symbol or len(extra) >= max_extra:
            break

        # ── callers (who calls this function?) ────────────────────────────
        for caller in graph.get_callers(symbol)[:MAX_CALLERS_PER_SYMBOL]:
            if len(extra) >= max_extra:
                break
            locs = graph.get_symbol_location(caller["symbol_name"])
            for loc in locs[:1]:
                fetched = _fetch_chunk_for_symbol(
                    caller["symbol_name"], loc["file_path"], embedder, store
                )
                if fetched and _chunk_key(fetched) not in seen:
                    seen.add(_chunk_key(fetched))
                    fetched["graph_expanded"] = True
                    fetched["graph_reason"] = f"calls {symbol}"
                    extra.append(fetched)

        # ── callees (what does this function call?) ────────────────────────
        for callee_name in graph.get_callees(symbol)[:MAX_CALLEES_PER_SYMBOL]:
            if len(extra) >= max_extra:
                break
            locs = graph.get_symbol_location(callee_name)
            for loc in locs[:1]:
                fetched = _fetch_chunk_for_symbol(
                    callee_name, loc["file_path"], embedder, store
                )
                if fetched and _chunk_key(fetched) not in seen:
                    seen.add(_chunk_key(fetched))
                    fetched["graph_expanded"] = True
                    fetched["graph_reason"] = f"called by {symbol}"
                    extra.append(fetched)

    return chunks + extra
