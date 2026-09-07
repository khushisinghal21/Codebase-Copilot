"""
Phase 7b: Onboarding Mode — Guided Reading Order for New Contributors.

Algorithm:
  1. Load all indexed symbols and call edges from the graph.
  2. Compute in-degree (how many symbols call this?) and
     out-degree (how many symbols does this call?).
  3. Categorise each symbol into a layer:
       Layer 0 — Foundation : heavily depended-upon utilities
       Layer 1 — Core Logic : main business logic
       Layer 2 — Entry Points: CLI commands / top-level orchestration
  4. Within each layer, sort by "interestingness" (in-degree desc).
  5. Retrieve the source chunk for each symbol.
  6. Ask the LLM to write a reading guide in plain English.

Output: a structured, LLM-narrated reading guide the new developer
        can follow from top to bottom.
"""

import os
from collections import defaultdict

from groq import Groq

from index.graph_store import GraphStore
from index.store import VectorStore
from ingest.embed import Embedder


# ── topology helpers ─────────────────────────────────────────────────────────

def _build_degree_maps(symbols: list[dict], edges: list[tuple]) -> tuple[dict, dict]:
    """Returns (in_degree, out_degree) keyed by symbol_name."""
    in_deg: dict[str, int] = {s["symbol_name"]: 0 for s in symbols}
    out_deg: dict[str, int] = {s["symbol_name"]: 0 for s in symbols}
    for caller, callee in edges:
        if caller in out_deg:
            out_deg[caller] += 1
        if callee in in_deg:
            in_deg[callee] += 1
    return in_deg, out_deg


def _categorise(sym: str, in_deg: dict, out_deg: dict) -> int:
    """
    Returns layer index:
      0 = Foundation  (in_deg > 3   — widely depended upon)
      2 = Entry Point (in_deg == 0 AND out_deg >= 2 — top-level orchestrators)
      1 = Core Logic  (everything else)
    """
    i = in_deg.get(sym, 0)
    o = out_deg.get(sym, 0)
    if i == 0 and o >= 2:
        return 2
    if i >= 3:
        return 0
    return 1


# ── chunk fetcher ─────────────────────────────────────────────────────────────

def _fetch_chunk(symbol_name: str, file_path: str, embedder: Embedder, store: VectorStore) -> str:
    vec = embedder.embed_query(f"def {symbol_name}")
    hits = store.query(vec, k=5)
    for hit in hits:
        if hit.get("symbol_name") == symbol_name and hit["file_path"] == file_path:
            return hit["text"]
    # Fallback: return any hit from the right file
    for hit in hits:
        if hit["file_path"] == file_path:
            return hit["text"][:500]
    return ""


# ── main function ─────────────────────────────────────────────────────────────

LAYER_META = [
    {
        "name":        "🧱 Layer 0 — Foundation",
        "description": "These are the building blocks everything else depends on. Read these first.",
        "emoji":       "🧱",
    },
    {
        "name":        "⚙️  Layer 1 — Core Logic",
        "description": "The main algorithms and business logic. Read after the foundations.",
        "emoji":       "⚙️",
    },
    {
        "name":        "🚀 Layer 2 — Entry Points",
        "description": "Where execution begins — CLI commands and top-level orchestrators. Read last.",
        "emoji":       "🚀",
    },
]


def build_reading_plan(
    graph: GraphStore,
    embedder: Embedder,
    store: VectorStore,
    max_per_layer: int = 8,
) -> list[dict]:
    """
    Returns a list of layer dicts, each with:
      { "meta": {...}, "symbols": [{symbol info + chunk text}] }
    """
    symbols = graph.get_all_symbols()
    edges = graph.get_all_calls()

    in_deg, out_deg = _build_degree_maps(symbols, edges)

    layers: dict[int, list[dict]] = defaultdict(list)
    for sym in symbols:
        layer_idx = _categorise(sym["symbol_name"], in_deg, out_deg)
        layers[layer_idx].append(
            {
                **sym,
                "in_degree":  in_deg.get(sym["symbol_name"], 0),
                "out_degree": out_deg.get(sym["symbol_name"], 0),
            }
        )

    result = []
    for layer_idx in [0, 1, 2]:
        syms = sorted(layers[layer_idx], key=lambda x: x["in_degree"], reverse=True)
        syms = syms[:max_per_layer]

        # Fetch source snippets
        for sym in syms:
            sym["chunk_text"] = _fetch_chunk(sym["symbol_name"], sym["file_path"], embedder, store)

        result.append({"meta": LAYER_META[layer_idx], "symbols": syms})

    return result


def generate_onboarding_guide(
    plan: list[dict],
    repo_name: str = "the codebase",
    api_key: str | None = None,
) -> str:
    """
    Ask the LLM to write a guided onboarding document from the reading plan.
    """
    # Build a compact context string for the LLM
    context_parts = []
    for layer in plan:
        context_parts.append(f"\n### {layer['meta']['name']}")
        context_parts.append(layer["meta"]["description"])
        for sym in layer["symbols"]:
            snippet = sym.get("chunk_text", "")[:400].replace("\n", " ")
            context_parts.append(
                f"  - `{sym['symbol_name']}` ({sym['symbol_type']}) in {sym['file_path'].split('/')[-1]}: {snippet}"
            )

    context = "\n".join(context_parts)

    prompt = f"""\
You are an experienced senior engineer writing an onboarding guide for a new contributor
joining the project: **{repo_name}**.

Here is the dependency-ordered symbol map of the codebase (foundation → core → entry points):

{context}

Write a friendly, structured onboarding guide with:
1. A short intro paragraph explaining what this project does (infer from the symbols).
2. For each layer (Foundation, Core Logic, Entry Points), write:
   - A 1-2 sentence explanation of what this layer is responsible for.
   - Bullet points naming the 3-5 most important symbols and what to look for when reading them.
   - A "read next" transition to the next layer.
3. End with a "First PR suggestion" — a small, well-scoped change a newcomer could make.

Be concise but warm. Use markdown formatting.
"""

    try:
        client = Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            max_tokens=1200,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"(LLM guide unavailable: {e})\n\n" + "\n".join(
            f"{layer['meta']['name']}\n" + "\n".join(f"  - {s['symbol_name']}" for s in layer["symbols"])
            for layer in plan
        )


def print_reading_plan(plan: list[dict]) -> None:
    """Print the topology-only plan without the LLM guide."""
    for layer in plan:
        print(f"\n{layer['meta']['name']}")
        print(f"  {layer['meta']['description']}")
        for sym in layer["symbols"]:
            f = sym["file_path"].split("/")[-1]
            print(f"  • {sym['symbol_name']:<30} ({sym['symbol_type']}) in {f}  "
                  f"[called_by={sym['in_degree']} calls={sym['out_degree']}]")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from dotenv import load_dotenv
    load_dotenv()
    from index.graph_store import GraphStore, DEFAULT_GRAPH_DB
    from index.store import VectorStore, DEFAULT_PERSIST_DIR
    from ingest.embed import Embedder

    graph = GraphStore(DEFAULT_GRAPH_DB)
    store = VectorStore(DEFAULT_PERSIST_DIR)
    embedder = Embedder()

    plan = build_reading_plan(graph, embedder, store)
    print_reading_plan(plan)
