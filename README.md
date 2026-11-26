# 🤖 Codebase Copilot

A local, privacy-first AI assistant that deeply understands your codebase. Ask natural language questions, trace call graphs, estimate change impact, and onboard new contributors — all from the CLI.

> Built in 8 progressive phases, from basic RAG to a code intelligence platform with memory that improves between sessions.

---

## ✨ Features

| Feature | Command |
|---------|---------|
| Ask questions about any code | `python cli.py ask "how does search work?"` |
| Filter to a specific repo | `python cli.py ask "..." --repo backend` |
| Full re-index | `python cli.py index /path/to/repo` |
| Incremental update (git diff) | `python cli.py update /path/to/repo` |
| List indexed repos | `python cli.py repos` |
| Blast radius — who breaks if I change X? | `python cli.py blast walk_repo` |
| Onboarding guide for new contributors | `python cli.py onboard` |
| Consolidate past sessions into memory ("dream") | `python cli.py dream /path/to/repo` |
| Ask with dreamed memory | `python cli.py ask "..." --use-memory` |

---

## 🏗️ Architecture — 8 Phases

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────┐
│  Phase 3: LLM Query Rewriting (Groq)            │
└───────────────────────┬─────────────────────────┘
                        │
          ┌─────────────▼──────────────┐
          │  Phase 2: Hybrid Retrieval  │
          │  Dense (ChromaDB)           │
          │     + BM25 (SQLite FTS5)   │
          │     → RRF Fusion           │
          │  Phase 3: Cross-encoder    │
          │  Reranker                  │
          └─────────────┬──────────────┘
                        │
          ┌─────────────▼──────────────┐
          │  Phase 4: Graph Expansion  │
          │  Call graph (331 edges)    │
          │  Callers + Callees added   │
          └─────────────┬──────────────┘
                        │
          ┌─────────────▼──────────────┐
          │  LLM Answer (Groq)         │
          │  + Citation Validation     │
          └────────────────────────────┘
```

### Phase 1 — Foundation
- Fixed-size text chunking
- Dense vector embeddings (`all-MiniLM-L6-v2`)
- ChromaDB persistent vector store
- Groq LLM for answer generation

### Phase 2 — Hybrid Retrieval
- **AST-aware chunking** with `tree-sitter` — functions/classes stay intact
- **BM25 full-text search** via SQLite FTS5
- **Reciprocal Rank Fusion (RRF)** merges dense + sparse results
- recall@10 = **1.0** on eval set

### Phase 3 — Intelligence Layer
- **LLM query rewriting** expands terse queries with synonyms
- **Cross-encoder reranking** (`ms-marco-MiniLM-L-6-v2`) for precision
- **Eval suite** (`eval/run_eval.py`) — MRR = **0.817**

### Phase 4 — Call Graph
- **Tree-sitter call extraction** — 331 directed edges indexed
- **Graph-based context expansion** — adds callers/callees to retrieval context
- **Citation validator** — flags hallucinated file references before showing the answer

### Phase 5 — Incremental Re-indexing
- `git diff` detects changed files since last index
- Surgically deletes stale chunks from all 3 stores
- Re-indexes only modified files — **seconds instead of minutes**
- Commit SHA tracked in SQLite metadata

### Phase 6 — Multi-repo
- Index multiple repos into a single unified store
- Each chunk tagged with `repo_id`
- Filter queries to a single repo with `--repo`
- Cross-repo import detection (e.g. `from shared_lib import X` → dependency edge)
- `python cli.py repos` lists all indexed repos

### Phase 7 — Code Intelligence
- **Blast Radius** — BFS through callers to estimate how many symbols/files break if you change a function. Layered by depth + LLM risk rating.
- **Onboarding Mode** — Topological sort of the dependency graph produces a Foundation → Core → Entry Point reading order. LLM writes a personalized guide.

### Phase 8 — Memory with offline "dreaming"
Inspired by Anthropic's [The components of agent memory](https://www.anthropic.com/engineering/components-of-agent-memory).

Memory is stored in layers. The bottom ones are kept raw; the top ones stay small:

| Layer | File | What it holds |
|---|---|---|
| Transcripts | `.copilot_memory/transcripts.jsonl` | Every `ask`: question, retrieved chunks, chunks confirmed as the answer (validated citations or `cli.py feedback`) |
| Notes | `notes.json` | Which symbol answered which question, and how far down search had ranked it |
| Wiki | `wiki/*.md` + `wiki.db` | One linked page per file: module purpose, symbols, call links, past questions. Indexed per symbol section |
| Injected index | `MEMORY.md` | Map of modules, most-asked symbols, search blind spots. Capped (150 lines by default) because it is added to every answer |

`python cli.py dream <repo>` is the offline pass. It consolidates new transcripts into notes, **re-checks every note against the current code** (repointing ones whose symbol moved and dropping ones whose symbol is gone), rebuilds the wiki, rewrites `MEMORY.md`, and warns if the wiki grows larger than the source. It makes no LLM calls, so it is cheap enough to run after every commit.

With `--use-memory`, retrieval adds the wiki as a third ranked list in RRF, and `MEMORY.md` is given to the LLM as a map (never as something to cite).

**Results** (`eval/memory_results.md`): BM25 retriever, the 20-question gold set, leave-one-out, so no question is ever in its own memory:

| Condition | recall@1 | recall@5 | MRR |
|---|---|---|---|
| Search alone | 0.45 | 0.70 | 0.589 |
| Search + wiki (from code only) | 0.50 | 0.80 | 0.648 |
| Search + wiki + memory | **0.60** | **0.85** | **0.715** |

Stale-note test: after a refactor that shifted line numbers in 9 files and renamed one function, 15/15 notes pointed at the wrong lines. One dream repointed 14 and dropped the renamed one, leaving 0 wrong.

Caveats: 20 questions is a small eval; recall@10 dipped from 0.95 to 0.90 because the wiki pushed one low-ranked correct chunk out of the top 10; the gold set predates Phase 8 and the wiki weights were fixed before running it. The dense half of hybrid search was not run for these numbers. Run `python eval/run_memory_eval.py --stale-test` to reproduce with the full hybrid retriever.

---

## ⚡ Quick Start

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set your API key

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

Get a free key at [console.groq.com](https://console.groq.com).

### 3. Index a repo

```bash
# Single repo
python cli.py index /path/to/your/repo

# Multiple repos (multi-repo mode)
python cli.py index /path/to/backend   --repo-name backend
python cli.py index /path/to/frontend  --repo-name frontend
python cli.py index /path/to/shared    --repo-name shared-lib
```

### 4. Ask questions

```bash
python cli.py ask "how does authentication work?"
python cli.py ask "what calls chunk_file?"
python cli.py ask "where is the database connection set up?"

# Filter to one repo
python cli.py ask "where is JWT defined?" --repo backend

# Skip expensive steps for faster answers
python cli.py ask "what does walk_repo do?" --no-rerank --no-graph
```

### 5. Analyse change impact

```bash
# Who breaks if I change this function?
python cli.py blast walk_repo
python cli.py blast BM25Store --no-llm   # faster, skip LLM

# Generate an onboarding guide for new contributors
python cli.py onboard
python cli.py onboard --no-llm           # topology only
```

### 6. Incremental updates

```bash
# After any git commit — only re-indexes changed files
python cli.py update /path/to/repo
```

---

## 📁 Project Structure

```
codebase-copilot/
├── cli.py                   # Main CLI entry point (all commands)
│
├── ingest/                  # Data ingestion pipeline
│   ├── walk_repo.py         # File walker with .gitignore support
│   ├── chunk.py             # Fixed-size chunker (Phase 1)
│   ├── chunk_ast.py         # AST-aware chunker with tree-sitter (Phase 2)
│   ├── embed.py             # Sentence-transformer embedder
│   ├── call_graph.py        # Tree-sitter call edge extractor (Phase 4)
│   ├── diff_indexer.py      # Git diff → changed file detection (Phase 5)
│   └── import_parser.py     # Cross-repo import detection (Phase 6)
│
├── index/                   # Storage layer (3 stores)
│   ├── store.py             # ChromaDB vector store
│   ├── bm25_store.py        # SQLite FTS5 BM25 index
│   ├── graph_store.py       # SQLite call graph + metadata
│   └── repo_registry.py     # Multi-repo registry (Phase 6)
│
├── retrieve/                # Retrieval pipeline
│   ├── search.py            # RRF fusion search (dense + BM25 + rerank)
│   ├── rerank.py            # Cross-encoder reranker (Phase 3)
│   ├── rewrite.py           # LLM query rewriting (Phase 3)
│   └── expand.py            # Call-graph context expansion (Phase 4)
│
├── generate/                # Answer generation
│   ├── answer.py            # Groq LLM answer synthesis
│   └── validate.py          # Citation hallucination validator (Phase 4)
│
├── analyze/                 # Code intelligence (Phase 7)
│   ├── blast_radius.py      # Change impact BFS analyser
│   └── onboarding.py        # Topological reading-order generator
│
├── memory/                  # Agent memory (Phase 8)
│   ├── transcripts.py       # Raw session log (ground truth)
│   ├── codemap.py           # Current symbols, docstrings, call edges
│   ├── dream.py             # Offline pass: notes, verification, wiki, MEMORY.md
│   └── wiki_store.py        # FTS5 index over wiki sections
│
└── eval/                    # Evaluation suite (Phase 3)
    ├── gold_set.json         # 20 query → expected file pairs
    ├── run_eval.py           # Computes recall@k and MRR
    └── run_memory_eval.py    # Memory vs plain search (Phase 8)
```

---

## 🔧 All CLI Commands

```bash
# ── Indexing ──────────────────────────────────────────────────
python cli.py index <repo_path>              # Full index
python cli.py index <repo_path> --reset      # Wipe + full re-index
python cli.py index <repo_path> --repo-name myrepo   # Named repo

python cli.py update <repo_path>             # Incremental (git diff)

python cli.py repos                          # List all indexed repos

# ── Querying ──────────────────────────────────────────────────
python cli.py ask "<question>"
python cli.py ask "<question>" --repo <repo-name>  # Filter by repo
python cli.py ask "<question>" --k 20              # Retrieve more chunks
python cli.py ask "<question>" --no-rerank         # Skip cross-encoder
python cli.py ask "<question>" --no-rewrite        # Skip query rewriting
python cli.py ask "<question>" --no-graph          # Skip graph expansion
python cli.py ask "<question>" --no-validate       # Skip citation check

# ── Analysis ──────────────────────────────────────────────────
python cli.py blast <symbol_name>            # Blast radius report + LLM risk
python cli.py blast <symbol_name> --no-llm  # Topology only (faster)

python cli.py onboard                        # Reading guide + LLM narrative
python cli.py onboard --no-llm              # Topology only (faster)

# ── Memory (Phase 8) ──────────────────────────────────────────
python cli.py dream <repo_path>              # Offline consolidation pass
python cli.py ask "<question>" --use-memory  # Retrieve with the dreamed wiki
python cli.py feedback <session_id> <rank>   # Confirm which chunk was right

# ── Evaluation ────────────────────────────────────────────────
python eval/run_eval.py                      # Full eval with reranker
python eval/run_eval.py --no-rerank         # Phase 2 baseline
python eval/run_memory_eval.py --stale-test # Memory vs plain search (Phase 8)
```

---

## 📊 Eval Results

| Metric | Phase 2 (no rerank) | Phase 3 (with rerank) |
|--------|--------------------|-----------------------|
| recall@5 | 0.800 | 0.800 |
| recall@10 | **1.000** | **1.000** |
| MRR | 0.750 | **0.817** |

---

## 🧰 Tech Stack

| Component | Technology |
|-----------|-----------|
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector store | `ChromaDB` (persistent, local) |
| Keyword search | SQLite FTS5 (BM25) |
| Call graph | `tree-sitter` + SQLite |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | Groq API (`openai/gpt-oss-120b`) |
| Query rewriting | Groq API |

---

## 🔑 Environment Variables

```bash
# .env
GROQ_API_KEY=your_key_here   # Required — get free at console.groq.com
```

---

## 📝 License

MIT
