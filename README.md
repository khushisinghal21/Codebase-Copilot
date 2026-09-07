# Codebase Intelligence Copilot — Phase 1 (MVP)

Answers questions about a codebase with grounded `path:line` citations.
Fixed-size chunking + dense retrieval only — this is the MVP from the
build plan. Phase 2+ will swap in AST chunking, BM25 hybrid search,
reranking, and call-graph expansion without changing the CLI interface.

## Setup

```bash
cd codebase-copilot
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and add your ANTHROPIC_API_KEY
```

First run will download the embedding model (~90MB, one-time).

## Usage

Index a repo:
```bash
python cli.py index /path/to/some/repo
```

Ask a question:
```bash
python cli.py ask "where is authentication handled?"
```

Re-index from scratch (wipes old index):
```bash
python cli.py index /path/to/repo --reset
```

Adjust how many chunks get retrieved:
```bash
python cli.py ask "what calls the login function?" --k 15
```

## How it works

1. `ingest/walk_repo.py` — walks the repo, respects `.gitignore`, filters to `.py` files
2. `ingest/chunk.py` — splits each file into ~300-line overlapping chunks
3. `ingest/embed.py` — embeds chunks locally with `all-MiniLM-L6-v2`
4. `index/store.py` — stores vectors in a local ChromaDB (`.copilot_index/`)
5. `retrieve/search.py` — embeds the query, does cosine similarity search
6. `generate/answer.py` — sends retrieved chunks to Claude, enforces `path:line` citations, refuses on low-confidence retrieval

## Known limitations (this is the point — Phase 1 is meant to be rough)

- Only indexes `.py` files — extend `DEFAULT_EXTENSIONS` in `walk_repo.py` for other languages
- Fixed-size chunking means a function can get split across two chunks (fixed in Phase 2 with AST chunking)
- No keyword/exact-match search yet, so literal symbol name queries can miss (fixed in Phase 2 with BM25 + rank fusion)
- No reranking (Phase 3), no call-graph expansion (Phase 4), no incremental re-indexing (Phase 5)

## Next steps (Phase 2)

- Replace `chunk.py` with tree-sitter AST-based chunking (function/class boundaries)
- Add a SQLite FTS5 keyword index alongside the vector store
- Fuse dense + keyword rankings with reciprocal rank fusion
- Build a 20-30 question gold set to start measuring recall@10 before/after changes
