#!/usr/bin/env python3
"""
CLI for the Codebase Intelligence Copilot (Phase 5).

Usage:
    python cli.py index /path/to/repo          # full index
    python cli.py update /path/to/repo         # incremental (only changed files)
    python cli.py ask "what calls chunk_file?"
    python cli.py ask "..." --no-rerank --no-graph
"""

import argparse
import sys
import time

from dotenv import load_dotenv

from ingest.walk_repo import walk_repo
from ingest.chunk_ast import chunk_repo   # Phase 2: AST-aware chunking
from ingest.embed import Embedder
from ingest.call_graph import build_call_graph
from ingest.diff_indexer import get_changed_files, get_current_commit, is_git_repo  # Phase 5
from index.store import VectorStore, DEFAULT_PERSIST_DIR
from index.bm25_store import BM25Store, DEFAULT_DB_PATH
from index.graph_store import GraphStore, DEFAULT_GRAPH_DB
from retrieve.search import search
from retrieve.rerank import Reranker
from retrieve.rewrite import rewrite_query
from retrieve.expand import expand_with_graph
from generate.answer import generate_answer
from generate.validate import validate_citations, format_validation_report

load_dotenv()


def cmd_index(args):
    print(f"Indexing {args.repo_path} ...")
    t0 = time.time()

    store = VectorStore(persist_dir=args.index_dir)
    bm25 = BM25Store(db_path=args.bm25_db)
    graph = GraphStore(db_path=args.graph_db)
    if args.reset:
        store.reset()
        bm25.reset()
        graph.reset()

    embedder = Embedder()

    all_chunks = []
    chunks_batch = []
    total = 0
    BATCH_SIZE = 100

    for chunk in chunk_repo(walk_repo(args.repo_path)):
        chunks_batch.append(chunk)
        all_chunks.append(chunk)
        if len(chunks_batch) >= BATCH_SIZE:
            texts = [c["text"] for c in chunks_batch]
            embeddings = embedder.embed_texts(texts)
            store.add_chunks(chunks_batch, embeddings)
            bm25.add_chunks(chunks_batch)
            total += len(chunks_batch)
            print(f"  indexed {total} chunks...", file=sys.stderr)
            chunks_batch = []

    if chunks_batch:
        texts = [c["text"] for c in chunks_batch]
        embeddings = embedder.embed_texts(texts)
        store.add_chunks(chunks_batch, embeddings)
        bm25.add_chunks(chunks_batch)
        total += len(chunks_batch)

    # Phase 4: build call graph
    print("Building call graph...", file=sys.stderr)
    graph.add_symbols(all_chunks)
    edges = build_call_graph(walk_repo(args.repo_path))
    graph.add_calls(edges)
    print(f"  {len(edges)} call edges indexed", file=sys.stderr)

    # Phase 5: save current commit SHA so `update` knows where we left off
    if is_git_repo(args.repo_path):
        commit = get_current_commit(args.repo_path)
        if commit:
            graph.set_meta("last_indexed_commit", commit)
            print(f"  Commit SHA saved: {commit[:12]}", file=sys.stderr)

    elapsed = time.time() - t0
    print(f"\nDone. {total} chunks indexed in {elapsed:.1f}s")
    print(f"  Vector store -> {args.index_dir}")
    print(f"  BM25 index   -> {args.bm25_db}")
    print(f"  Call graph   -> {args.graph_db}")

def cmd_update(args):
    """
    Phase 5: Incremental re-index — only process files changed since the
    last full index (tracked by commit SHA in the graph store).
    """
    graph = GraphStore(db_path=args.graph_db)
    last_commit = graph.get_meta("last_indexed_commit")

    if not last_commit:
        print(
            "No previous index commit found. Run `python cli.py index <repo>` first.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not is_git_repo(args.repo_path):
        print("Not a git repository. Use `python cli.py index` instead.", file=sys.stderr)
        sys.exit(1)

    print(f"Checking changes since commit {last_commit[:12]}...")
    diff = get_changed_files(args.repo_path, last_commit)

    if not diff.to_delete and not diff.to_index:
        print("✅ Index is up to date — no Python files changed.")
        # Still update the commit pointer
        if diff.new_commit:
            graph.set_meta("last_indexed_commit", diff.new_commit)
        return

    print(f"  {len(diff.to_delete)} file(s) to remove from index")
    print(f"  {len(diff.to_index)} file(s) to (re-)index")
    t0 = time.time()

    store = VectorStore(persist_dir=args.index_dir)
    bm25 = BM25Store(db_path=args.bm25_db)
    embedder = Embedder()

    # Step 1: delete stale chunks from all three stores
    total_deleted = 0
    for file_path in diff.to_delete:
        file_str = str(file_path)
        n = store.delete_by_file(file_str)
        bm25.delete_by_file(file_str)
        graph.delete_by_file(file_str)
        total_deleted += n
        print(f"  deleted {n} chunk(s) from {file_path.name}", file=sys.stderr)

    # Step 2: re-chunk and re-embed changed/new files
    from ingest.chunk_ast import chunk_file
    total_added = 0
    new_chunks_all = []

    for file_path in diff.to_index:
        new_chunks = chunk_file(file_path)
        if not new_chunks:
            continue
        texts = [c["text"] for c in new_chunks]
        embeddings = embedder.embed_texts(texts)
        store.add_chunks(new_chunks, embeddings)
        bm25.add_chunks(new_chunks)
        new_chunks_all.extend(new_chunks)
        total_added += len(new_chunks)
        print(f"  indexed {len(new_chunks)} chunk(s) from {file_path.name}", file=sys.stderr)

    # Step 3: update call graph for changed files
    if new_chunks_all:
        graph.add_symbols(new_chunks_all)
    from ingest.call_graph import extract_calls
    for file_path in diff.to_index:
        edges = extract_calls(file_path)
        graph.add_calls(edges)

    # Step 4: advance the commit pointer
    if diff.new_commit:
        graph.set_meta("last_indexed_commit", diff.new_commit)
        print(f"  Commit SHA updated: {diff.new_commit[:12]}", file=sys.stderr)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s — deleted {total_deleted} old chunk(s), added {total_added} new chunk(s).")




def cmd_ask(args):
    store = VectorStore(persist_dir=args.index_dir)
    if store.count() == 0:

        print(
            f"Index at '{args.index_dir}' is empty. Run `python cli.py index <repo_path>` first.",
            file=sys.stderr,
        )
        sys.exit(1)

    bm25 = BM25Store(db_path=args.bm25_db)
    graph = GraphStore(db_path=args.graph_db)
    embedder = Embedder()

    # Phase 3: query rewriting
    query = args.question
    if not args.no_rewrite:
        expanded = rewrite_query(query)
        if expanded != query:
            print(f"Query expanded: {expanded}", file=sys.stderr)
            query = expanded

    # Phase 3: cross-encoder reranker
    reranker = None
    if not args.no_rerank:
        try:
            reranker = Reranker()
        except Exception as e:
            print(f"Reranker unavailable ({e}), skipping.", file=sys.stderr)

    chunks = search(args.question, embedder, store, bm25_store=bm25, reranker=reranker, k=args.k)

    # Phase 4: graph expansion
    if not args.no_graph:
        chunks = expand_with_graph(args.question, chunks, graph, embedder, store)

    print("--- Retrieved chunks ---")
    for c in chunks:
        symbol = f" [{c.get('symbol_type','?')}:{c.get('symbol_name','?')}]" if c.get('symbol_name') else ""
        graph_tag = f" 🔗 {c.get('graph_reason', 'graph')}" if c.get('graph_expanded') else ""
        print(f"  {c['file_path']}:{c['start_line']}-{c['end_line']}  (score={c['score']:.3f}){symbol}{graph_tag}")
    print()

    answer = generate_answer(args.question, chunks)

    # Phase 4: citation validation
    if not args.no_validate:
        report = validate_citations(answer, chunks)
        if report["hallucinated"]:
            answer = report["flagged_answer"]
        print(format_validation_report(report))
        print()

    print("--- Answer ---")
    print(answer)


def main():
    parser = argparse.ArgumentParser(description="Codebase Intelligence Copilot")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Index a repository")
    index_parser.add_argument("repo_path", help="Path to the repo to index")
    index_parser.add_argument(
        "--index-dir", default=DEFAULT_PERSIST_DIR, help="Where to store the vector index"
    )
    index_parser.add_argument(
        "--bm25-db", default=DEFAULT_DB_PATH, help="Where to store the BM25 index"
    )
    index_parser.add_argument(
        "--graph-db", default=DEFAULT_GRAPH_DB, help="Where to store the call graph"
    )
    index_parser.add_argument(
        "--reset", action="store_true", help="Wipe existing index before indexing"
    )
    index_parser.set_defaults(func=cmd_index)

    # ── update (Phase 5) ──────────────────────────────────────────────────────
    update_parser = subparsers.add_parser(
        "update", help="Incrementally re-index only files changed since last index"
    )
    update_parser.add_argument("repo_path", help="Path to the repo")
    update_parser.add_argument(
        "--index-dir", default=DEFAULT_PERSIST_DIR, help="Vector index location"
    )
    update_parser.add_argument(
        "--bm25-db", default=DEFAULT_DB_PATH, help="BM25 index location"
    )
    update_parser.add_argument(
        "--graph-db", default=DEFAULT_GRAPH_DB, help="Call graph location"
    )
    update_parser.set_defaults(func=cmd_update)

    # ── ask ───────────────────────────────────────────────────────────────────
    ask_parser = subparsers.add_parser("ask", help="Ask a question about the indexed repo")
    ask_parser.add_argument("question", help="Your question")
    ask_parser.add_argument("--k", type=int, default=10, help="Number of chunks to retrieve")
    ask_parser.add_argument(
        "--index-dir", default=DEFAULT_PERSIST_DIR, help="Where the vector index is stored"
    )
    ask_parser.add_argument(
        "--bm25-db", default=DEFAULT_DB_PATH, help="Where the BM25 index is stored"
    )
    ask_parser.add_argument(
        "--graph-db", default=DEFAULT_GRAPH_DB, help="Where the call graph is stored"
    )
    ask_parser.add_argument(
        "--no-rerank", action="store_true", help="Skip cross-encoder reranking (Phase 3)"
    )
    ask_parser.add_argument(
        "--no-rewrite", action="store_true", help="Skip LLM query rewriting (Phase 3)"
    )
    ask_parser.add_argument(
        "--no-graph", action="store_true", help="Skip call-graph expansion (Phase 4)"
    )
    ask_parser.add_argument(
        "--no-validate", action="store_true", help="Skip citation validation (Phase 4)"
    )
    ask_parser.set_defaults(func=cmd_ask)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
