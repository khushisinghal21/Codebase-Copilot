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
from pathlib import Path

from dotenv import load_dotenv

from ingest.walk_repo import walk_repo
from ingest.chunk_ast import chunk_repo   # Phase 2: AST-aware chunking
from ingest.embed import Embedder
from ingest.call_graph import build_call_graph
from ingest.diff_indexer import get_changed_files, get_current_commit, is_git_repo
from ingest.import_parser import find_cross_repo_imports  # Phase 6
from index.store import VectorStore, DEFAULT_PERSIST_DIR
from index.bm25_store import BM25Store, DEFAULT_DB_PATH
from index.graph_store import GraphStore, DEFAULT_GRAPH_DB
from index.repo_registry import RepoRegistry, derive_repo_id  # Phase 6
from retrieve.search import search
from retrieve.rerank import Reranker
from retrieve.rewrite import rewrite_query
from retrieve.expand import expand_with_graph
from generate.answer import generate_answer
from generate.validate import validate_citations, format_validation_report
from analyze.blast_radius import compute_blast_radius, format_blast_radius, llm_impact_summary  # Phase 7
from analyze.onboarding import build_reading_plan, generate_onboarding_guide, print_reading_plan  # Phase 7

load_dotenv()


def cmd_index(args):
    repo_id = args.repo_name or derive_repo_id(args.repo_path)
    print(f"Indexing {args.repo_path} as repo '{repo_id}'...")
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
            store.add_chunks(chunks_batch, embeddings, repo_id=repo_id)
            bm25.add_chunks(chunks_batch, repo_id=repo_id)
            total += len(chunks_batch)
            print(f"  indexed {total} chunks...", file=sys.stderr)
            chunks_batch = []

    if chunks_batch:
        texts = [c["text"] for c in chunks_batch]
        embeddings = embedder.embed_texts(texts)
        store.add_chunks(chunks_batch, embeddings, repo_id=repo_id)
        bm25.add_chunks(chunks_batch, repo_id=repo_id)
        total += len(chunks_batch)

    # Phase 4: build call graph
    print("Building call graph...", file=sys.stderr)
    graph.add_symbols(all_chunks)
    edges = build_call_graph(walk_repo(args.repo_path))
    graph.add_calls(edges)
    print(f"  {len(edges)} call edges indexed", file=sys.stderr)

    # Phase 6: detect cross-repo import dependencies
    registry = RepoRegistry(graph.conn)
    registry.register(repo_id, str(Path(args.repo_path).resolve()),
                      last_commit=get_current_commit(args.repo_path) if is_git_repo(args.repo_path) else None,
                      chunk_count=total)
    all_repo_ids = registry.all_repo_ids()
    cross_repo_edges = 0
    for file_path in walk_repo(args.repo_path):
        deps = find_cross_repo_imports(file_path, all_repo_ids)
        for dep_repo in deps:
            graph.add_calls([(str(file_path), f"<imports:{dep_repo}>", dep_repo)])
            cross_repo_edges += 1
    if cross_repo_edges:
        print(f"  {cross_repo_edges} cross-repo dependency edges", file=sys.stderr)

    # Phase 5: save current commit SHA so `update` knows where we left off
    if is_git_repo(args.repo_path):
        commit = get_current_commit(args.repo_path)
        if commit:
            graph.set_meta("last_indexed_commit", commit)
            print(f"  Commit SHA saved: {commit[:12]}", file=sys.stderr)

    elapsed = time.time() - t0
    print(f"\nDone. {total} chunks indexed in {elapsed:.1f}s")
    print(f"  Repo ID      -> {repo_id}")
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

    repo_id_filter = getattr(args, "repo", None)
    chunks = search(args.question, embedder, store, bm25_store=bm25, reranker=reranker,
                    k=args.k, repo_id=repo_id_filter)

    # Phase 4: graph expansion
    if not args.no_graph:
        chunks = expand_with_graph(args.question, chunks, graph, embedder, store)

    print("--- Retrieved chunks ---")
    for c in chunks:
        symbol = f" [{c.get('symbol_type','?')}:{c.get('symbol_name','?')}]" if c.get('symbol_name') else ""
        graph_tag = f" 🔗 {c.get('graph_reason', 'graph')}" if c.get('graph_expanded') else ""
        repo_tag = f" [repo:{c.get('repo_id','?')}]" if c.get('repo_id') and c.get('repo_id') != 'default' else ""
        print(f"  {c['file_path']}:{c['start_line']}-{c['end_line']}  (score={c['score']:.3f}){symbol}{repo_tag}{graph_tag}")
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


def cmd_repos(args):
    """Phase 6: List all indexed repos."""
    graph = GraphStore(db_path=args.graph_db)
    registry = RepoRegistry(graph.conn)
    repos = registry.list_all()
    if not repos:
        print("No repos indexed yet. Run `python cli.py index <repo_path>` first.")
        return
    print(f"{'Repo ID':<20} {'Chunks':<8} {'Last Commit':<14} {'Path'}")
    print("-" * 80)
    for r in repos:
        commit = (r['last_commit'] or 'unknown')[:12]
        print(f"{r['repo_id']:<20} {r['chunk_count']:<8} {commit:<14} {r['repo_path']}")
    print(f"\nTotal: {len(repos)} repo(s) indexed.")

def cmd_blast(args):
    """Phase 7: Blast radius analysis for a given symbol."""
    graph = GraphStore(db_path=args.graph_db)
    store = VectorStore(persist_dir=args.index_dir)
    embedder = Embedder()

    result = compute_blast_radius(args.symbol, graph)

    # Try to fetch the source chunk for the LLM summary
    from ingest.embed import Embedder as _E
    vec = embedder.embed_query(f"def {args.symbol}")
    hits = store.query(vec, k=5)
    chunk_text = ""
    for hit in hits:
        if hit.get("symbol_name") == args.symbol:
            chunk_text = hit["text"]
            break

    print(format_blast_radius(result))

    if not args.no_llm:
        print("\n--- LLM Impact Summary ---")
        summary = llm_impact_summary(args.symbol, result, chunk_text)
        print(summary)


def cmd_onboard(args):
    """Phase 7: Generate a guided reading order for a new contributor."""
    graph = GraphStore(db_path=args.graph_db)
    store = VectorStore(persist_dir=args.index_dir)
    embedder = Embedder()

    print("Building dependency topology...")
    plan = build_reading_plan(graph, embedder, store)
    print_reading_plan(plan)

    if not args.no_llm:
        registry = RepoRegistry(graph.conn)
        repos = registry.list_all()
        repo_name = repos[0]["repo_id"] if repos else "this codebase"
        print("\n--- Generating Onboarding Guide (LLM) ---")
        guide = generate_onboarding_guide(plan, repo_name=repo_name)
        print(guide)




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
    index_parser.add_argument(
        "--repo-name", default=None,
        help="Short name for this repo (default: folder name). Used for multi-repo filtering."
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
    ask_parser.add_argument(
        "--repo", default=None,
        help="Filter results to a specific repo (Phase 6). Omit to search all repos."
    )
    ask_parser.set_defaults(func=cmd_ask)

    # ── repos (Phase 6) ───────────────────────────────────────────────────────
    repos_parser = subparsers.add_parser("repos", help="List all indexed repos")
    repos_parser.add_argument(
        "--graph-db", default=DEFAULT_GRAPH_DB, help="Call graph location"
    )
    repos_parser.set_defaults(func=cmd_repos)

    # ── blast (Phase 7) ───────────────────────────────────────────────────────
    blast_parser = subparsers.add_parser(
        "blast", help="Estimate impact of changing a symbol (blast radius)"
    )
    blast_parser.add_argument("symbol", help="Symbol name to analyse (e.g. walk_repo)")
    blast_parser.add_argument(
        "--index-dir", default=DEFAULT_PERSIST_DIR, help="Vector index location"
    )
    blast_parser.add_argument(
        "--graph-db", default=DEFAULT_GRAPH_DB, help="Call graph location"
    )
    blast_parser.add_argument(
        "--no-llm", action="store_true", help="Skip LLM impact summary (faster)"
    )
    blast_parser.set_defaults(func=cmd_blast)

    # ── onboard (Phase 7) ─────────────────────────────────────────────────────
    onboard_parser = subparsers.add_parser(
        "onboard", help="Generate a guided reading order for new contributors"
    )
    onboard_parser.add_argument(
        "--index-dir", default=DEFAULT_PERSIST_DIR, help="Vector index location"
    )
    onboard_parser.add_argument(
        "--graph-db", default=DEFAULT_GRAPH_DB, help="Call graph location"
    )
    onboard_parser.add_argument(
        "--no-llm", action="store_true", help="Show topology only, skip LLM guide (faster)"
    )
    onboard_parser.set_defaults(func=cmd_onboard)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
