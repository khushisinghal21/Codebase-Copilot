"""
Evaluation script: measures retrieval quality against the gold set.

Metrics:
  recall@k  — did the expected chunk appear anywhere in the top-k results?
  MRR       — Mean Reciprocal Rank (1/rank of first correct hit, averaged)

Usage:
  python eval/run_eval.py
  python eval/run_eval.py --k 5
  python eval/run_eval.py --no-rerank   # compare reranked vs raw RRF
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Make sure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from ingest.embed import Embedder
from index.store import VectorStore
from index.bm25_store import BM25Store
from retrieve.search import search

load_dotenv()

GOLD_SET_PATH = Path(__file__).parent / "gold_set.json"


def _chunk_matches(chunk: dict, expected_file: str, expected_symbol: str) -> bool:
    """
    A retrieved chunk is a match if:
      1. Its file path ends with expected_file, AND
      2. Its symbol_name matches expected_symbol (for AST chunks), OR
         the expected_symbol string appears in the chunk text (fallback).
    """
    file_match = chunk["file_path"].endswith(expected_file.replace("/", os.sep))
    if not file_match:
        return False

    # AST chunk with symbol metadata
    if chunk.get("symbol_name") == expected_symbol:
        return True

    # Fallback: symbol name appears in the chunk text
    if expected_symbol in chunk.get("text", ""):
        return True

    return False


def run_eval(k: int = 10, use_reranker: bool = True) -> dict:
    gold = json.loads(GOLD_SET_PATH.read_text())

    print(f"Loading models and indexes...")
    embedder = Embedder()
    store = VectorStore()
    bm25 = BM25Store()

    reranker = None
    if use_reranker:
        try:
            from retrieve.rerank import Reranker
            reranker = Reranker()
            print(f"Reranker loaded ✅")
        except Exception as e:
            print(f"Reranker unavailable ({e}), running without reranking.")

    print(f"\nRunning eval on {len(gold)} questions (k={k})...\n")
    print(f"{'#':<4} {'Hit?':<6} {'Rank':<6} {'Question'}")
    print("-" * 70)

    hits_at_k = 0
    reciprocal_ranks = []
    results = []

    for i, entry in enumerate(gold, 1):
        question = entry["question"]
        expected_file = entry["expected_file"]
        expected_symbol = entry["expected_symbol"]

        chunks = search(question, embedder, store, bm25_store=bm25, k=k)

        if reranker and chunks:
            chunks = reranker.rerank(question, chunks, top_n=k)

        # Find rank of first correct chunk (1-indexed)
        hit_rank = None
        for rank, chunk in enumerate(chunks, 1):
            if _chunk_matches(chunk, expected_file, expected_symbol):
                hit_rank = rank
                break

        hit = hit_rank is not None
        if hit:
            hits_at_k += 1
            reciprocal_ranks.append(1.0 / hit_rank)
        else:
            reciprocal_ranks.append(0.0)

        rank_str = str(hit_rank) if hit_rank else "—"
        hit_str = "✅" if hit else "❌"
        short_q = question[:55] + "..." if len(question) > 55 else question
        print(f"{i:<4} {hit_str:<6} {rank_str:<6} {short_q}")

        results.append({
            "question": question,
            "expected_file": expected_file,
            "expected_symbol": expected_symbol,
            "hit": hit,
            "rank": hit_rank,
        })

    recall_at_k = hits_at_k / len(gold)
    mrr = sum(reciprocal_ranks) / len(gold)

    print("\n" + "=" * 70)
    print(f"  recall@{k}  : {recall_at_k:.3f}  ({hits_at_k}/{len(gold)} questions answered correctly)")
    print(f"  MRR       : {mrr:.3f}")
    print("=" * 70)

    return {
        "k": k,
        "recall_at_k": recall_at_k,
        "mrr": mrr,
        "results": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality")
    parser.add_argument("--k", type=int, default=10, help="Top-k to evaluate (default 10)")
    parser.add_argument("--no-rerank", action="store_true", help="Skip cross-encoder reranking")
    args = parser.parse_args()

    run_eval(k=args.k, use_reranker=not args.no_rerank)
