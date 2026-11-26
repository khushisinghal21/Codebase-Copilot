"""
Retrieval: embed the query, search both the vector store and the BM25
index, then fuse the ranked lists with Reciprocal Rank Fusion (RRF).
Optionally re-rank the shortlist with a cross-encoder (Phase 3).

Phase 1: dense-only
Phase 2: dense + BM25 + RRF
Phase 3: dense + BM25 + RRF + cross-encoder reranking
Phase 8: + wiki sections built by the offline dreaming pass, fused in by RRF

The function signature stays query -> list[chunk] so cli.py is unchanged.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

from ingest.embed import Embedder
from index.store import VectorStore
from index.bm25_store import BM25Store

if TYPE_CHECKING:
    from retrieve.rerank import Reranker
    from memory.wiki_store import WikiStore


def reciprocal_rank_fusion(
    dense_hits: list[dict],
    bm25_hits: list[dict],
    k: int = 60,
    top_n: int = 10,
) -> list[dict]:
    """
    Merge two ranked lists using RRF: score = Σ 1 / (k + rank).

    k=60 is the standard constant from the original RRF paper.
    NOTE: RRF scores are small by design (~0.01-0.03 range) and are NOT
    comparable to cosine similarity scores. Downstream code must handle this.
    """
    scores: dict[tuple, dict] = {}

    for rank, hit in enumerate(dense_hits):
        key = (hit["file_path"], hit["start_line"])
        if key not in scores:
            scores[key] = {**hit, "score": 0.0}
        scores[key]["score"] += 1.0 / (k + rank + 1)

    for rank, hit in enumerate(bm25_hits):
        key = (hit["file_path"], hit["start_line"])
        if key not in scores:
            scores[key] = {**hit, "score": 0.0}
        scores[key]["score"] += 1.0 / (k + rank + 1)

    merged = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
    # Tag results so answer.py knows to use RRF-appropriate confidence threshold
    for hit in merged:
        hit["is_rrf"] = True
    return merged[:top_n]


def fuse_with_wiki(
    query: str,
    candidates: list[dict],
    wiki_store: "WikiStore",
    bm25_store: BM25Store,
    top_n: int,
    repo_id: str | None = None,
) -> list[dict]:
    """
    Phase 8: add the wiki as a third ranked list and fuse with RRF.

    The wiki ranks symbols by their file's purpose, call relations and the
    past questions they answered, then each section is resolved back to its
    code chunk. Chunks found by both search and the wiki rise to the top.
    """
    from memory.wiki_store import wiki_hits_to_chunks

    wiki_hits = wiki_store.search(query, k=top_n)
    wiki_chunks = wiki_hits_to_chunks(wiki_hits, bm25_store, repo_id=repo_id)
    if not wiki_chunks:
        return candidates[:top_n]
    return reciprocal_rank_fusion(candidates, wiki_chunks, top_n=top_n)


def search(
    query: str,
    embedder: Embedder | None,
    store: VectorStore | None,
    bm25_store: BM25Store | None = None,
    reranker: "Reranker | None" = None,
    k: int = 10,
    repo_id: str | None = None,
    wiki_store: "WikiStore | None" = None,
) -> list[dict]:
    """
    Retrieve the top-k most relevant chunks for a query.

    Pipeline:
      1. Dense vector search (if embedder and store provided)
      2. BM25 keyword search + RRF fusion (if bm25_store provided)
      3. Wiki sections from the last dream, fused by RRF (if wiki_store provided)
      4. Cross-encoder reranking of top candidates (if reranker provided)

    repo_id: if set, restrict search to that repo only (Phase 6).
             If None, searches across all indexed repos.
    """
    # Fetch more candidates if reranking — reranker needs material to work with
    fetch_k = min(k * 2, max(k + 5, 20)) if not reranker else max(k * 3, 20)

    if embedder is not None and store is not None:
        query_embedding = embedder.embed_query(query)
        dense_hits = store.query(query_embedding, k=fetch_k, repo_id=repo_id)
    else:
        dense_hits = []  # BM25-only mode (no embedding model available)

    if bm25_store is None:
        candidates = dense_hits[:k]
    elif not dense_hits:
        candidates = bm25_store.search(query, k=fetch_k, repo_id=repo_id)
    else:
        bm25_hits = bm25_store.search(query, k=fetch_k, repo_id=repo_id)
        if not bm25_hits:
            candidates = dense_hits[:fetch_k]
        else:
            candidates = reciprocal_rank_fusion(dense_hits, bm25_hits, top_n=fetch_k)

    # Phase 8: fuse in the dreamed wiki before reranking
    if wiki_store is not None and bm25_store is not None:
        candidates = fuse_with_wiki(query, candidates, wiki_store, bm25_store,
                                    top_n=fetch_k, repo_id=repo_id)

    # Phase 3: cross-encoder reranking
    if reranker is not None and candidates:
        return reranker.rerank(query, candidates, top_n=k)

    return candidates[:k]


