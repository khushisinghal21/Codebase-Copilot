"""
Phase 3 reranking: cross-encoder re-scores the top-k candidates
from RRF with much higher accuracy.

Why two stages?
  - Bi-encoders (Phase 2) encode query and chunk separately → fast, but
    miss fine-grained query-chunk interactions.
  - Cross-encoders encode (query, chunk) together → slow, but accurate.
  - Solution: RRF narrows 35+ chunks to top-20, cross-encoder re-ranks
    that shortlist to top-8.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - ~80 MB, runs locally, no API key needed.
  - Scores are logits (not probabilities). Typical range: -10 to +10.
    Scores > 0 are generally relevant; < -5 are almost certainly not.
"""

from sentence_transformers import CrossEncoder

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Chunks whose rerank score is below this are considered irrelevant.
# ms-marco logits: 0 ≈ borderline relevant, -5 ≈ clearly irrelevant.
RERANK_THRESHOLD = -5.0


class Reranker:
    def __init__(self, model_name: str = _MODEL_NAME):
        # Lazy import already handled by sentence_transformers
        self.model = CrossEncoder(model_name)

    def rerank(
        self, query: str, hits: list[dict], top_n: int = 8
    ) -> list[dict]:
        """
        Re-score hits by running (query, chunk_text) through the cross-encoder.
        Returns up to top_n hits, sorted by rerank score descending.
        """
        if not hits:
            return hits

        pairs = [(query, h["text"]) for h in hits]
        raw_scores = self.model.predict(pairs).tolist()

        reranked = []
        for hit, raw_score in zip(hits, raw_scores):
            updated = {**hit, "score": raw_score, "is_rrf": False, "is_reranked": True}
            reranked.append(updated)

        reranked.sort(key=lambda x: x["score"], reverse=True)
        return reranked[:top_n]


if __name__ == "__main__":
    r = Reranker()
    hits = [
        {"text": "def walk_repo(repo_path): ...", "file_path": "walk_repo.py",
         "start_line": 1, "end_line": 10},
        {"text": "def chunk_file(file_path): ...", "file_path": "chunk.py",
         "start_line": 1, "end_line": 10},
    ]
    results = r.rerank("what does walk_repo do?", hits)
    for h in results:
        print(f"{h['file_path']}  score={h['score']:.3f}")
