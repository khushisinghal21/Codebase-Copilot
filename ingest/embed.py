"""
Embedding wrapper.

Kept behind a small class so you can swap the model (e.g. to a
code-tuned embedding model, or an API-based one) later without
touching anything else in the pipeline - the guide specifically
flags this as worth doing up front.

Default: sentence-transformers/all-MiniLM-L6-v2 - free, runs locally,
no API key needed, good enough to prove the pipeline works end-to-end.
"""

from typing import Union

_MODEL_NAME = "all-MiniLM-L6-v2"


class Embedder:
    def __init__(self, model_name: str = _MODEL_NAME):
        # Imported lazily so `pip install` failures for this one
        # dependency don't break unrelated parts of the CLI.
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(
            texts,
            show_progress_bar=len(texts) > 20,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query])[0]


if __name__ == "__main__":
    e = Embedder()
    vecs = e.embed_texts(["def foo(): pass", "class Bar: pass"])
    print(f"Embedded {len(vecs)} texts, dim={len(vecs[0])}")
