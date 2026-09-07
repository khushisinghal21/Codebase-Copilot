"""
Vector store wrapper around ChromaDB.

ChromaDB is chosen for Phase 1 because it needs zero infrastructure
(no Postgres/Docker) - just a local persistent directory. The guide's
Phase 2 suggests moving to pgvector once you add BM25 + rank fusion,
since you'll want one database for both dense and keyword search.
The interface below (add_chunks / query) is written so that swap is
mechanical: only this file changes.
"""

from pathlib import Path

import chromadb

DEFAULT_PERSIST_DIR = ".copilot_index"
COLLECTION_NAME = "codebase_chunks"


class VectorStore:
    def __init__(self, persist_dir: str = DEFAULT_PERSIST_DIR):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[dict], embeddings: list[list[float]]) -> None:
        """
        chunks: list of {file_path, start_line, end_line, text}
        embeddings: parallel list of embedding vectors
        """
        if not chunks:
            return

        ids = [
            f"{c['file_path']}:{c['start_line']}-{c['end_line']}:{i}"
            for i, c in enumerate(chunks)
        ]
        documents = [c["text"] for c in chunks]
        metadatas = []
        for c in chunks:
            meta = {
                "file_path": c["file_path"],
                "start_line": c["start_line"],
                "end_line": c["end_line"],
            }
            # Phase 2 AST metadata (optional — not present in Phase 1 chunks)
            for key in ("symbol_name", "symbol_type", "parent_class"):
                if c.get(key) is not None:
                    meta[key] = c[key]
            metadatas.append(meta)


        # Chroma has a batch size ceiling; chunk the insert to be safe on large repos
        BATCH = 500
        for i in range(0, len(ids), BATCH):
            self.collection.upsert(
                ids=ids[i : i + BATCH],
                embeddings=embeddings[i : i + BATCH],
                documents=documents[i : i + BATCH],
                metadatas=metadatas[i : i + BATCH],
            )

    def query(self, query_embedding: list[float], k: int = 10) -> list[dict]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
        )
        hits = []
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            hits.append(
                {
                    "text": doc,
                    "file_path": meta["file_path"],
                    "start_line": meta["start_line"],
                    "end_line": meta["end_line"],
                    "score": 1 - dist,  # cosine distance -> similarity
                }
            )
        return hits

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def delete_by_file(self, file_path: str) -> int:
        """
        Delete all chunks belonging to the given file.
        Returns the number of chunks deleted.
        """
        results = self.collection.get(where={"file_path": {"$eq": file_path}})
        ids = results.get("ids", [])
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)
