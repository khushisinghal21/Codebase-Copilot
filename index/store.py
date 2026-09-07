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


def _chunk_id(chunk: dict) -> str:
    """Stable unique ID for a chunk — used as ChromaDB document ID."""
    return f"{chunk['file_path']}:{chunk['start_line']}-{chunk['end_line']}"


class VectorStore:
    def __init__(self, persist_dir: str = DEFAULT_PERSIST_DIR):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[dict], embeddings: list[list[float]], repo_id: str = "default") -> None:
        """
        chunks: list of {file_path, start_line, end_line, text}
        embeddings: parallel list of embedding vectors
        """
        if not chunks:
            return

        ids = [_chunk_id(c) for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [
            {
                "file_path": c["file_path"],
                "start_line": c["start_line"],
                "end_line": c["end_line"],
                "symbol_name": c.get("symbol_name") or "",
                "symbol_type": c.get("symbol_type") or "",
                "parent_class": c.get("parent_class") or "",
                "repo_id": repo_id,
            }
            for c in chunks
        ]

        # Chroma has a batch size ceiling; chunk the insert to be safe on large repos
        BATCH = 500
        for i in range(0, len(ids), BATCH):
            self.collection.upsert(
                ids=ids[i : i + BATCH],
                embeddings=embeddings[i : i + BATCH],
                documents=documents[i : i + BATCH],
                metadatas=metadatas[i : i + BATCH],
            )

    def query(
        self,
        query_embedding: list[float],
        k: int = 10,
        repo_id: str | None = None,
    ) -> list[dict]:
        """
        Find the top-k nearest chunks by cosine similarity.
        If repo_id is given, restrict search to that repo only.
        """
        where = {"repo_id": {"$eq": repo_id}} if repo_id else None
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self.count()),
            include=["documents", "metadatas", "distances"],
            where=where,
        )
        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append(
                {
                    "text": doc,
                    "file_path": meta["file_path"],
                    "start_line": meta["start_line"],
                    "end_line": meta["end_line"],
                    "symbol_name": meta.get("symbol_name"),
                    "symbol_type": meta.get("symbol_type"),
                    "parent_class": meta.get("parent_class"),
                    "repo_id": meta.get("repo_id", "default"),
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

    def delete_by_repo(self, repo_id: str) -> int:
        """Delete all chunks belonging to a repo. Returns count deleted."""
        results = self.collection.get(where={"repo_id": {"$eq": repo_id}})
        ids = results.get("ids", [])
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)
