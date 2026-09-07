"""
BM25 keyword index using SQLite FTS5.

Phase 2 adds this alongside ChromaDB to handle exact symbol name
queries that dense embeddings miss (e.g. "where is chunk_file called?").

SQLite FTS5 uses BM25 ranking by default — no extra dependencies needed,
it's built into Python's sqlite3 module.
"""

import re
import sqlite3
from pathlib import Path

DEFAULT_DB_PATH = ".copilot_bm25.db"


class BM25Store:
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._setup()

    def _setup(self) -> None:
        # Migration: if repo_id column is missing, drop and recreate
        try:
            self.conn.execute("SELECT repo_id FROM chunks_fts LIMIT 0")
        except Exception:
            self.conn.execute("DROP TABLE IF EXISTS chunks_fts")

        self.conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                text,
                file_path UNINDEXED,
                start_line UNINDEXED,
                end_line   UNINDEXED,
                symbol_name UNINDEXED,
                repo_id    UNINDEXED
            )
            """
        )
        self.conn.commit()

    def add_chunks(self, chunks: list[dict], repo_id: str = "default") -> None:
        """Insert chunks in the FTS index, tagged with repo_id."""
        rows = []
        for c in chunks:
            rows.append(
                (
                    c["text"],
                    c["file_path"],
                    c["start_line"],
                    c["end_line"],
                    c.get("symbol_name") or "",
                    repo_id,
                )
            )
        self.conn.executemany(
            "INSERT INTO chunks_fts (text, file_path, start_line, end_line, symbol_name, repo_id) VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        self.conn.commit()

    @staticmethod
    def _sanitize_query(query: str) -> str:
        """
        Convert a natural language query to a safe FTS5 MATCH expression.
        Strips FTS5 special syntax and wraps each word in quotes so the
        query can't accidentally trigger syntax errors.
        """
        # Extract alphanumeric tokens (including underscores for identifiers)
        tokens = re.findall(r"[\w]+", query)
        if not tokens:
            return '""'
        # Quote each token to treat it as a literal phrase
        return " OR ".join(f'"{t}"' for t in tokens)

    def search(self, query: str, k: int = 10, repo_id: str | None = None) -> list[dict]:
        """
        BM25 full-text search. Optionally filter by repo_id.
        Returns up to k results sorted by BM25 rank.
        """
        safe_query = self._sanitize_query(query)
        if not safe_query:
            return []

        repo_filter = f"AND repo_id = ?" if repo_id else ""
        params = (safe_query, k) if not repo_id else (safe_query, repo_id, k)

        try:
            cursor = self.conn.execute(
                f"""
                SELECT text, file_path, start_line, end_line, symbol_name, repo_id,
                       rank
                FROM   chunks_fts
                WHERE  chunks_fts MATCH ? {repo_filter}
                ORDER  BY rank
                LIMIT  ?
                """,
                params,
            )
        except Exception:
            return []

        hits = []
        for row in cursor:
            hits.append(
                {
                    "text":        row[0],
                    "file_path":   row[1],
                    "start_line":  int(row[2]),
                    "end_line":    int(row[3]),
                    "symbol_name": row[4] or None,
                    "repo_id":     row[5],
                    "score":       -row[6],
                }
            )
        return hits

    def reset(self) -> None:
        self.conn.execute("DELETE FROM chunks_fts")
        self.conn.commit()

    def delete_by_file(self, file_path: str) -> int:
        """
        Delete all BM25 entries for the given file.
        Returns the number of rows deleted.
        """
        cur = self.conn.execute(
            "DELETE FROM chunks_fts WHERE file_path = ?", (file_path,)
        )
        self.conn.commit()
        return cur.rowcount

    def delete_by_repo(self, repo_id: str) -> int:
        """Delete all BM25 entries for the given repo."""
        cur = self.conn.execute(
            "DELETE FROM chunks_fts WHERE repo_id = ?", (repo_id,)
        )
        self.conn.commit()
        return cur.rowcount

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]

    def close(self) -> None:
        self.conn.close()
