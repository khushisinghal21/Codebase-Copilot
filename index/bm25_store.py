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
        # FTS5 with porter stemmer for better recall on word variants
        self.conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_id,
                text,
                file_path UNINDEXED,
                start_line UNINDEXED,
                end_line UNINDEXED,
                tokenize = 'porter ascii'
            )
            """
        )
        self.conn.commit()

    def add_chunks(self, chunks: list[dict]) -> None:
        """Insert or replace chunks in the FTS index."""
        rows = []
        for i, c in enumerate(chunks):
            chunk_id = f"{c['file_path']}:{c['start_line']}-{c['end_line']}:{i}"
            rows.append(
                (chunk_id, c["text"], c["file_path"], c["start_line"], c["end_line"])
            )
        self.conn.executemany(
            "INSERT OR REPLACE INTO chunks_fts VALUES (?, ?, ?, ?, ?)", rows
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

    def search(self, query: str, k: int = 10) -> list[dict]:
        """Return top-k chunks matching the query by BM25 rank."""
        fts_query = self._sanitize_query(query)
        try:
            cursor = self.conn.execute(
                """
                SELECT file_path, start_line, end_line, text, rank
                FROM chunks_fts
                WHERE chunks_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (fts_query, k),
            )
        except sqlite3.OperationalError:
            # FTS5 syntax error — return empty rather than crashing
            return []

        hits = []
        for row in cursor:
            hits.append(
                {
                    "file_path": row[0],
                    "start_line": row[1],
                    "end_line": row[2],
                    "text": row[3],
                    "score": -row[4],   # FTS5 rank is negative; flip for consistency
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

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]

    def close(self) -> None:
        self.conn.close()
