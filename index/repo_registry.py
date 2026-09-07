"""
Phase 6: Repo registry — tracks every repo that has been indexed.

Stored as a `repos` table in .copilot_graph.db so it lives alongside
the call graph and metadata (no extra file).

Schema:
  repo_id      — short name used for filtering, e.g. "backend"
  repo_path    — absolute path on disk
  last_commit  — commit SHA at time of last index/update
  indexed_at   — ISO timestamp of last indexing
  chunk_count  — number of chunks currently in the index for this repo
"""

import sqlite3
from datetime import datetime, timezone


class RepoRegistry:
    def __init__(self, conn: sqlite3.Connection):
        """Shares the graph store's SQLite connection."""
        self.conn = conn
        self._setup()

    def _setup(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS repos (
                repo_id      TEXT PRIMARY KEY,
                repo_path    TEXT NOT NULL,
                last_commit  TEXT,
                indexed_at   TEXT,
                chunk_count  INTEGER DEFAULT 0
            )
            """
        )
        self.conn.commit()

    # ── write ─────────────────────────────────────────────────────────────────

    def register(
        self,
        repo_id: str,
        repo_path: str,
        last_commit: str | None = None,
        chunk_count: int = 0,
    ) -> None:
        """Register or update a repo entry."""
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            """
            INSERT INTO repos (repo_id, repo_path, last_commit, indexed_at, chunk_count)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(repo_id) DO UPDATE SET
                repo_path   = excluded.repo_path,
                last_commit = excluded.last_commit,
                indexed_at  = excluded.indexed_at,
                chunk_count = excluded.chunk_count
            """,
            (repo_id, repo_path, last_commit, now, chunk_count),
        )
        self.conn.commit()

    def update_chunk_count(self, repo_id: str, delta: int) -> None:
        """Increment/decrement the chunk counter."""
        self.conn.execute(
            "UPDATE repos SET chunk_count = MAX(0, chunk_count + ?) WHERE repo_id = ?",
            (delta, repo_id),
        )
        self.conn.commit()

    def remove(self, repo_id: str) -> None:
        """Deregister a repo (call after deleting its chunks)."""
        self.conn.execute("DELETE FROM repos WHERE repo_id = ?", (repo_id,))
        self.conn.commit()

    # ── read ──────────────────────────────────────────────────────────────────

    def get(self, repo_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT repo_id, repo_path, last_commit, indexed_at, chunk_count FROM repos WHERE repo_id = ?",
            (repo_id,),
        ).fetchone()
        return _row_to_dict(row) if row else None

    def list_all(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT repo_id, repo_path, last_commit, indexed_at, chunk_count FROM repos ORDER BY indexed_at DESC"
        ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def all_repo_ids(self) -> list[str]:
        return [r["repo_id"] for r in self.list_all()]


def _row_to_dict(row) -> dict:
    return {
        "repo_id": row[0],
        "repo_path": row[1],
        "last_commit": row[2],
        "indexed_at": row[3],
        "chunk_count": row[4],
    }


def derive_repo_id(repo_path: str) -> str:
    """Fall back to the folder name if --repo-name not specified."""
    from pathlib import Path
    return Path(repo_path).resolve().name
