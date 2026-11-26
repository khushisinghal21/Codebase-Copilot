"""
Searchable index over the wiki: one FTS5 row per symbol section.

The markdown pages in wiki/ are for humans (and for the LLM to read); this
index is what retrieval queries. Each row describes one symbol with what the
raw chunk often lacks: the file's module docstring, its call relations, and
the past questions that symbol has answered.

Column weights are fixed a priori and were not tuned on the eval set:
  name 1.0 | doc 1.0 | context 0.5 (module docstring, repeated per section) | learned 1.0
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from index.bm25_store import BM25Store

WIKI_DB_FILE = "wiki.db"
COLUMN_WEIGHTS = (1.0, 1.0, 0.5, 1.0)


class WikiStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS wiki_fts USING fts5(
                name_text,
                doc_text,
                context_text,
                learned_text,
                rel_path    UNINDEXED,
                symbol_name UNINDEXED
            )
            """
        )
        self.conn.commit()

    def rebuild(self, sections: list[dict]) -> None:
        self.conn.execute("DELETE FROM wiki_fts")
        self.conn.executemany(
            """
            INSERT INTO wiki_fts (name_text, doc_text, context_text, learned_text, rel_path, symbol_name)
            VALUES (:name_text, :doc_text, :context_text, :learned_text, :rel_path, :symbol_name)
            """,
            sections,
        )
        self.conn.commit()

    def search(self, query: str, k: int = 20) -> list[dict]:
        safe = BM25Store._sanitize_query(query)
        if not safe or safe == '""':
            return []
        w = ", ".join(str(x) for x in COLUMN_WEIGHTS)
        try:
            cur = self.conn.execute(
                f"""
                SELECT rel_path, symbol_name, bm25(wiki_fts, {w}) AS r
                FROM wiki_fts WHERE wiki_fts MATCH ?
                ORDER BY r LIMIT ?
                """,
                (safe, k),
            )
        except sqlite3.OperationalError:
            return []
        return [{"rel_path": r[0], "symbol_name": r[1], "score": -r[2]} for r in cur]

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM wiki_fts").fetchone()[0]

    def close(self) -> None:
        self.conn.close()


def wiki_hits_to_chunks(wiki_hits: list[dict], bm25_store: BM25Store, repo_id: str | None = None) -> list[dict]:
    """Resolve wiki sections to indexed chunks, keeping wiki rank order."""
    chunks: list[dict] = []
    seen: set[tuple] = set()
    for hit in wiki_hits:
        for c in bm25_store.get_symbol_chunks(hit["rel_path"], hit["symbol_name"], repo_id=repo_id):
            key = (c["file_path"], c["start_line"])
            if key not in seen:
                seen.add(key)
                chunks.append({**c, "from_wiki": True})
    return chunks
