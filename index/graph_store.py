"""
SQLite-backed call graph store.

Two tables:
  symbols — every indexed function/class/method with its file location
  calls   — directed edges (caller_symbol → callee_name)

This lets us answer:
  "What does chunk_file call?"  → callees of chunk_file
  "What calls chunk_file?"      → callers of chunk_file
  "Where is walk_repo defined?" → symbol location lookup
"""

import sqlite3

DEFAULT_GRAPH_DB = ".copilot_graph.db"


class GraphStore:
    def __init__(self, db_path: str = DEFAULT_GRAPH_DB):
        self.conn = sqlite3.connect(db_path)
        self._setup()

    def _setup(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS symbols (
                file_path    TEXT NOT NULL,
                symbol_name  TEXT NOT NULL,
                symbol_type  TEXT,
                parent_class TEXT,
                start_line   INTEGER,
                end_line     INTEGER,
                PRIMARY KEY (file_path, symbol_name)
            );

            CREATE TABLE IF NOT EXISTS calls (
                caller_file   TEXT NOT NULL,
                caller_symbol TEXT NOT NULL,
                callee_name   TEXT NOT NULL,
                PRIMARY KEY (caller_file, caller_symbol, callee_name)
            );

            -- Phase 5: key-value metadata (e.g. last indexed commit SHA)
            CREATE TABLE IF NOT EXISTS meta (
                key   TEXT PRIMARY KEY,
                value TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_symbol_name  ON symbols(symbol_name);
            CREATE INDEX IF NOT EXISTS idx_callee_name  ON calls(callee_name);
            CREATE INDEX IF NOT EXISTS idx_caller_symbol ON calls(caller_symbol);
            """
        )
        self.conn.commit()

    # ── write ─────────────────────────────────────────────────────────────────

    def add_symbols(self, chunks: list[dict]) -> None:
        """Register every AST chunk that has a symbol_name."""
        rows = [
            (
                c["file_path"],
                c["symbol_name"],
                c.get("symbol_type"),
                c.get("parent_class"),
                c["start_line"],
                c["end_line"],
            )
            for c in chunks
            if c.get("symbol_name")
        ]
        if rows:
            self.conn.executemany(
                """
                INSERT OR REPLACE INTO symbols
                  (file_path, symbol_name, symbol_type, parent_class, start_line, end_line)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            self.conn.commit()

    def add_calls(self, edges: list[tuple]) -> None:
        """
        edges: list of (caller_file, caller_symbol, callee_name)
        Duplicates are silently ignored (PRIMARY KEY constraint).
        """
        self.conn.executemany(
            "INSERT OR IGNORE INTO calls (caller_file, caller_symbol, callee_name) VALUES (?,?,?)",
            edges,
        )
        self.conn.commit()

    # ── read ──────────────────────────────────────────────────────────────────

    def get_callees(self, symbol_name: str) -> list[str]:
        """Names of functions/methods that symbol_name calls."""
        cur = self.conn.execute(
            "SELECT DISTINCT callee_name FROM calls WHERE caller_symbol = ?",
            (symbol_name,),
        )
        return [r[0] for r in cur]

    def get_callers(self, symbol_name: str) -> list[dict]:
        """All (file_path, symbol_name) pairs that call this symbol."""
        cur = self.conn.execute(
            "SELECT DISTINCT caller_file, caller_symbol FROM calls WHERE callee_name = ?",
            (symbol_name,),
        )
        return [{"file_path": r[0], "symbol_name": r[1]} for r in cur]

    def get_symbol_location(self, symbol_name: str) -> list[dict]:
        """Find where a symbol is defined (may match across files)."""
        cur = self.conn.execute(
            """
            SELECT file_path, start_line, end_line, symbol_type, parent_class
            FROM symbols WHERE symbol_name = ?
            """,
            (symbol_name,),
        )
        return [
            {
                "file_path": r[0],
                "start_line": r[1],
                "end_line": r[2],
                "symbol_type": r[3],
                "parent_class": r[4],
            }
            for r in cur
        ]

    def reset(self) -> None:
        self.conn.executescript("DELETE FROM symbols; DELETE FROM calls;")
        self.conn.commit()

    def delete_by_file(self, file_path: str) -> int:
        """
        Remove all symbols and call edges for the given file.
        Returns total rows deleted.
        """
        cur1 = self.conn.execute(
            "DELETE FROM symbols WHERE file_path = ?", (file_path,)
        )
        cur2 = self.conn.execute(
            "DELETE FROM calls WHERE caller_file = ?", (file_path,)
        )
        self.conn.commit()
        return cur1.rowcount + cur2.rowcount

    # ── metadata (Phase 5) ────────────────────────────────────────────────────

    def get_meta(self, key: str) -> str | None:
        """Retrieve a metadata value by key."""
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        """Store a metadata value."""
        self.conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            (key, value),
        )
        self.conn.commit()
