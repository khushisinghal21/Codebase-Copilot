"""
Transcript log: the bottom layer of memory.

Every `ask` session is appended to a JSONL file. Nothing here is summarised or
interpreted; it is the ground truth that `dream.py` checks notes against.

A session records:
  question   the user's question
  retrieved  the chunks the retriever returned, in rank order
  confirmed  chunks known to have answered the question: citations that passed
             validation, or answers a user confirmed with `cli.py feedback`
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from memory import DEFAULT_MEMORY_DIR

TRANSCRIPTS_FILE = "transcripts.jsonl"


def _ref(chunk: dict) -> dict:
    return {
        "file_path": chunk["file_path"],
        "symbol_name": chunk.get("symbol_name") or None,
        "start_line": int(chunk["start_line"]),
        "end_line": int(chunk["end_line"]),
    }


class TranscriptLog:
    def __init__(self, memory_dir: str = DEFAULT_MEMORY_DIR):
        self.dir = Path(memory_dir)
        self.path = self.dir / TRANSCRIPTS_FILE

    def log_session(
        self,
        question: str,
        retrieved: list[dict],
        confirmed: list[dict] | None = None,
        source: str = "ask",
    ) -> str:
        """Append one session and return its id."""
        self.dir.mkdir(parents=True, exist_ok=True)
        session_id = uuid.uuid4().hex[:12]
        record = {
            "id": session_id,
            "ts": time.time(),
            "source": source,
            "question": question,
            "retrieved": [_ref(c) for c in retrieved],
            "confirmed": [_ref(c) for c in (confirmed or [])],
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        return session_id

    def add_confirmation(self, session_id: str, chunk: dict) -> bool:
        """Record that `chunk` answered a past session (explicit user feedback)."""
        sessions = self.read_all()
        found = False
        for s in sessions:
            if s["id"] == session_id:
                s["confirmed"].append(_ref(chunk))
                found = True
        if found:
            self._rewrite(sessions)
        return found

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        sessions = []
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    sessions.append(json.loads(line))
        return sessions

    def _rewrite(self, sessions: list[dict]) -> None:
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for s in sessions:
                f.write(json.dumps(s) + "\n")
        tmp.replace(self.path)


def confirmed_from_citations(chunks: list[dict], valid_citations: list[tuple]) -> list[dict]:
    """
    Map validated (path, start, end) citations back to the chunks they point at.
    A citation the validator accepted is the weakest useful positive signal:
    the model relied on that chunk and the reference was real.
    """
    confirmed = []
    for path_str, start, end in valid_citations:
        for c in chunks:
            if c["file_path"].endswith(path_str) or path_str in c["file_path"]:
                if start <= c["end_line"] and c["start_line"] <= end:
                    if c not in confirmed:
                        confirmed.append(c)
    return confirmed
