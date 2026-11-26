"""
The offline "dreaming" pass.

Run it between sessions (`python cli.py dream <repo>`). It never runs during
an `ask`, so it costs nothing at question time. In order, it:

  1. Consolidates new transcripts into memory notes: which symbol answered
     which question, and how far down search had ranked it.
  2. Verifies every note against the current code. A note whose symbol moved
     is repointed to the new lines (or new file); a note whose symbol is gone
     is dropped. This is what stops a stale note from producing a wrong
     path:line citation after a refactor.
  3. Rebuilds the wiki: one linked markdown page per file, plus a search index
     with one section per symbol.
  4. Writes MEMORY.md, a capped index that gets injected into every answer.
  5. Writes dream_report.json, including a bloat check (wiki size vs source).

This pass is rule-based: it makes no LLM calls, so it is cheap, deterministic
and safe to run on every commit.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from memory import DEFAULT_MEMORY_DIR
from memory.codemap import FileInfo, load_codemap, to_rel
from memory.transcripts import TranscriptLog
from memory.wiki_store import WIKI_DB_FILE, WikiStore

NOTES_FILE = "notes.json"
INDEX_FILE = "MEMORY.md"
REPORT_FILE = "dream_report.json"
WIKI_DIR = "wiki"

DEFAULT_MAX_INDEX_LINES = 150
MAX_QUESTIONS_PER_NOTE = 8


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _page_name(rel_path: str) -> str:
    return rel_path.replace("/", "__") + ".md"


def _split_ident(name: str) -> str:
    """walk_repo -> 'walk repo', BM25Store -> 'BM25 Store'."""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)
    return spaced.replace("_", " ").strip()


# ── 1. consolidate transcripts into notes ────────────────────────────────────

def _load_notes(memory_dir: Path) -> dict:
    path = memory_dir / NOTES_FILE
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"processed_sessions": [], "notes": []}


def consolidate(sessions: list[dict], store: dict, codemap: dict[str, FileInfo]) -> int:
    """Fold unprocessed sessions into notes. Returns how many sessions were new."""
    processed = set(store["processed_sessions"])
    by_key = {(n["file"], n["symbol"]): n for n in store["notes"]}
    new_sessions = 0

    for s in sessions:
        if s["id"] in processed:
            continue
        processed.add(s["id"])
        new_sessions += 1
        retrieved_keys = []
        for r in s.get("retrieved", []):
            rel = to_rel(r["file_path"], codemap)
            retrieved_keys.append((rel, r.get("symbol_name")))

        for c in s.get("confirmed", []):
            if not c.get("symbol_name"):
                continue  # fixed-size chunks have no stable identity to remember
            rel = to_rel(c["file_path"], codemap) or c["file_path"]
            key = (rel, c["symbol_name"])
            rank = retrieved_keys.index(key) + 1 if key in retrieved_keys else None
            note = by_key.get(key)
            if note is None:
                note = {
                    "file": rel,
                    "symbol": c["symbol_name"],
                    "start_line": c["start_line"],
                    "end_line": c["end_line"],
                    "questions": [],
                    "sessions": [],
                    "question_ranks": {},
                    "created": s.get("ts", time.time()),
                }
                by_key[key] = note
            if s["question"] not in note["questions"]:
                note["questions"].append(s["question"])
                note["questions"] = note["questions"][-MAX_QUESTIONS_PER_NOTE:]
            note["sessions"].append(s["id"])
            # where search ranked this symbol for that question (None = missed)
            note["question_ranks"][s["question"]] = rank

    store["processed_sessions"] = sorted(processed)
    store["notes"] = list(by_key.values())
    return new_sessions


# ── 2. verify notes against the current code ─────────────────────────────────

def verify(store: dict, codemap: dict[str, FileInfo]) -> dict:
    """Repoint moved notes, drop notes whose symbol no longer exists."""
    kept, repointed, moved, dropped = [], [], [], []
    for note in store["notes"]:
        info = codemap.get(note["file"])
        sym = info.symbols.get(note["symbol"]) if info else None

        if sym is None:
            # Did the symbol move to another file? Only follow a unique match.
            elsewhere = [f for f in codemap.values() if note["symbol"] in f.symbols]
            if len(elsewhere) == 1:
                old = f"{note['file']}:{note['start_line']}-{note['end_line']}"
                note["file"] = elsewhere[0].rel_path
                sym = elsewhere[0].symbols[note["symbol"]]
                moved.append({"symbol": note["symbol"], "from": old,
                              "to": f"{note['file']}:{sym.start_line}-{sym.end_line}"})
            else:
                dropped.append({"symbol": note["symbol"],
                                "was": f"{note['file']}:{note['start_line']}-{note['end_line']}",
                                "reason": "symbol no longer exists" if not elsewhere
                                          else "symbol is ambiguous across files"})
                continue

        if (sym.start_line, sym.end_line) != (note["start_line"], note["end_line"]):
            repointed.append({"symbol": note["symbol"],
                              "from": f"{note['file']}:{note['start_line']}-{note['end_line']}",
                              "to": f"{note['file']}:{sym.start_line}-{sym.end_line}"})
            note["start_line"], note["end_line"] = sym.start_line, sym.end_line
        note["verified_at"] = time.time()
        kept.append(note)

    store["notes"] = kept
    return {"kept": len(kept), "repointed": repointed, "moved": moved, "dropped": dropped}


# ── 3. build the wiki ────────────────────────────────────────────────────────

def _defining_files(codemap: dict[str, FileInfo]) -> dict[str, list[str]]:
    where: dict[str, list[str]] = {}
    for rel, info in codemap.items():
        for name in info.symbols:
            where.setdefault(name, []).append(rel)
    return where


def build_wiki(codemap: dict[str, FileInfo], notes: list[dict], memory_dir: Path) -> dict:
    wiki_dir = memory_dir / WIKI_DIR
    wiki_dir.mkdir(parents=True, exist_ok=True)
    for old in wiki_dir.glob("*.md"):
        old.unlink()

    where = _defining_files(codemap)
    notes_by_key = {(n["file"], n["symbol"]): n for n in notes}
    called_by: dict[str, list[str]] = {}
    for rel, info in codemap.items():
        for sym in info.symbols.values():
            for callee in sym.calls:
                called_by.setdefault(callee, []).append(sym.name)

    sections: list[dict] = []
    wiki_bytes = 0

    for rel in sorted(codemap):
        info = codemap[rel]
        purpose = " ".join(info.module_doc.split("\n\n")[0].split()) if info.module_doc else ""
        lines = [f"# {rel}", ""]
        if purpose:
            lines += [f"> {purpose}", ""]
        related: set[str] = set()

        if info.symbols:
            lines += ["## Symbols", ""]
        for sym in info.symbols.values():
            owner = f"{sym.parent_class}." if sym.parent_class else ""
            lines.append(f"### `{owner}{sym.name}` ({sym.symbol_type}, lines {sym.start_line}-{sym.end_line})")
            if sym.signature:
                lines.append(f"`{sym.signature}`")
            doc = " ".join(sym.docstring.split("\n\n")[0].split()) if sym.docstring else ""
            if doc:
                lines.append(doc)

            links = []
            for callee in sym.calls:
                targets = where.get(callee, [])
                if len(targets) == 1:
                    links.append(f"[[{targets[0]}#{callee}]]")
                    if targets[0] != rel:
                        related.add(targets[0])
            if links:
                lines.append("Calls: " + ", ".join(links))

            note = notes_by_key.get((rel, sym.name))
            if note:
                lines.append("Answered before:")
                lines += [f'- "{q}"' for q in note["questions"]]
            lines.append("")

            sections.append({
                "name_text": f"{rel} {sym.name} {_split_ident(sym.name)} {sym.parent_class or ''} {sym.symbol_type}",
                "doc_text": f"{sym.signature} {doc}",
                "context_text": f"{purpose} calls {' '.join(sym.calls)} called by {' '.join(called_by.get(sym.name, []))}",
                "learned_text": " ".join(note["questions"]) if note else "",
                "rel_path": rel,
                "symbol_name": sym.name,
            })

        if related:
            lines += ["## Related files", ""]
            lines += [f"- [[{r}]]" for r in sorted(related)]
            lines.append("")

        page = "\n".join(lines)
        (wiki_dir / _page_name(rel)).write_text(page, encoding="utf-8")
        wiki_bytes += len(page.encode("utf-8"))

    store = WikiStore(str(memory_dir / WIKI_DB_FILE))
    store.rebuild(sections)
    store.close()
    return {"pages": len(codemap), "sections": len(sections), "wiki_bytes": wiki_bytes}


# ── 4. the injected index ────────────────────────────────────────────────────

def build_index(codemap: dict[str, FileInfo], notes: list[dict], max_lines: int) -> str:
    """
    MEMORY.md is loaded into every answer, so it stays small: a map of modules
    and the few lessons worth paying tokens for. Everything else lives in the
    wiki, one search away. Lower sections are cut first when over the cap.
    """
    header = [
        "# Codebase memory",
        "_Generated by dream.py. Do not edit; it is rebuilt from transcripts and code._",
        "",
    ]
    modules = ["## Modules", ""]
    for rel in sorted(codemap):
        info = codemap[rel]
        if not info.symbols:
            continue
        purpose = " ".join(info.module_doc.split("\n\n")[0].split())[:90] if info.module_doc else ""
        modules.append(f"- `{rel}`" + (f": {purpose}" if purpose else ""))
    modules.append("")

    asked = sorted(notes, key=lambda n: len(n["sessions"]), reverse=True)
    frequent = ["## Most-asked symbols", ""]
    for n in asked[:10]:
        frequent.append(f"- `{n['symbol']}` ({n['file']}:{n['start_line']}-{n['end_line']}), asked {len(n['sessions'])}x")
    frequent.append("")

    lessons = ["## Search blind spots", "_Questions where search ranked the right code low or missed it._", ""]
    for n in notes:
        ranks = n.get("question_ranks", {})
        misses = [q for q in n["questions"] if q in ranks and (ranks[q] is None or ranks[q] > 3)]
        for q in misses[:2]:
            lessons.append(f'- "{q}" -> `{n["symbol"]}` in {n["file"]}')
    lessons.append("")

    out: list[str] = []
    for block in (header, modules, frequent, lessons):
        room = max_lines - len(out)
        if room <= 0:
            break
        out += block[:room]
    return "\n".join(out).rstrip() + "\n"


# ── entry point ──────────────────────────────────────────────────────────────

def dream(
    repo_path: str,
    memory_dir: str = DEFAULT_MEMORY_DIR,
    sessions: list[dict] | None = None,
    max_index_lines: int = DEFAULT_MAX_INDEX_LINES,
    reset_notes: bool = False,
) -> dict:
    """
    Run one dreaming pass. `sessions` overrides the transcript log (used by
    the eval to dream on a held-out subset).
    """
    t0 = time.time()
    mdir = Path(memory_dir)
    mdir.mkdir(parents=True, exist_ok=True)

    codemap = load_codemap(repo_path)
    store = {"processed_sessions": [], "notes": []} if reset_notes else _load_notes(mdir)
    if sessions is None:
        sessions = TranscriptLog(memory_dir).read_all()

    new_sessions = consolidate(sessions, store, codemap)
    verification = verify(store, codemap)
    wiki = build_wiki(codemap, store["notes"], mdir)
    index_text = build_index(codemap, store["notes"], max_index_lines)
    (mdir / INDEX_FILE).write_text(index_text, encoding="utf-8")
    (mdir / NOTES_FILE).write_text(json.dumps(store, indent=2), encoding="utf-8")

    source_bytes = sum(f.n_bytes for f in codemap.values())
    report = {
        "repo": str(Path(repo_path).resolve()),
        "sessions_total": len(sessions),
        "sessions_new": new_sessions,
        "notes": verification["kept"],
        "notes_repointed": verification["repointed"],
        "notes_moved": verification["moved"],
        "notes_dropped": verification["dropped"],
        "wiki_pages": wiki["pages"],
        "wiki_sections": wiki["sections"],
        "source_bytes": source_bytes,
        "wiki_bytes": wiki["wiki_bytes"],
        "wiki_to_source_ratio": round(wiki["wiki_bytes"] / max(source_bytes, 1), 3),
        "index_lines": index_text.count("\n"),
        "index_tokens_approx": _approx_tokens(index_text),
        "seconds": round(time.time() - t0, 2),
    }
    (mdir / REPORT_FILE).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def load_index(memory_dir: str = DEFAULT_MEMORY_DIR) -> str:
    path = Path(memory_dir) / INDEX_FILE
    return path.read_text(encoding="utf-8") if path.exists() else ""


def format_report(report: dict) -> str:
    lines = [
        f"Dreamed over {report['sessions_total']} session(s) ({report['sessions_new']} new) in {report['seconds']}s",
        f"  notes kept      : {report['notes']}",
        f"  repointed       : {len(report['notes_repointed'])} (symbol moved within its file)",
        f"  moved           : {len(report['notes_moved'])} (symbol moved to another file)",
        f"  dropped         : {len(report['notes_dropped'])} (symbol gone)",
        f"  wiki            : {report['wiki_pages']} pages, {report['wiki_sections']} sections",
        f"  wiki vs source  : {report['wiki_to_source_ratio']}x ({report['wiki_bytes']} / {report['source_bytes']} bytes)",
        f"  MEMORY.md       : {report['index_lines']} lines, ~{report['index_tokens_approx']} tokens per answer",
    ]
    if report["wiki_to_source_ratio"] > 1.0:
        lines.append("  ⚠️  wiki is larger than the code it describes; it is not abstracting anything")
    for d in report["notes_dropped"][:5]:
        lines.append(f"  dropped {d['symbol']} ({d['was']}): {d['reason']}")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))
    p = argparse.ArgumentParser(description="Run one offline dreaming pass")
    p.add_argument("repo_path")
    p.add_argument("--memory-dir", default=DEFAULT_MEMORY_DIR)
    p.add_argument("--max-index-lines", type=int, default=DEFAULT_MAX_INDEX_LINES)
    a = p.parse_args()
    print(format_report(dream(a.repo_path, a.memory_dir, max_index_lines=a.max_index_lines)))
