"""
Phase 8 eval: does a dreamed wiki + memory improve retrieval?

Conditions (same corpus, same gold set, same k):
  search                 the existing retriever, no memory
  wiki only (cold)       dreamed wiki from code alone, no search
  search + wiki (cold)   wiki built from code alone, zero past sessions
  search + wiki + memory wiki that has also consolidated past sessions

No test question is ever seen by its own memory. The memory condition uses
leave-one-out: for question i, we dream over the sessions of the other 19
questions (each logged with the chunk that answered it), then ask question i.
Results are split into questions whose answer symbol was asked about before
("seen") and ones that weren't ("unseen").

The --stale-test refactors a copy of the corpus (shifts line numbers in every
file that has notes and renames one noted function) and checks how many notes
would cite the wrong lines before and after a dream.

Usage:
  python eval/run_memory_eval.py                      # hybrid: dense + BM25
  python eval/run_memory_eval.py --retriever bm25     # no embedding model needed
  python eval/run_memory_eval.py --corpus /path/to/snapshot --stale-test
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.run_eval import _chunk_matches
from index.bm25_store import BM25Store
from ingest.chunk_ast import chunk_repo
from ingest.walk_repo import walk_repo
from memory.codemap import load_codemap
from memory.dream import dream, load_index
from memory.wiki_store import WIKI_DB_FILE, WikiStore, wiki_hits_to_chunks
from retrieve.search import search

ROOT = Path(__file__).parent.parent
GOLD_SET_PATH = Path(__file__).parent / "gold_set.json"


def _tokens(text: str) -> int:
    return len(text) // 4


def _rank(chunks: list[dict], entry: dict) -> int | None:
    for i, c in enumerate(chunks, 1):
        if _chunk_matches(c, entry["expected_file"], entry["expected_symbol"]):
            return i
    return None


def _metrics(ranks: list[int | None], contexts: list[int]) -> dict:
    n = len(ranks)
    return {
        "n": n,
        "recall@1": sum(1 for r in ranks if r == 1) / n,
        "recall@5": sum(1 for r in ranks if r and r <= 5) / n,
        "recall@10": sum(1 for r in ranks if r and r <= 10) / n,
        "mrr": sum(1 / r for r in ranks if r) / n,
        "top5_context_tokens": round(sum(contexts) / max(len(contexts), 1)),
    }


class Index:
    """Temporary BM25 (+ optional dense) index over a corpus."""

    def __init__(self, corpus: str, retriever: str, workdir: Path):
        self.chunks = list(chunk_repo(walk_repo(corpus)))
        self.bm25 = BM25Store(str(workdir / "bm25.db"))
        self.bm25.add_chunks(self.chunks)
        self.embedder = self.store = None
        if retriever == "hybrid":
            from ingest.embed import Embedder
            from index.store import VectorStore
            self.embedder = Embedder()
            self.store = VectorStore(persist_dir=str(workdir / "chroma"))
            self.store.add_chunks(self.chunks, self.embedder.embed_texts([c["text"] for c in self.chunks]))

    def query(self, question: str, k: int, wiki: WikiStore | None = None) -> list[dict]:
        return search(question, self.embedder, self.store, bm25_store=self.bm25, k=k, wiki_store=wiki)

    def gold_chunk(self, entry: dict) -> dict | None:
        exact = [c for c in self.chunks
                 if c["file_path"].endswith(entry["expected_file"]) and c.get("symbol_name") == entry["expected_symbol"]]
        if exact:
            return exact[0]
        loose = [c for c in self.chunks if _chunk_matches(c, entry["expected_file"], entry["expected_symbol"])]
        return loose[0] if loose else None


def _session(i: int, entry: dict, retrieved: list[dict], gold: dict | None) -> dict:
    return {"id": f"gold-{i:02d}", "ts": float(i), "source": "eval", "question": entry["question"],
            "retrieved": [{"file_path": c["file_path"], "symbol_name": c.get("symbol_name"),
                           "start_line": c["start_line"], "end_line": c["end_line"]} for c in retrieved],
            "confirmed": [{"file_path": gold["file_path"], "symbol_name": gold.get("symbol_name"),
                           "start_line": gold["start_line"], "end_line": gold["end_line"]}] if gold else []}


def run(corpus: str, retriever: str, k: int, workdir: Path) -> dict:
    gold = json.loads(GOLD_SET_PATH.read_text())
    idx = Index(corpus, retriever, workdir)
    print(f"Corpus: {corpus}  ({len(idx.chunks)} chunks)  retriever={retriever}  k={k}\n")

    rows: dict[str, list] = {"search": [], "wiki only (cold)": [], "search + wiki (cold)": [],
                             "search + wiki + memory (LOO)": []}
    ctx: dict[str, list] = {name: [] for name in rows}

    # 1. baseline + the sessions it would have produced
    baseline_results, sessions = [], []
    for i, e in enumerate(gold):
        res = idx.query(e["question"], k)
        baseline_results.append(res)
        sessions.append(_session(i, e, res, idx.gold_chunk(e)))

    # 2. cold wiki: dreamed from code alone, no sessions
    cold_dir = workdir / "mem_cold"
    cold_report = dream(corpus, memory_dir=str(cold_dir), sessions=[], reset_notes=True)
    cold_wiki = WikiStore(str(cold_dir / WIKI_DB_FILE))

    for i, e in enumerate(gold):
        for name, res in (
            ("search", baseline_results[i]),
            ("wiki only (cold)", wiki_hits_to_chunks(cold_wiki.search(e["question"], k=k), idx.bm25)[:k]),
            ("search + wiki (cold)", idx.query(e["question"], k, wiki=cold_wiki)),
        ):
            rows[name].append(_rank(res, e))
            ctx[name].append(sum(_tokens(c["text"]) for c in res[:5]))

    # 3. memory, leave-one-out
    seen_flags, loo_reports = [], []
    for i, e in enumerate(gold):
        others = [s for j, s in enumerate(sessions) if j != i]
        fold_dir = workdir / f"mem_loo_{i:02d}"
        rep = dream(corpus, memory_dir=str(fold_dir), sessions=others, reset_notes=True)
        loo_reports.append(rep)
        wiki = WikiStore(str(fold_dir / WIKI_DB_FILE))
        res = idx.query(e["question"], k, wiki=wiki)
        wiki.close()
        rows["search + wiki + memory (LOO)"].append(_rank(res, e))
        ctx["search + wiki + memory (LOO)"].append(sum(_tokens(c["text"]) for c in res[:5]))
        gold_key = (e["expected_file"], e["expected_symbol"])
        seen_flags.append(any((gold[j]["expected_file"], gold[j]["expected_symbol"]) == gold_key
                              for j in range(len(gold)) if j != i))

    summary = {name: _metrics(rows[name], ctx[name]) for name in rows}
    split = {}
    for label, flag in (("seen", True), ("unseen", False)):
        ids = [i for i, f in enumerate(seen_flags) if f == flag]
        if ids:
            split[label] = {
                "search": _metrics([rows["search"][i] for i in ids], [ctx["search"][i] for i in ids]),
                "search + wiki + memory (LOO)": _metrics(
                    [rows["search + wiki + memory (LOO)"][i] for i in ids],
                    [ctx["search + wiki + memory (LOO)"][i] for i in ids]),
            }

    per_question = [{"question": e["question"], **{name: rows[name][i] for name in rows},
                     "answer_seen_before": seen_flags[i]} for i, e in enumerate(gold)]
    full_mem_dir = workdir / "mem_full"
    full_report = dream(corpus, memory_dir=str(full_mem_dir), sessions=sessions, reset_notes=True)

    return {
        "corpus": corpus,
        "retriever": retriever,
        "k": k,
        "chunks": len(idx.chunks),
        "questions": len(gold),
        "summary": summary,
        "seen_unseen": split,
        "per_question": per_question,
        "cold_dream": {x: cold_report[x] for x in ("wiki_pages", "wiki_sections", "wiki_to_source_ratio",
                                                   "index_lines", "index_tokens_approx", "seconds")},
        "full_dream": {x: full_report[x] for x in ("notes", "wiki_to_source_ratio", "index_lines",
                                                   "index_tokens_approx", "seconds")},
        "memory_index_preview": load_index(str(full_mem_dir))[:1500],
        "_sessions": sessions,
    }


def stale_test(corpus: str, sessions: list[dict], workdir: Path) -> dict:
    """Refactor a copy of the corpus and check notes before/after a dream."""
    copy = workdir / "refactored"
    shutil.copytree(corpus, copy, ignore=shutil.ignore_patterns(".git", "__pycache__", ".copilot_*"))
    mem = workdir / "mem_stale"
    # Sessions point at the original absolute paths; re-root them onto the copy.
    src_root = str(Path(corpus).resolve())
    moved_sessions = json.loads(json.dumps(sessions).replace(src_root, str(copy.resolve())))
    dream(str(copy), memory_dir=str(mem), sessions=moved_sessions, reset_notes=True)
    notes = json.loads((mem / "notes.json").read_text())["notes"]

    # Refactor: add a 6-line helper above the first symbol in every noted file,
    # and rename one noted function (picked deterministically: first by name).
    codemap = load_codemap(str(copy))
    noted_files = sorted({n["file"] for n in notes})
    for rel in noted_files:
        path = copy / rel
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        first = min(s.start_line for s in codemap[rel].symbols.values())
        helper = ["def _refactor_helper():\n", '    """Added by the stale-note test."""\n',
                  "    return None\n", "\n", "\n", ""]
        path.write_text("".join(lines[: first - 1] + helper + lines[first - 1:]), encoding="utf-8")
    renamable = sorted((n for n in notes if codemap[n["file"]].symbols[n["symbol"]].symbol_type == "function"),
                       key=lambda n: n["symbol"])
    victim = renamable[0]
    vpath = copy / victim["file"]
    vpath.write_text(vpath.read_text(encoding="utf-8").replace(
        f"def {victim['symbol']}(", f"def {victim['symbol']}_v2(", 1), encoding="utf-8")

    after = load_codemap(str(copy))

    def wrong(ns: list[dict]) -> int:
        bad = 0
        for n in ns:
            sym = after.get(n["file"]) and after[n["file"]].symbols.get(n["symbol"])
            if not sym or (sym.start_line, sym.end_line) != (n["start_line"], n["end_line"]):
                bad += 1
        return bad

    wrong_before = wrong(notes)
    report = dream(str(copy), memory_dir=str(mem), sessions=moved_sessions)
    notes_after = json.loads((mem / "notes.json").read_text())["notes"]
    return {
        "notes": len(notes),
        "files_refactored": len(noted_files),
        "renamed": f"{victim['file']}::{victim['symbol']} -> {victim['symbol']}_v2",
        "wrong_line_refs_before_dream": wrong_before,
        "repointed": len(report["notes_repointed"]),
        "dropped": len(report["notes_dropped"]),
        "wrong_line_refs_after_dream": wrong(notes_after),
        "example_repoint": report["notes_repointed"][0] if report["notes_repointed"] else None,
        "example_drop": report["notes_dropped"][0] if report["notes_dropped"] else None,
    }


def _md(results: dict) -> str:
    s = results["summary"]
    out = [f"# Phase 8 memory eval", "",
           f"- Corpus: `{results['corpus']}` ({results['chunks']} chunks), "
           f"retriever: `{results['retriever']}`, k={results['k']}, {results['questions']} gold questions",
           "- Memory condition is leave-one-out: no question is ever in its own memory.", "",
           "| Condition | recall@1 | recall@5 | recall@10 | MRR | top-5 context (≈tokens) |",
           "|---|---|---|---|---|---|"]
    for name, m in s.items():
        out.append(f"| {name} | {m['recall@1']:.2f} | {m['recall@5']:.2f} | {m['recall@10']:.2f} | "
                   f"{m['mrr']:.3f} | {m['top5_context_tokens']} |")
    if results["seen_unseen"]:
        out += ["", "| Subset | n | search MRR | search + wiki + memory MRR | search r@1 | memory r@1 |",
                "|---|---|---|---|---|---|"]
        for label, d in results["seen_unseen"].items():
            a, b = d["search"], d["search + wiki + memory (LOO)"]
            out.append(f"| answer {label} before | {a['n']} | {a['mrr']:.3f} | {b['mrr']:.3f} | "
                       f"{a['recall@1']:.2f} | {b['recall@1']:.2f} |")
    cd, fd = results["cold_dream"], results["full_dream"]
    out += ["", f"- Wiki: {cd['wiki_pages']} pages, {cd['wiki_sections']} sections, "
                f"{cd['wiki_to_source_ratio']}x the size of the source (cold), {fd['wiki_to_source_ratio']}x with memory",
            f"- MEMORY.md (injected into every answer): {fd['index_lines']} lines, ~{fd['index_tokens_approx']} tokens",
            f"- Dream time: {cd['seconds']}s cold, {fd['seconds']}s with {fd['notes']} notes"]
    if "stale_test" in results:
        st = results["stale_test"]
        out += ["", "## Stale-note test", "",
                f"Refactored {st['files_refactored']} files (shifted line numbers) and renamed "
                f"`{st['renamed']}`.", "",
                f"- Notes citing wrong lines before dreaming: **{st['wrong_line_refs_before_dream']} / {st['notes']}**",
                f"- Dream repointed {st['repointed']}, dropped {st['dropped']}",
                f"- Notes citing wrong lines after dreaming: **{st['wrong_line_refs_after_dream']} / "
                f"{st['notes'] - st['dropped']}**"]
    out += ["", "## Per question (rank of first correct chunk, — = not in top k)", "",
            "| # | question | search | search + wiki (cold) | + memory (LOO) | seen |", "|---|---|---|---|---|---|"]
    for i, q in enumerate(results["per_question"], 1):
        f = lambda r: str(r) if r else "—"
        out.append(f"| {i} | {q['question']} | {f(q['search'])} | {f(q['search + wiki (cold)'])} | "
                   f"{f(q['search + wiki + memory (LOO)'])} | {'yes' if q['answer_seen_before'] else 'no'} |")
    return "\n".join(out) + "\n"


def main():
    p = argparse.ArgumentParser(description="Evaluate dreamed memory against plain search")
    p.add_argument("--corpus", default=str(ROOT), help="Repo to index (default: this repo)")
    p.add_argument("--retriever", choices=["hybrid", "bm25"], default="hybrid")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--stale-test", action="store_true")
    p.add_argument("--label", default=None, help="Corpus name shown in the results")
    p.add_argument("--out", default=str(Path(__file__).parent / "memory_results"),
                   help="Output path prefix (.json and .md are written)")
    a = p.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        results = run(a.corpus, a.retriever, a.k, work)
        if a.stale_test:
            results["stale_test"] = stale_test(a.corpus, results["_sessions"], work)
    results.pop("_sessions")
    results["corpus"] = a.label or Path(a.corpus).resolve().name

    Path(a.out + ".json").write_text(json.dumps(results, indent=2))
    md = _md(results)
    Path(a.out + ".md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
