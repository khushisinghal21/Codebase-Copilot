"""Tests for Phase 8 memory: consolidation, verification, wiki, index cap."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from memory.dream import dream
from memory.transcripts import TranscriptLog, confirmed_from_citations
from memory.wiki_store import WikiStore, WIKI_DB_FILE

MATH = '''"""Arithmetic helpers for the billing service."""


def add_tax(amount, rate):
    """Return amount with sales tax applied."""
    return amount * (1 + rate)


def round_cents(x):
    """Round a price to two decimals."""
    return round(x, 2)
'''


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "billing.py").write_text(MATH)
    return repo


def _session(repo: Path, question: str, symbol: str, start: int, end: int) -> dict:
    ref = {"file_path": str(repo / "billing.py"), "symbol_name": symbol, "start_line": start, "end_line": end}
    return {"id": f"s-{symbol}-{start}", "ts": 1.0, "question": question, "retrieved": [ref], "confirmed": [ref]}


def test_consolidate_and_wiki(tmp_path):
    repo = _repo(tmp_path)
    mem = tmp_path / "mem"
    report = dream(str(repo), str(mem), sessions=[_session(repo, "how is VAT added?", "add_tax", 4, 6)])
    assert report["notes"] == 1
    page = (mem / "wiki" / "billing.py.md").read_text()
    assert "Arithmetic helpers for the billing service." in page
    assert '"how is VAT added?"' in page
    hits = WikiStore(str(mem / WIKI_DB_FILE)).search("VAT")
    assert hits and hits[0]["symbol_name"] == "add_tax"   # learned phrase, not in the code


def test_verify_repoints_and_drops(tmp_path):
    repo = _repo(tmp_path)
    mem = tmp_path / "mem"
    sessions = [_session(repo, "tax?", "add_tax", 4, 6), _session(repo, "rounding?", "round_cents", 9, 11)]
    dream(str(repo), str(mem), sessions=sessions)

    # shift every line by 3 and rename one function
    src = MATH.replace('"""Arithmetic helpers for the billing service."""\n',
                       '"""Arithmetic helpers for the billing service."""\nimport math\nimport os\nimport sys\n')
    (repo / "billing.py").write_text(src.replace("def round_cents(", "def round_to_cents("))

    report = dream(str(repo), str(mem), sessions=sessions)
    notes = json.loads((mem / "notes.json").read_text())["notes"]
    assert [n["symbol"] for n in notes] == ["add_tax"]
    assert (notes[0]["start_line"], notes[0]["end_line"]) == (7, 9)
    assert len(report["notes_repointed"]) == 1
    assert report["notes_dropped"][0]["symbol"] == "round_cents"


def test_index_is_capped(tmp_path):
    repo = _repo(tmp_path)
    mem = tmp_path / "mem"
    report = dream(str(repo), str(mem), sessions=[], max_index_lines=4)
    assert (mem / "MEMORY.md").read_text().count("\n") <= 4
    assert report["index_lines"] <= 4


def test_transcript_log_and_citation_confirmation(tmp_path):
    chunks = [{"file_path": "/x/a.py", "symbol_name": "f", "start_line": 1, "end_line": 9},
              {"file_path": "/x/b.py", "symbol_name": "g", "start_line": 1, "end_line": 9}]
    confirmed = confirmed_from_citations(chunks, [("a.py", 2, 3)])
    assert [c["symbol_name"] for c in confirmed] == ["f"]

    log = TranscriptLog(str(tmp_path / "mem"))
    sid = log.log_session("q", chunks, confirmed)
    assert log.add_confirmation(sid, chunks[1])
    [s] = log.read_all()
    assert [c["symbol_name"] for c in s["confirmed"]] == ["f", "g"]
