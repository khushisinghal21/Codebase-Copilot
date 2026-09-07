"""
FastAPI backend for Codebase Copilot.

Endpoints:
  POST /index          - index a repo (from a GitHub URL or local path)
  POST /ask            - ask a question
  POST /blast          - blast radius for a symbol
  GET  /onboard        - onboarding guide
  GET  /repos          - list indexed repos
  GET  /health         - health check
"""

import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# ── bootstrap path so imports work from api/ ─────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from index.store import VectorStore, DEFAULT_PERSIST_DIR
from index.bm25_store import BM25Store, DEFAULT_DB_PATH
from index.graph_store import GraphStore, DEFAULT_GRAPH_DB
from index.repo_registry import RepoRegistry, derive_repo_id
from ingest.embed import Embedder
from ingest.walk_repo import walk_repo
from ingest.chunk_ast import chunk_repo
from ingest.call_graph import build_call_graph
from ingest.import_parser import find_cross_repo_imports
from retrieve.search import search
from retrieve.rewrite import rewrite_query
from retrieve.expand import expand_with_graph
from generate.answer import generate_answer
from generate.validate import validate_citations, format_validation_report
from analyze.blast_radius import compute_blast_radius, format_blast_radius, llm_impact_summary
from analyze.onboarding import build_reading_plan, generate_onboarding_guide, print_reading_plan

# ── app setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Codebase Copilot",
    description="AI-powered codebase intelligence — ask questions, trace call graphs, analyse change impact.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ── shared singletons (loaded once) ───────────────────────────────────────────
_embedder: Embedder | None = None
_store: VectorStore | None = None
_bm25: BM25Store | None = None
_graph: GraphStore | None = None

def get_stores():
    global _embedder, _store, _bm25, _graph
    if _embedder is None:
        _embedder = Embedder()
        _store    = VectorStore(persist_dir=DEFAULT_PERSIST_DIR)
        _bm25     = BM25Store(db_path=DEFAULT_DB_PATH)
        _graph    = GraphStore(db_path=DEFAULT_GRAPH_DB)
    return _embedder, _store, _bm25, _graph


# ── request / response models ──────────────────────────────────────────────────
class IndexRequest(BaseModel):
    github_url: str                    # e.g. https://github.com/org/repo
    repo_name: Optional[str] = None    # defaults to repo folder name
    reset: bool = False

class AskRequest(BaseModel):
    question: str
    repo: Optional[str] = None         # filter to specific repo
    k: int = 10
    no_rerank: bool = False
    no_graph: bool = False
    no_rewrite: bool = False

class BlastRequest(BaseModel):
    symbol: str
    no_llm: bool = False

class IndexStatus(BaseModel):
    status: str
    message: str

# ── background index jobs ──────────────────────────────────────────────────────
_index_jobs: dict[str, dict] = {}   # repo_name -> {status, message}

def _clone_and_index(github_url: str, repo_name: str, reset: bool):
    """Clone a GitHub repo and index it. Runs in a background thread."""
    _index_jobs[repo_name] = {"status": "cloning", "message": f"Cloning {github_url}..."}

    # Clone to a temp dir
    tmpdir = tempfile.mkdtemp(prefix="copilot_")
    try:
        result = subprocess.run(
            ["git", "clone", "--depth", "1", github_url, tmpdir],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            _index_jobs[repo_name] = {"status": "error", "message": result.stderr}
            return

        _index_jobs[repo_name] = {"status": "indexing", "message": "Chunking and embedding..."}

        embedder, store, bm25, graph = get_stores()

        if reset:
            store.reset(); bm25.reset(); graph.reset()

        all_chunks = []
        chunks_batch = []
        BATCH_SIZE = 50

        for chunk in chunk_repo(walk_repo(tmpdir)):
            # Rewrite file_path to show repo-relative path
            rel = str(Path(chunk["file_path"]).relative_to(tmpdir))
            chunk["file_path"] = f"{repo_name}/{rel}"
            chunks_batch.append(chunk)
            all_chunks.append(chunk)
            if len(chunks_batch) >= BATCH_SIZE:
                embeddings = embedder.embed_texts([c["text"] for c in chunks_batch])
                store.add_chunks(chunks_batch, embeddings, repo_id=repo_name)
                bm25.add_chunks(chunks_batch, repo_id=repo_name)
                chunks_batch = []

        if chunks_batch:
            embeddings = embedder.embed_texts([c["text"] for c in chunks_batch])
            store.add_chunks(chunks_batch, embeddings, repo_id=repo_name)
            bm25.add_chunks(chunks_batch, repo_id=repo_name)

        # Call graph
        graph.add_symbols(all_chunks)
        edges = build_call_graph(walk_repo(tmpdir))
        # Rewrite file paths in edges
        edges = [(f"{repo_name}/{Path(c).relative_to(tmpdir)}", s, e) for c, s, e in edges]
        graph.add_calls(edges)

        # Registry
        registry = RepoRegistry(graph.conn)
        registry.register(repo_name, tmpdir, chunk_count=len(all_chunks))

        # Reset singletons so next query uses fresh data
        global _store, _bm25, _graph
        _store = VectorStore(persist_dir=DEFAULT_PERSIST_DIR)
        _bm25  = BM25Store(db_path=DEFAULT_DB_PATH)
        _graph = GraphStore(db_path=DEFAULT_GRAPH_DB)

        _index_jobs[repo_name] = {
            "status": "done",
            "message": f"Indexed {len(all_chunks)} chunks from {github_url}",
        }

    except Exception as e:
        _index_jobs[repo_name] = {"status": "error", "message": str(e)}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    index_html = FRONTEND_DIR / "index.html"
    if index_html.exists():
        return FileResponse(str(index_html))
    return {"message": "Codebase Copilot API", "docs": "/docs"}


@app.get("/health")
async def health():
    return {"status": "ok", "groq_key_set": bool(os.environ.get("GROQ_API_KEY"))}


@app.post("/index")
async def index_repo(req: IndexRequest, background_tasks: BackgroundTasks):
    repo_name = req.repo_name or req.github_url.rstrip("/").split("/")[-1]
    _index_jobs[repo_name] = {"status": "queued", "message": "Job queued"}
    background_tasks.add_task(_clone_and_index, req.github_url, repo_name, req.reset)
    return {"repo_name": repo_name, "status": "queued", "check": f"/index/status/{repo_name}"}


@app.get("/index/status/{repo_name}")
async def index_status(repo_name: str):
    job = _index_jobs.get(repo_name, {"status": "not_found", "message": "No job found"})
    return {"repo_name": repo_name, **job}


@app.get("/repos")
async def list_repos():
    _, _, _, graph = get_stores()
    registry = RepoRegistry(graph.conn)
    return {"repos": registry.list_all()}


@app.post("/ask")
async def ask(req: AskRequest):
    if not os.environ.get("GROQ_API_KEY"):
        raise HTTPException(status_code=400, detail="GROQ_API_KEY not set")

    embedder, store, bm25, graph = get_stores()
    if store.count() == 0:
        raise HTTPException(status_code=400, detail="No repos indexed yet. POST /index first.")

    # Query rewriting
    query = req.question
    rewritten = query
    if not req.no_rewrite:
        try:
            rewritten = rewrite_query(query)
        except Exception:
            pass

    chunks = search(
        rewritten, embedder, store, bm25_store=bm25,
        k=req.k, repo_id=req.repo,
    )

    if not req.no_graph:
        try:
            chunks = expand_with_graph(query, chunks, graph, embedder, store)
        except Exception:
            pass

    answer = generate_answer(query, chunks)

    validation = None
    try:
        report = validate_citations(answer, chunks)
        validation = {
            "hallucinated": report["hallucinated"],
            "verified_count": report["verified"],
        }
    except Exception:
        pass

    return {
        "question": req.question,
        "rewritten_query": rewritten if rewritten != query else None,
        "answer": answer,
        "chunks": [
            {
                "file_path": c["file_path"],
                "start_line": c["start_line"],
                "end_line": c["end_line"],
                "score": round(c["score"], 4),
                "repo_id": c.get("repo_id"),
                "symbol_name": c.get("symbol_name"),
            }
            for c in chunks
        ],
        "citation_validation": validation,
    }


@app.post("/blast")
async def blast(req: BlastRequest):
    _, store, _, graph = get_stores()

    result = compute_blast_radius(req.symbol, graph)

    llm_summary = None
    if not req.no_llm:
        # Fetch source chunk
        embedder, store, _, _ = get_stores()
        vec = embedder.embed_query(f"def {req.symbol}")
        hits = store.query(vec, k=5)
        chunk_text = next(
            (h["text"] for h in hits if h.get("symbol_name") == req.symbol), ""
        )
        try:
            llm_summary = llm_impact_summary(req.symbol, result, chunk_text)
        except Exception as e:
            llm_summary = f"(LLM unavailable: {e})"

    return {
        "symbol":          result["symbol"],
        "affected_count":  len(result["all_affected"]),
        "affected_files":  result["affected_files"],
        "affected_symbols": result["all_affected"],
        "impact_pct":      round(result["impact_pct"], 2),
        "depth":           result["depth"],
        "by_depth":        result["by_depth"],
        "llm_summary":     llm_summary,
    }


@app.get("/onboard")
async def onboard(no_llm: bool = False):
    embedder, store, _, graph = get_stores()

    if store.count() == 0:
        raise HTTPException(status_code=400, detail="No repos indexed yet. POST /index first.")

    plan = build_reading_plan(graph, embedder, store)

    llm_guide = None
    if not no_llm:
        registry = RepoRegistry(graph.conn)
        repos = registry.list_all()
        repo_name = repos[0]["repo_id"] if repos else "this codebase"
        try:
            llm_guide = generate_onboarding_guide(plan, repo_name=repo_name)
        except Exception as e:
            llm_guide = f"(LLM unavailable: {e})"

    return {
        "layers": [
            {
                "name": layer["meta"]["name"],
                "description": layer["meta"]["description"],
                "symbols": [
                    {
                        "symbol_name": s["symbol_name"],
                        "symbol_type": s["symbol_type"],
                        "file_path":   s["file_path"],
                        "in_degree":   s["in_degree"],
                        "out_degree":  s["out_degree"],
                    }
                    for s in layer["symbols"]
                ],
            }
            for layer in plan
        ],
        "llm_guide": llm_guide,
    }
