# Phase 8 memory eval

- Corpus: `codebase-copilot @ a8b3916 (pre-Phase-8 snapshot)` (117 chunks), retriever: `bm25`, k=10, 20 gold questions
- Memory condition is leave-one-out: no question is ever in its own memory.

| Condition | recall@1 | recall@5 | recall@10 | MRR | top-5 context (≈tokens) |
|---|---|---|---|---|---|
| search | 0.45 | 0.70 | 0.95 | 0.589 | 1931 |
| wiki only (cold) | 0.60 | 0.80 | 0.90 | 0.704 | 1330 |
| search + wiki (cold) | 0.50 | 0.80 | 0.90 | 0.648 | 1729 |
| search + wiki + memory (LOO) | 0.60 | 0.85 | 0.90 | 0.715 | 1869 |

| Subset | n | search MRR | search + wiki + memory MRR | search r@1 | memory r@1 |
|---|---|---|---|---|---|
| answer seen before | 9 | 0.522 | 0.593 | 0.44 | 0.44 |
| answer unseen before | 11 | 0.643 | 0.814 | 0.45 | 0.73 |

- Wiki: 24 pages, 117 sections, 0.238x the size of the source (cold), 0.249x with memory
- MEMORY.md (injected into every answer): 53 lines, ~795 tokens
- Dream time: 0.09s cold, 0.09s with 15 notes

## Stale-note test

Refactored 9 files (shifted line numbers) and renamed `ingest/chunk_ast.py::_chunk_class -> _chunk_class_v2`.

- Notes citing wrong lines before dreaming: **15 / 15**
- Dream repointed 14, dropped 1
- Notes citing wrong lines after dreaming: **0 / 14**

## Per question (rank of first correct chunk, — = not in top k)

| # | question | search | search + wiki (cold) | + memory (LOO) | seen |
|---|---|---|---|---|---|
| 1 | what does the walk_repo function do? | 9 | 3 | 2 | yes |
| 2 | how does fixed-size chunking work? | 2 | 2 | 2 | no |
| 3 | what embedding model is used? | 10 | 7 | 8 | no |
| 4 | how does the vector store save chunks? | — | 7 | — | yes |
| 5 | how does cosine similarity search work in the vector store? | 2 | 2 | 1 | no |
| 6 | how is BM25 keyword search implemented? | 2 | 1 | 1 | no |
| 7 | how does reciprocal rank fusion work? | 1 | 1 | 1 | no |
| 8 | what is the low confidence threshold for refusing answers? | 1 | 1 | 1 | yes |
| 9 | how does the CLI index command work? | 3 | 1 | 2 | yes |
| 10 | how does the CLI ask command work? | 1 | 1 | 1 | no |
| 11 | how does the system prompt enforce citations? | 1 | 2 | 3 | yes |
| 12 | how does gitignore filtering work in walk_repo? | 1 | 1 | 1 | no |
| 13 | what file extensions are indexed by default? | 1 | 1 | 1 | yes |
| 14 | how does AST chunking extract function boundaries? | 1 | 1 | 1 | no |
| 15 | how are class methods chunked separately? | 7 | 3 | 3 | no |
| 16 | how does the embedder embed a query? | 3 | 2 | 1 | no |
| 17 | how does chunk batching work during indexing? | 9 | — | 1 | yes |
| 18 | how does BM25 query sanitization prevent syntax errors? | 1 | 1 | 1 | no |
| 19 | what metadata is stored alongside each chunk in chromadb? | 7 | — | — | yes |
| 20 | how is the Groq API called to generate answers? | 1 | 1 | 1 | yes |
