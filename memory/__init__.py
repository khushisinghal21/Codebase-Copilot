"""
Phase 8: agent memory with offline "dreaming".

Layers (bottom to top):
  transcripts.jsonl  raw log of every ask session (ground truth)
  notes.json         verified memory notes: which symbol answered which question
  wiki/*.md + wiki.db  one linked page per file, searchable by section
  MEMORY.md          small, capped index injected into every answer

`dream.py` is the offline pass that turns the bottom layers into the top ones,
re-checks every note against the current code, and repairs or drops stale ones.

Inspired by Anthropic's engineering post "The components of agent memory"
(https://www.anthropic.com/engineering/components-of-agent-memory).
"""

DEFAULT_MEMORY_DIR = ".copilot_memory"
