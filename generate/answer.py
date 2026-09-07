"""
Generation: turn retrieved chunks into a grounded answer with
mandatory path:line citations, refusing when retrieval confidence
is low - per the guide's "Generation" bullet under Reference architecture.
"""

import os

from groq import Groq

MODEL = "openai/gpt-oss-120b"

SYSTEM_PROMPT = """You are a codebase Q&A assistant. You will be given a question \
and a set of code excerpts, each labeled with its file path and line range.

Rules:
1. Answer ONLY using the provided excerpts. Do not use outside knowledge of \
common libraries or frameworks to fill gaps.
2. Every factual claim about the code MUST be followed by a citation in the \
exact form (path:start_line-end_line), pointing to the excerpt that supports it.
3. If the excerpts don't contain enough information to answer confidently, say \
so explicitly rather than guessing - do not fabricate file paths or line numbers.
4. Be concise. Prefer a direct answer over a long essay.
"""

# Below these scores, treat retrieval as too weak to answer confidently.
# Each search mode produces scores in a different range:
#   Dense cosine similarity : 0.0 – 1.0  (threshold 0.3)
#   RRF fused               : ~0.01–0.03  (threshold 0.01)
#   Cross-encoder logits    : -10 to +10  (threshold -5.0)
LOW_CONFIDENCE_THRESHOLD = 0.3
RRF_LOW_CONFIDENCE_THRESHOLD = 0.01
RERANK_LOW_CONFIDENCE_THRESHOLD = -5.0


def _format_chunks(chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        header = f"--- {c['file_path']}:{c['start_line']}-{c['end_line']} ---"
        parts.append(f"{header}\n{c['text']}")
    return "\n\n".join(parts)


def generate_answer(query: str, chunks: list[dict], api_key: str | None = None) -> str:
    if not chunks:
        return "I couldn't find any relevant code for this question in the index."

    top_score = chunks[0].get("score", 1.0)
    is_rrf = chunks[0].get("is_rrf", False)
    is_reranked = chunks[0].get("is_reranked", False)

    if is_reranked:
        threshold = RERANK_LOW_CONFIDENCE_THRESHOLD
    elif is_rrf:
        threshold = RRF_LOW_CONFIDENCE_THRESHOLD
    else:
        threshold = LOW_CONFIDENCE_THRESHOLD

    if top_score < threshold:
        return (
            "Retrieval confidence is too low to answer reliably "
            f"(top match score: {top_score:.4f}). Try rephrasing the question, "
            "or the relevant code may not be indexed."
        )

    client = Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))

    user_message = f"""Question: {query}

Code excerpts:

{_format_chunks(chunks)}"""

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )

    return response.choices[0].message.content
