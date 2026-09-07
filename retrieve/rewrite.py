"""
Phase 3 query rewriting: expand the user's query before searching.

Problem: Users ask "how does auth work?" but code says "authenticate",
"verify_token", "session_manager". BM25 misses these because the tokens
don't match. The LLM knows the synonyms and can expand the query.

Strategy: generate 3-5 alternative phrasings + key terms, then search
with both the original and expanded query and union the results before RRF.
"""

import os
from groq import Groq

_REWRITE_PROMPT = """\
You are a code search query expander. Given a natural-language question about \
a codebase, output a single line of search terms that will help retrieve the \
relevant source code.

Rules:
- Include the key nouns, verbs, and identifiers from the question.
- Add likely code-level synonyms (e.g. "auth" → "authenticate login session token").
- Add likely function/class name patterns (e.g. snake_case variants).
- Output ONLY the expanded search terms, no explanation, no bullet points.
- Keep it under 30 words.

Question: {query}
Expanded search terms:"""


def rewrite_query(query: str, api_key: str | None = None) -> str:
    """
    Use the LLM to expand a natural language query into richer search terms.

    Falls back to the original query if the LLM call fails, so this is
    always safe to call.
    """
    try:
        client = Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",   # use the lighter model — this is a quick utility call
            max_tokens=60,
            messages=[
                {
                    "role": "user",
                    "content": _REWRITE_PROMPT.format(query=query),
                }
            ],
        )
        expanded = response.choices[0].message.content.strip()
        return expanded if expanded else query
    except Exception:
        # Never let a rewrite failure break the main flow
        return query


if __name__ == "__main__":
    test_queries = [
        "how does auth work?",
        "where is the database connection?",
        "what does walk_repo do?",
    ]
    for q in test_queries:
        expanded = rewrite_query(q)
        print(f"Original : {q}")
        print(f"Expanded : {expanded}")
        print()
