"""
LLM: Ollama running Qwen2.5:7B locally.

Ollama + Qwen2.5 is currently one of the strongest fully-local, fully-free
LLM setups available -- runs on consumer hardware (CPU or modest GPU), no
API key, no per-token cost, no data leaving the machine. This is the
direct local analogue of Gemini in the production stack.

Setup (one-time, on your machine -- not something this code can do for
you): install Ollama from https://ollama.com, then run
`ollama pull qwen2.5:7b`. Once the Ollama app is running in the
background, this module talks to it over localhost automatically.

Production swap: replace `OllamaLLM.generate()`'s body with a call to
`genai.GenerativeModel("gemini-1.5-flash").generate_content(...)` --
the `generate(system_prompt, query, context_chunks)` signature stays
identical, so nothing calling this class needs to change.
"""
from __future__ import annotations

import textwrap

from app.config import OLLAMA_BASE_URL, OLLAMA_MODEL


def _build_context_block(chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        parts.append(f"### Source: {c['source']} (chunk {c['chunk_id']}, score={c['score']:.2f})\n{c['text']}")
    return "\n\n".join(parts)


class OllamaLLM:
    def __init__(self, model: str = OLLAMA_MODEL, base_url: str = OLLAMA_BASE_URL):
        self.model = model
        self.base_url = base_url

    def is_available(self) -> bool:
        try:
            import ollama

            client = ollama.Client(host=self.base_url)
            client.list()
            return True
        except Exception:
            return False

    def generate(self, system_prompt: str, query: str, context_chunks: list[dict]) -> str:
        import ollama

        client = ollama.Client(host=self.base_url)
        context = _build_context_block(context_chunks)
        response = client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ],
        )
        return response["message"]["content"]

    def generate_sql(self, schema: str, question: str) -> str:
        import ollama

        client = ollama.Client(host=self.base_url)
        prompt = (
            f"Given this schema:\n{schema}\n\n"
            f"Write a single DuckDB SQL query (no explanation, no markdown "
            f"fences, no trailing semicolon commentary) that answers: {question}"
        )
        response = client.chat(model=self.model, messages=[{"role": "user", "content": prompt}])
        sql = response["message"]["content"].strip()
        return sql.strip("`").replace("sql\n", "").strip()


class ExtractiveFallbackLLM:
    """Zero-dependency fallback used only when Ollama isn't running.
    Produces a grounded, cited answer without any generation model, so
    the pipeline still demonstrably works before you've installed Ollama."""

    def is_available(self) -> bool:
        return True

    def generate(self, system_prompt: str, query: str, context_chunks: list[dict]) -> str:
        if not context_chunks:
            return "No relevant context found in the knowledge base."
        lines = [f'Based on the retrieved context for: "{query}"\n']
        for c in context_chunks:
            snippet = textwrap.shorten(c["text"], width=240, placeholder="...")
            lines.append(f"- {snippet} [{c['source']}]")
        lines.append(
            "\n(Extractive fallback mode -- install Ollama and run "
            "`ollama pull qwen2.5:7b` to enable real grounded generation.)"
        )
        return "\n".join(lines)

    def generate_sql(self, schema: str, question: str) -> str:
        raise RuntimeError("SQL generation requires Ollama; use the rule-based SQL agent fallback instead.")
