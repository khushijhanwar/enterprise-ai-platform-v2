"""
Planner: the routing brain of the agent workflow. Decides whether a query
needs structured data (SQL agent) or unstructured knowledge (document
agent). Uses Ollama for real LLM-based routing when available, falls
back to keyword heuristics otherwise.

This is the "Planner" node concept from Google ADK / LangGraph multi-agent
patterns -- a dedicated routing step before any tool is invoked, rather
than each tool deciding for itself whether it applies.
"""
from __future__ import annotations

from app.llm.ollama import OllamaLLM

SQL_KEYWORDS = ["top ", "how many", "spend", "revenue", "count", "usage", "api call", "rank", "average", "total"]


class Planner:
    def __init__(self):
        self.llm = OllamaLLM()

    def route(self, query: str) -> str:
        """Returns 'sql_query' or 'document_retrieval'."""
        if self.llm.is_available():
            try:
                return self._route_with_llm(query)
            except Exception:
                pass
        return self._route_with_keywords(query)

    def _route_with_keywords(self, query: str) -> str:
        q = query.lower()
        if any(sig in q for sig in SQL_KEYWORDS):
            return "sql_query"
        return "document_retrieval"

    def _route_with_llm(self, query: str) -> str:
        import ollama as ollama_pkg
        from app.config import OLLAMA_BASE_URL, OLLAMA_MODEL

        client = ollama_pkg.Client(host=OLLAMA_BASE_URL)
        prompt = (
            "Classify this question as exactly one of: sql_query, document_retrieval.\n"
            "sql_query = quantitative questions about accounts (spend, usage, region, plan, counts, rankings).\n"
            "document_retrieval = policy/process/product questions answered from documents.\n"
            f"Question: {query}\n"
            "Reply with only the label, nothing else."
        )
        resp = client.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}])
        label = resp["message"]["content"].strip().lower()
        return "sql_query" if "sql" in label else "document_retrieval"
