"""
SQL agent: turns a natural-language question into SQL against the
warehouse. Uses Ollama for real NL->SQL when available; falls back to
intent-matching rules (still real SQL execution, just not LLM-written)
so the agent works before Ollama is set up.
"""
from __future__ import annotations

import re

from app.warehouse.duckdb import execute_query, SCHEMA_DESCRIPTION
from app.llm.ollama import OllamaLLM


def _rule_based_sql(question: str) -> str:
    q = question.lower()

    m = re.search(r"top (\d+)", q)
    top_n = int(m.group(1)) if m else 5

    if "spend" in q or "revenue" in q:
        order_col = "monthly_spend_usd"
    elif "api call" in q or "usage" in q:
        order_col = "api_calls_last_30d"
    else:
        order_col = "monthly_spend_usd"

    where_clauses = []
    for region in ["NA", "EMEA", "APAC", "LATAM"]:
        if region.lower() in q:
            where_clauses.append(f"region = '{region}'")
    for plan in ["Starter", "Growth", "Enterprise"]:
        if plan.lower() in q:
            where_clauses.append(f"plan = '{plan}'")

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    if "how many" in q or re.search(r"\bcount\b", q):
        return f"SELECT COUNT(*) AS account_count FROM accounts {where_sql}"

    return (
        f"SELECT account_id, account_name, region, plan, {order_col} "
        f"FROM accounts {where_sql} ORDER BY {order_col} DESC LIMIT {top_n}"
    )


class SQLAgent:
    def __init__(self):
        self.llm = OllamaLLM()

    def answer(self, question: str) -> dict:
        if self.llm.is_available():
            try:
                sql = self.llm.generate_sql(SCHEMA_DESCRIPTION, question)
                result_df = execute_query(sql)
                return {"question": question, "sql": sql, "result": result_df, "engine": "ollama-nl2sql"}
            except Exception:
                pass  # fall through to rule-based

        sql = _rule_based_sql(question)
        result_df = execute_query(sql)
        return {"question": question, "sql": sql, "result": result_df, "engine": "rule-based"}


if __name__ == "__main__":
    agent = SQLAgent()
    out = agent.answer("What are the top 5 accounts by spend in the Enterprise plan?")
    print(out["engine"], "\n", out["sql"], "\n", out["result"])
