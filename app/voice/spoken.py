"""
Turns an agent result into something worth saying out loud.

A chat UI can show a SQL table and a list of cited chunks. A voice
answer cannot: nobody wants a DataFrame read to them. This module
converts either agent's output into one short spoken answer, and caps
its length, since text-to-speech is billed per character.

Pure functions, no network and no SDK, so they are unit-tested directly.
"""
from __future__ import annotations

import re
from typing import Any

MAX_ROWS_SPOKEN = 5

# Column names as they should be said, for the columns the warehouse has.
_COLUMN_SPEECH = {
    "monthly_spend_usd": "monthly spend",
    "api_calls_last_30d": "API calls in the last 30 days",
}


def _rows(result: Any) -> list[dict]:
    """Accepts a pandas DataFrame or an already-serialized list of dicts."""
    if result is None:
        return []
    if hasattr(result, "to_dict"):
        return result.to_dict(orient="records")
    return list(result)


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _say_number(value: Any, column: str) -> str:
    if column.endswith("_usd"):
        return f"{round(value):,} dollars"
    if isinstance(value, float) and not value.is_integer():
        return f"{value:,.2f}"
    return f"{int(value):,}"


def _say_column(column: str) -> str:
    if column in _COLUMN_SPEECH:
        return _COLUMN_SPEECH[column]
    return column.removesuffix("_usd").replace("_", " ")


def _speak_sql(result: Any) -> str:
    rows = _rows(result)
    if not rows:
        return "I ran the query, but it returned no rows."

    columns = list(rows[0].keys())

    # Single value, e.g. SELECT COUNT(*) AS account_count
    if len(rows) == 1 and len(columns) == 1:
        col, val = columns[0], rows[0][columns[0]]
        said = _say_number(val, col) if _is_number(val) else str(val)
        return f"The {_say_column(col)} is {said}."

    label_col = next((c for c in columns if "name" in c), columns[0])
    value_col = next((c for c in reversed(columns) if _is_number(rows[0][c])), None)

    spoken_rows = rows[:MAX_ROWS_SPOKEN]
    parts = []
    for r in spoken_rows:
        if value_col:
            parts.append(f"{r[label_col]} with {_say_number(r[value_col], value_col)}")
        else:
            parts.append(str(r[label_col]))

    count = len(rows)
    lead = f"I found {count} result{'s' if count != 1 else ''}"
    if value_col:
        # "showing", not "ranked by": the SQL is LLM-written, so the
        # spoken answer only claims what the rows themselves prove.
        lead += f", showing {_say_column(value_col)}"
    text = f"{lead}. " + "; ".join(parts) + "."
    if count > len(spoken_rows):
        text += f" The other {count - len(spoken_rows)} are on screen."
    return text


def _clean_for_speech(text: str) -> str:
    text = re.sub(r"\(Extractive fallback mode.*?\)", "", text, flags=re.S)
    text = re.sub(r"\[[^\]]+\.(?:txt|md|pdf)\]", "", text)  # inline [file.txt] citations
    text = re.sub(r"[*_`#>]", "", text)                      # markdown
    text = re.sub(r"^\s*-\s+", "", text, flags=re.M)         # bullets
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"\s+([.,;:!?])", r"\1", text)              # gap left by a removed citation


def _source_name(sources: list[dict] | None) -> str | None:
    if not sources:
        return None
    name = str(sources[0].get("source", "")).rsplit("/", 1)[-1]
    name = re.sub(r"\.[a-z0-9]+$", "", name).replace("_", " ").replace("-", " ").strip()
    return name or None


def truncate_at_sentence(text: str, max_chars: int) -> str:
    """Cuts at the last full sentence that fits, so speech never stops
    mid-word. Falls back to a word boundary for one very long sentence."""
    if len(text) <= max_chars:
        return text
    window = text[:max_chars]
    end = max(window.rfind(". "), window.rfind("? "), window.rfind("! "))
    if end >= max_chars // 3:
        return window[: end + 1]
    return window.rsplit(" ", 1)[0].rstrip(",;:") + "."


def to_spoken_text(agent_output: dict, max_chars: int = 400) -> str:
    """Agent output (either route) -> one short, speakable answer."""
    if agent_output.get("tool_used") == "sql_query":
        return truncate_at_sentence(_speak_sql(agent_output.get("result")), max_chars)

    answer = _clean_for_speech(agent_output.get("answer") or "")
    if not answer:
        return "I could not find an answer to that in the documents."

    source = _source_name(agent_output.get("sources"))
    citation = f" Source: {source}." if source else ""
    # Reserve room for the citation: a spoken answer keeps its source
    # even when the body has to be cut.
    body = truncate_at_sentence(answer, max(max_chars - len(citation), 40))
    return body + citation
