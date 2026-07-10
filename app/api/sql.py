"""SQL endpoint: direct access to the NL-to-SQL agent, bypassing routing."""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from app.agents.sql_agent import SQLAgent

router = APIRouter()
_sql_agent: SQLAgent | None = None


def get_sql_agent() -> SQLAgent:
    global _sql_agent
    if _sql_agent is None:
        _sql_agent = SQLAgent()
    return _sql_agent


class SQLRequest(BaseModel):
    question: str


class SQLResponse(BaseModel):
    question: str
    sql: str
    engine: str
    result: list[dict]


@router.post("/sql", response_model=SQLResponse)
def query_sql(req: SQLRequest) -> SQLResponse:
    agent = get_sql_agent()
    out = agent.answer(req.question)
    return SQLResponse(
        question=req.question,
        sql=out["sql"],
        engine=out["engine"],
        result=out["result"].to_dict(orient="records"),
    )
