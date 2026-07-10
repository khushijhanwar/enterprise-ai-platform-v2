"""Chat endpoint: routes through the LangGraph workflow agent."""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from app.agents.workflow_agent import WorkflowAgent

router = APIRouter()
_agent: WorkflowAgent | None = None


def get_agent() -> WorkflowAgent:
    global _agent
    if _agent is None:
        _agent = WorkflowAgent()
    return _agent


class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    query: str
    tool_used: str
    answer: str | None = None
    sql: str | None = None
    result: list[dict] | None = None
    sources: list[dict] | None = None


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    agent = get_agent()
    out = agent.handle(req.query)

    if out["tool_used"] == "sql_query":
        return ChatResponse(
            query=req.query,
            tool_used="sql_query",
            sql=out["sql"],
            result=out["result"].to_dict(orient="records"),
        )
    return ChatResponse(
        query=req.query,
        tool_used="document_retrieval",
        answer=out["answer"],
        sources=out["sources"],
    )
