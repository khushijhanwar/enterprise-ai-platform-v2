"""
Workflow agent: a real LangGraph StateGraph orchestrating the planner,
SQL agent, and document agent. This is the direct open-source analogue
of a Google ADK multi-agent workflow -- a typed state object passed
between nodes, explicit conditional routing, and a graph that can be
visualized, inspected, and extended with more nodes (e.g. a validation
step, a human-in-the-loop approval node) without restructuring the
control flow.

Graph shape::

    planner --route--> sql_node --------> answer_node --> END
                   \\--> document_node --/
"""
from __future__ import annotations

from typing import TypedDict, Optional, Any

from langgraph.graph import StateGraph, END

from app.agents.planner import Planner
from app.agents.sql_agent import SQLAgent
from app.agents.document_agent import DocumentAgent


class WorkflowState(TypedDict):
    query: str
    route: Optional[str]
    sql: Optional[str]
    result: Optional[Any]      # pandas DataFrame when routed to SQL
    answer: Optional[str]
    sources: Optional[list]
    tool_used: Optional[str]


class WorkflowAgent:
    def __init__(self, rag_pipeline=None):
        self.planner = Planner()
        self.sql_agent = SQLAgent()
        self.document_agent = DocumentAgent(rag_pipeline)
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(WorkflowState)

        graph.add_node("planner", self._planner_node)
        graph.add_node("sql_node", self._sql_node)
        graph.add_node("document_node", self._document_node)

        graph.set_entry_point("planner")
        graph.add_conditional_edges(
            "planner",
            lambda state: state["route"],
            {"sql_query": "sql_node", "document_retrieval": "document_node"},
        )
        graph.add_edge("sql_node", END)
        graph.add_edge("document_node", END)

        return graph.compile()

    def _planner_node(self, state: WorkflowState) -> dict:
        route = self.planner.route(state["query"])
        return {"route": route}

    def _sql_node(self, state: WorkflowState) -> dict:
        out = self.sql_agent.answer(state["query"])
        return {
            "sql": out["sql"],
            "result": out["result"],
            "tool_used": "sql_query",
        }

    def _document_node(self, state: WorkflowState) -> dict:
        out = self.document_agent.answer(state["query"])
        return {
            "answer": out["answer"],
            "sources": out["sources"],
            "tool_used": "document_retrieval",
        }

    def handle(self, query: str) -> dict:
        final_state = self.graph.invoke({"query": query})
        return final_state


if __name__ == "__main__":
    agent = WorkflowAgent()
    for q in [
        "What are the top 5 accounts by spend?",
        "How long does enterprise onboarding take?",
    ]:
        out = agent.handle(q)
        print(f"\nQ: {q}\n-> tool: {out['tool_used']}")
        if out["tool_used"] == "sql_query":
            print(out["result"])
        else:
            print(out["answer"])
