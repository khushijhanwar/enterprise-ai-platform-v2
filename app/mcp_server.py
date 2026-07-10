"""
MCP server: exposes the platform's SQL and document-retrieval
capabilities as standardized MCP tools using FastMCP.

This is real MCP -- any MCP-compatible client (Claude Desktop, an ADK
agent configured with an MCP client, etc.) can discover and call these
tools without knowing anything about DuckDB, FAISS, or how they're
implemented internally. That decoupling -- tool contract separate from
implementation -- is the actual point of MCP, and it's preserved here.

Run standalone (stdio transport, for MCP clients like Claude Desktop):
    python -m app.mcp_server

Run as HTTP (for testing/inspection):
    fastmcp run app/mcp_server.py --transport http --port 8001
"""
from __future__ import annotations

from fastmcp import FastMCP

from app.agents.sql_agent import SQLAgent
from app.agents.document_agent import DocumentAgent
from app.graph.knowledge_graph import build_knowledge_graph, related_concepts

mcp = FastMCP("enterprise-ai-platform")

_sql_agent: SQLAgent | None = None
_document_agent: DocumentAgent | None = None
_graph = None


def _get_sql_agent() -> SQLAgent:
    global _sql_agent
    if _sql_agent is None:
        _sql_agent = SQLAgent()
    return _sql_agent


def _get_document_agent() -> DocumentAgent:
    global _document_agent
    if _document_agent is None:
        _document_agent = DocumentAgent()
    return _document_agent


def _get_graph():
    global _graph
    if _graph is None:
        _graph = build_knowledge_graph()
    return _graph


@mcp.tool
def sql_query(question: str) -> dict:
    """Answer a quantitative question about customer accounts (spend,
    usage, region, plan, counts, rankings) by generating and running SQL
    against the analytics warehouse."""
    agent = _get_sql_agent()
    out = agent.answer(question)
    return {
        "sql": out["sql"],
        "engine": out["engine"],
        "rows": out["result"].to_dict(orient="records"),
    }


@mcp.tool
def document_search(question: str) -> dict:
    """Answer a policy, process, or product question by retrieving
    grounded context from company documents (refunds, onboarding,
    security, pricing, architecture) and generating a cited answer."""
    agent = _get_document_agent()
    out = agent.answer(question)
    return {"answer": out["answer"], "sources": out["sources"]}


@mcp.tool
def related_knowledge_concepts(concept: str) -> list[str]:
    """Given a concept (e.g. 'refund', 'security', 'onboarding'), return
    related concepts from the knowledge graph built over the document
    corpus -- useful for exploring what topics are connected."""
    G = _get_graph()
    return related_concepts(G, concept.lower())


if __name__ == "__main__":
    mcp.run()
