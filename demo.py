"""
One-shot CLI demo: exercises the full architecture end-to-end in a single
process, with no servers to start. Good for a first run to confirm
everything works before spinning up FastAPI + Streamlit.

Run with: python demo.py
"""
import logging

logging.basicConfig(level=logging.WARNING, format="%(message)s")


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main():
    section("STAGE 1 — Spark ETL (real PySpark, local mode) → DuckDB")
    from app.etl.spark_pipeline import run_etl

    run_etl()

    section("STAGE 2 — Agent workflow (LangGraph: planner → SQL / RAG)")
    from app.agents.workflow_agent import WorkflowAgent

    agent = WorkflowAgent()
    questions = [
        "What are the top 5 accounts by spend?",
        "How long does enterprise onboarding take?",
        "What's the refund policy for annual contracts?",
        "How many accounts are on the Enterprise plan in EMEA?",
        "What compliance certifications does the platform have?",
    ]
    for q in questions:
        out = agent.handle(q)
        print(f"\nQ: {q}")
        print(f"[routed to: {out['tool_used']}]")
        if out["tool_used"] == "sql_query":
            print(out["sql"])
            print(out["result"].to_string(index=False))
        else:
            print(out["answer"])

    section("STAGE 3 — Graph analytics (NetworkX knowledge graph)")
    from app.graph.knowledge_graph import build_knowledge_graph, top_central_concepts

    G = build_knowledge_graph()
    print(f"Knowledge graph: {G.number_of_nodes()} concepts, {G.number_of_edges()} relations")
    for term, score in top_central_concepts(G):
        print(f"  bridging concept: {term} ({score:.3f})")

    section("STAGE 4 — MCP tool layer (FastMCP)")
    import asyncio
    from app.mcp_server import mcp

    async def check_mcp():
        tools = await mcp.list_tools()
        print(f"MCP server '{mcp.name}' exposing {len(tools)} tools:")
        for t in tools:
            print(f"  - {t.name}")
        result = await mcp.call_tool("sql_query", {"question": "How many accounts are in NA?"})
        print(f"\nSample MCP call (sql_query): {result.structured_content}")

    asyncio.run(check_mcp())

    section("Done")
    print(
        "Next: start the full app with two terminals —\n"
        "  uvicorn app.main:app --reload --port 8000\n"
        "  streamlit run streamlit_app.py\n"
    )


if __name__ == "__main__":
    main()
