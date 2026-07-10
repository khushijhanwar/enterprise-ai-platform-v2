"""
Streamlit frontend.

Talks to the FastAPI backend over HTTP -- the same client/server split a
production deployment would have (Streamlit or a real frontend calling a
FastAPI service), rather than importing the pipeline in-process. This
means the two run as separate processes:

    Terminal 1:  uvicorn app.main:app --reload --port 8000
    Terminal 2:  streamlit run streamlit_app.py
"""
import os

import requests
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Enterprise AI Platform", layout="wide")

st.title("Enterprise AI Platform")
st.caption(
    "Production-architecture RAG + agent platform on a free local stack: "
    "PySpark → DuckDB → LangChain → BGE embeddings → FAISS → cross-encoder rerank "
    "→ LangGraph → Ollama (Qwen2.5) → FastAPI → FastMCP."
)


def api_available() -> bool:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=2)
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False


if not api_available():
    st.error(
        f"Can't reach the FastAPI backend at {API_BASE}. Start it first with:\n\n"
        f"`uvicorn app.main:app --reload --port 8000`"
    )
    st.stop()

tab1, tab2, tab3 = st.tabs(["Ask the platform", "Knowledge graph", "Ingest"])

with tab1:
    st.subheader("Ask a question")
    st.write("Analytical questions route to the SQL agent; policy/product questions route to RAG.")

    example_cols = st.columns(3)
    examples = [
        "What are the top 5 accounts by spend?",
        "How long does enterprise onboarding take?",
        "What compliance certifications does the platform have?",
    ]
    for col, ex in zip(example_cols, examples):
        if col.button(ex):
            st.session_state["query"] = ex

    query = st.text_input("Your question", value=st.session_state.get("query", ""))

    if query:
        with st.spinner("Routing through the agent workflow..."):
            try:
                resp = requests.post(f"{API_BASE}/api/chat", json={"query": query}, timeout=60)
                resp.raise_for_status()
                result = resp.json()
            except requests.exceptions.RequestException as e:
                st.error(f"Request failed: {e}")
                result = None

        if result:
            st.markdown(f"**Routed to tool:** `{result['tool_used']}`")
            if result["tool_used"] == "sql_query":
                st.code(result["sql"], language="sql")
                st.dataframe(result["result"], use_container_width=True)
            else:
                st.write(result["answer"])
                with st.expander("Retrieved sources"):
                    for s in result["sources"]:
                        st.markdown(f"**{s['source']}** (score={s['score']:.3f})")
                        st.text(s["text"])

with tab2:
    st.subheader("Knowledge graph — concept co-occurrence")
    st.write("Built with NetworkX over the document corpus. Betweenness centrality surfaces bridging concepts.")

    concept = st.text_input("Explore related concepts for:", value="security")
    if concept:
        try:
            from app.graph.knowledge_graph import build_knowledge_graph, top_central_concepts, related_concepts

            G = build_knowledge_graph()
            st.write(f"{G.number_of_nodes()} concepts, {G.number_of_edges()} relations extracted.")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Top bridging concepts**")
                top_terms = top_central_concepts(G, top_k=8)
                st.bar_chart({t: s for t, s in top_terms})
            with col2:
                st.markdown(f"**Related to '{concept}'**")
                related = related_concepts(G, concept.lower())
                if related:
                    for r in related:
                        st.markdown(f"- {r}")
                else:
                    st.write("No related concepts found (try: security, refund, onboarding, pricing, enterprise).")
        except Exception as e:
            st.error(f"Graph module error: {e}")

with tab3:
    st.subheader("Data ingestion")
    st.write("Triggers the real PySpark ETL job: reads `accounts_raw.csv`, transforms, lands in DuckDB.")
    if st.button("Run Spark ETL pipeline"):
        with st.spinner("Running Spark job (local mode)..."):
            try:
                resp = requests.post(f"{API_BASE}/api/ingest", timeout=120)
                resp.raise_for_status()
                st.success(resp.json())
            except requests.exceptions.RequestException as e:
                st.error(f"Request failed: {e}")
