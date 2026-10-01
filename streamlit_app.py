"""
Streamlit frontend.

Talks to the FastAPI backend over HTTP -- the same client/server split a
production deployment would have (Streamlit or a real frontend calling a
FastAPI service), rather than importing the pipeline in-process. This
means the two run as separate processes:

    Terminal 1:  uvicorn app.main:app --reload --port 8000
    Terminal 2:  streamlit run streamlit_app.py
"""
import base64
import hashlib
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

tab1, tab_voice, tab2, tab3 = st.tabs(["Ask the platform", "Ask by voice", "Knowledge graph", "Ingest"])

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

with tab_voice:
    st.subheader("Ask out loud")
    st.write(
        "Same agent workflow, with speech on both ends: ElevenLabs Scribe transcribes the "
        "question, the planner routes it, and the answer is spoken back with its source."
    )

    try:
        status = requests.get(f"{API_BASE}/api/voice/status", timeout=5).json()
    except requests.exceptions.RequestException:
        status = {"voice_enabled": False}

    if not status.get("voice_enabled"):
        st.info("Voice is off: add ELEVENLABS_API_KEY to .env and restart the API. Typed questions still work below.")

    recording = None
    if status.get("voice_enabled"):
        if hasattr(st, "audio_input"):
            recording = st.audio_input("Record your question")
        else:  # older Streamlit: no microphone widget
            recording = st.file_uploader("Upload a recorded question", type=["wav", "mp3", "m4a", "webm"])
    typed = st.text_input("...or type it", key="voice_typed")

    # Streamlit reruns this script on every interaction. Cache the last
    # answer by a hash of the question so a rerun never re-bills a
    # transcription or a text-to-speech call.
    request_key = None
    if recording is not None:
        audio_bytes = recording.getvalue()
        request_key = "audio:" + hashlib.sha1(audio_bytes).hexdigest()
    elif typed:
        request_key = "text:" + typed

    if request_key and st.session_state.get("voice_key") != request_key:
        with st.spinner("Listening, routing, speaking..."):
            try:
                if recording is not None:
                    resp = requests.post(
                        f"{API_BASE}/api/voice/ask-audio",
                        files={"file": (recording.name or "question.wav", audio_bytes, recording.type or "audio/wav")},
                        timeout=120,
                    )
                else:
                    resp = requests.post(f"{API_BASE}/api/voice/ask", json={"query": typed}, timeout=120)
                if resp.status_code >= 400:
                    st.error(resp.json().get("detail", resp.text))
                else:
                    st.session_state["voice_key"] = request_key
                    st.session_state["voice_result"] = resp.json()
                    st.session_state["voice_autoplay"] = True
            except requests.exceptions.RequestException as e:
                st.error(f"Request failed: {e}")

    voice_result = st.session_state.get("voice_result") if request_key else None
    if voice_result:
        if voice_result.get("transcript"):
            st.markdown(f"**Heard:** {voice_result['transcript']}")
        st.markdown(f"**Routed to tool:** `{voice_result['tool_used']}`")
        st.markdown(f"**Spoken answer:** {voice_result['spoken_text']}")

        if voice_result.get("audio_base64"):
            st.audio(
                base64.b64decode(voice_result["audio_base64"]),
                format=voice_result["audio_mime"],
                autoplay=st.session_state.pop("voice_autoplay", False),
            )

        timings = voice_result.get("timings_ms", {})
        if timings:
            for col, (stage, ms) in zip(st.columns(len(timings)), timings.items()):
                col.metric(stage, f"{ms} ms")

        with st.expander("What the agent returned"):
            if voice_result["tool_used"] == "sql_query":
                st.code(voice_result["sql"], language="sql")
                st.dataframe(voice_result["result"], use_container_width=True)
            else:
                st.write(voice_result["answer"])
                for s in voice_result["sources"] or []:
                    st.markdown(f"**{s['source']}** (score={s['score']:.3f})")

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
