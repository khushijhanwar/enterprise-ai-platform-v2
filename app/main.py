"""
FastAPI application entry point.

Run with: uvicorn app.main:app --reload --port 8000
Docs at:  http://localhost:8000/docs
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import chat, sql, ingest

app = FastAPI(
    title="Enterprise AI Platform",
    description=(
        "Production-architecture RAG + agent platform, running on a fully "
        "free/local stack: PySpark, DuckDB, FAISS, BGE embeddings, "
        "LangChain, LangGraph, Ollama, FastMCP."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/api", tags=["chat"])
app.include_router(sql.router, prefix="/api", tags=["sql"])
app.include_router(ingest.router, prefix="/api", tags=["ingest"])


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
