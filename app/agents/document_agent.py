"""
Document agent: thin wrapper exposing the RAG pipeline as an agent tool
with a uniform interface, so the workflow graph can call it the same way
it calls the SQL agent.
"""
from __future__ import annotations

from app.rag.pipeline import RAGPipeline


class DocumentAgent:
    def __init__(self, rag_pipeline: RAGPipeline | None = None):
        self.rag = rag_pipeline or RAGPipeline()

    def answer(self, question: str) -> dict:
        return self.rag.answer(question)
