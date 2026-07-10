"""
Retriever: orchestrates embed -> FAISS search -> cross-encoder rerank.

This is the "wide recall, then precise rerank" pattern used in
production retrieval systems -- pull a generous candidate set (top ~8)
from the fast vector index, then spend the more expensive cross-encoder
pass only on those candidates to pick the final top ~3.
"""
from __future__ import annotations

from langchain_core.documents import Document

from app.config import RETRIEVAL_TOP_K, RERANK_TOP_K, RERANK_ENABLED
from app.rag.vectorstore import FaissVectorStore


class Retriever:
    def __init__(self, embedder, vectorstore: FaissVectorStore, reranker=None):
        self.embedder = embedder
        self.vectorstore = vectorstore
        self.reranker = reranker

    def retrieve(self, query: str, top_k: int = RETRIEVAL_TOP_K, rerank_top_k: int = RERANK_TOP_K) -> list[dict]:
        query_embedding = self.embedder.embed_query(query)
        candidates = self.vectorstore.search(query_embedding, top_k=top_k)

        if RERANK_ENABLED and self.reranker is not None:
            final = self.reranker.rerank(query, candidates, top_k=rerank_top_k)
        else:
            final = candidates[:rerank_top_k]

        return [
            {
                "source": doc.metadata.get("source", "unknown"),
                "chunk_id": doc.metadata.get("chunk_id", -1),
                "text": doc.page_content,
                "score": score,
            }
            for doc, score in final
        ]
