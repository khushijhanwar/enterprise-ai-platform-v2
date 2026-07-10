"""
Reranker: BGE cross-encoder (BAAI/bge-reranker-base).

A cross-encoder scores (query, passage) pairs jointly rather than
comparing separately-computed embeddings, which is slower but
meaningfully more accurate -- this is why production RAG systems
retrieve a wide candidate set (top ~8) via the vector store and then
rerank down to the top ~3 before generation, rather than trusting raw
vector similarity alone.

Falls back to a pass-through (keep vector-search order) if the model
can't be downloaded, same reasoning as embeddings.py.
"""
from __future__ import annotations

import logging

from langchain_core.documents import Document

from app.config import RERANKER_MODEL, RERANK_TOP_K

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    def __init__(self, model_name: str = RERANKER_MODEL):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def rerank(
        self, query: str, candidates: list[tuple[Document, float]], top_k: int = RERANK_TOP_K
    ) -> list[tuple[Document, float]]:
        if not candidates:
            return []
        pairs = [[query, doc.page_content] for doc, _ in candidates]
        scores = self.model.predict(pairs)
        reranked = sorted(zip([c[0] for c in candidates], scores), key=lambda x: x[1], reverse=True)
        return [(doc, float(score)) for doc, score in reranked[:top_k]]


class PassthroughReranker:
    """Fallback: keeps the vector-search ranking as-is."""

    def rerank(
        self, query: str, candidates: list[tuple[Document, float]], top_k: int = RERANK_TOP_K
    ) -> list[tuple[Document, float]]:
        return candidates[:top_k]


def get_reranker():
    try:
        reranker = CrossEncoderReranker()
        logger.info(f"Loaded cross-encoder reranker: {RERANKER_MODEL}")
        return reranker
    except Exception as e:
        logger.warning(
            f"Could not load cross-encoder reranker ({e}). Falling back to "
            f"passthrough (vector-search order preserved)."
        )
        return PassthroughReranker()
