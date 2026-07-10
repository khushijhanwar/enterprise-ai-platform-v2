"""
Vector store: FAISS (Facebook AI Similarity Search).

FAISS is the same ANN library that underpins a large share of production
vector search systems -- it's not a toy stand-in, it's genuinely
production-grade, just running in-process instead of as a managed
service. This uses a flat inner-product index (exact search); for larger
corpora you'd swap `IndexFlatIP` for an approximate index like
`IndexIVFFlat` or `IndexHNSWFlat`, same FAISS API.

Production swap: replace this module with AlloyDB Vector Search --
`CREATE INDEX ... USING ivfflat` plus `SELECT ... ORDER BY embedding <-> query`.
The `add()`/`search()` interface below is written to mirror that query
shape so the swap is mechanical.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import faiss
import numpy as np
from langchain_core.documents import Document

from app.config import FAISS_INDEX_PATH


class FaissVectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # inner product == cosine sim on normalized vectors
        self.documents: list[Document] = []

    def add(self, embeddings: np.ndarray, documents: list[Document]) -> None:
        embeddings = embeddings.astype("float32")
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.documents.extend(documents)

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[Document, float]]:
        q = query_embedding.astype("float32").reshape(1, -1)
        faiss.normalize_L2(q)
        scores, indices = self.index.search(q, min(top_k, len(self.documents)))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.documents[idx], float(score)))
        return results

    def save(self, path: str = FAISS_INDEX_PATH) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, f"{path}/index.faiss")
        with open(f"{path}/documents.pkl", "wb") as f:
            pickle.dump(self.documents, f)

    @classmethod
    def load(cls, path: str = FAISS_INDEX_PATH) -> "FaissVectorStore":
        index = faiss.read_index(f"{path}/index.faiss")
        with open(f"{path}/documents.pkl", "rb") as f:
            documents = pickle.load(f)
        store = cls(dim=index.d)
        store.index = index
        store.documents = documents
        return store

    @staticmethod
    def exists(path: str = FAISS_INDEX_PATH) -> bool:
        return Path(f"{path}/index.faiss").exists()
