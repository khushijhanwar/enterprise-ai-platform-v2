"""
Embeddings: BAAI/bge-small-en-v1.5 via sentence-transformers.

BGE is currently one of the strongest open-weight embedding model
families (top of the MTEB leaderboard among small local-friendly models),
which is why it's the primary path here rather than a placeholder. It
runs entirely locally, on CPU, no API key required.

Production swap: replace `BGEEmbeddings` with a thin wrapper around
`genai.embed_content(model="text-embedding-004", ...)` -- the
`embed_documents`/`embed_query` interface below matches LangChain's
Embeddings base class, so the swap doesn't touch any calling code.

Offline fallback: the very first time this runs, sentence-transformers
downloads the BGE weights (~130MB) from Hugging Face. If that download
can't complete (no network, or a locked-down environment), this module
transparently falls back to a local TF-IDF vectorizer so the rest of the
pipeline keeps working end-to-end. The fallback is logged loudly so it's
never silently mistaken for the real model.
"""
from __future__ import annotations

import logging

import numpy as np

from app.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)


class BGEEmbeddings:
    """Real BGE embeddings via sentence-transformers."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        # BGE recommends this instruction prefix for retrieval passages.
        return np.array(self.model.encode(texts, normalize_embeddings=True))

    def embed_query(self, text: str) -> np.ndarray:
        # BGE recommends an instruction prefix for queries specifically,
        # which measurably improves retrieval quality over no prefix.
        prefixed = f"Represent this sentence for searching relevant passages: {text}"
        return np.array(self.model.encode([prefixed], normalize_embeddings=True))[0]

    def fit(self, texts: list[str]) -> None:
        pass  # BGE is a pretrained model, no corpus-specific fitting needed


class TfidfFallbackEmbeddings:
    """Zero-dependency fallback used only if BGE weights can't be
    downloaded. Not the intended production path -- see module docstring."""

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(stop_words="english", max_features=4096)
        self._fitted = False

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        matrix = self._vectorizer.fit_transform(texts)
        self._fitted = True
        return matrix.toarray()

    def embed_query(self, text: str) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call embed_documents on the corpus before embed_query().")
        return self._vectorizer.transform([text]).toarray()[0]

    def fit(self, texts: list[str]) -> None:
        """Re-fit the vectorizer on the same corpus without touching the
        FAISS index -- needed when loading a previously-saved index from
        disk, since the fallback vectorizer's fitted state isn't itself
        persisted."""
        self._vectorizer.fit(texts)
        self._fitted = True


def get_embedder():
    """Returns real BGE embeddings, or logs a clear warning and falls
    back to TF-IDF if the model can't be loaded (e.g. no internet access
    to Hugging Face on first run)."""
    try:
        embedder = BGEEmbeddings()
        logger.info(f"Loaded BGE embedding model: {EMBEDDING_MODEL}")
        return embedder
    except Exception as e:
        logger.warning(
            f"Could not load BGE embeddings ({e}). Falling back to local "
            f"TF-IDF embeddings. Run again with internet access to "
            f"auto-download {EMBEDDING_MODEL} for real semantic embeddings."
        )
        return TfidfFallbackEmbeddings()
