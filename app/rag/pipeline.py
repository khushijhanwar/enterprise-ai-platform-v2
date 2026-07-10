"""
RAG pipeline: wires loader -> splitter -> embeddings -> FAISS -> retriever
-> reranker -> LLM into one buildable/queryable object. This is the
LangChain-loader-to-Ollama chain described in the architecture doc.
"""
from __future__ import annotations

import logging

from app.config import SAMPLE_DOCS_DIR, EMBEDDING_DIM, FAISS_INDEX_PATH
from app.rag.loader import load_documents
from app.rag.splitter import split_documents
from app.rag.embeddings import get_embedder
from app.rag.vectorstore import FaissVectorStore
from app.rag.reranker import get_reranker
from app.rag.retriever import Retriever
from app.llm.ollama import OllamaLLM, ExtractiveFallbackLLM

logger = logging.getLogger(__name__)

RAG_SYSTEM_PROMPT = """You are an enterprise knowledge assistant. Answer the
user's question using ONLY the provided context. If the context does not
contain the answer, say so explicitly. Cite the source filename for each
claim in square brackets, e.g. [refund_policy.txt]."""


class RAGPipeline:
    def __init__(self, doc_dir=SAMPLE_DOCS_DIR, rebuild: bool = False):
        self.embedder = get_embedder()
        self.reranker = get_reranker()
        self.llm = self._get_llm()

        if not rebuild and FaissVectorStore.exists():
            logger.info("Loading existing FAISS index from disk.")
            self.vectorstore = FaissVectorStore.load()
            # The fallback TF-IDF vectorizer isn't itself persisted, so
            # re-fit it on the same (already-chunked) corpus in the same
            # order to reproduce the identical vector space. BGE's fit()
            # is a no-op since it's a pretrained model.
            texts = [d.page_content for d in self.vectorstore.documents]
            self.embedder.fit(texts)
        else:
            self.vectorstore = self._build_index(doc_dir)

        self.retriever = Retriever(self.embedder, self.vectorstore, self.reranker)

    def _get_llm(self):
        llm = OllamaLLM()
        if llm.is_available():
            logger.info("Ollama server detected -- using real Qwen2.5 generation.")
            return llm
        logger.warning(
            "Ollama server not reachable at startup. Falling back to extractive "
            "answers. Install Ollama (https://ollama.com) and run "
            "`ollama pull qwen2.5:7b` to enable full grounded generation."
        )
        return ExtractiveFallbackLLM()

    def _build_index(self, doc_dir) -> FaissVectorStore:
        docs = load_documents(doc_dir)
        chunks = split_documents(docs)
        texts = [c.page_content for c in chunks]

        embeddings = self.embedder.embed_documents(texts)
        dim = embeddings.shape[1]

        store = FaissVectorStore(dim=dim)
        store.add(embeddings, chunks)
        store.save()
        logger.info(f"Built FAISS index: {len(chunks)} chunks, dim={dim}")
        return store

    def answer(self, query: str) -> dict:
        sources = self.retriever.retrieve(query)
        answer_text = self.llm.generate(RAG_SYSTEM_PROMPT, query, sources)
        return {"query": query, "answer": answer_text, "sources": sources}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    rag = RAGPipeline(rebuild=True)
    result = rag.answer("How long does enterprise onboarding take?")
    print(result["answer"])
