"""Shows exactly what vector search + reranking choose for a query, with
scores, so retrieval issues are visible instead of just wrong answers."""
import sys
from app.rag.embeddings import get_embedder
from app.rag.vectorstore import FaissVectorStore
from app.rag.reranker import get_reranker

def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "How long does enterprise onboarding take?"
    embedder = get_embedder()
    reranker = get_reranker()

    if not FaissVectorStore.exists():
        print("No FAISS index found. Run `python demo.py` once first.")
        return

    store = FaissVectorStore.load()
    embedder.fit([d.page_content for d in store.documents])  # no-op for real BGE

    print(f"Corpus: {len(store.documents)} chunks\nQuery: {query}\n")
    query_embedding = embedder.embed_query(query)

    print("=== STAGE A: raw vector search (top 10, before rerank) ===")
    candidates = store.search(query_embedding, top_k=10)
    for doc, score in candidates:
        preview = doc.page_content[:90].replace("\n", " ")
        print(f"  [{score:.3f}] {doc.metadata['source']} (chunk {doc.metadata['chunk_id']}): {preview}...")

    print("\n=== STAGE B: after cross-encoder rerank (top 5) ===")
    reranked = reranker.rerank(query, candidates, top_k=5)
    for doc, score in reranked:
        preview = doc.page_content[:90].replace("\n", " ")
        print(f"  [{score:.3f}] {doc.metadata['source']} (chunk {doc.metadata['chunk_id']}): {preview}...")

if __name__ == "__main__":
    main()
