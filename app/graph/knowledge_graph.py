"""
Graph analytics for knowledge discovery, built with NetworkX.

Entities/concepts are extracted per chunk and linked by co-occurrence,
producing a graph that surfaces which concepts bridge multiple topics
(via betweenness centrality) -- the same kind of signal a production
knowledge-graph layer (Neo4j, Spanner Graph) would surface, at a scale
NetworkX handles comfortably in-memory.
"""
from __future__ import annotations

from collections import Counter

import networkx as nx

from app.rag.loader import load_documents
from app.rag.splitter import split_documents

KEY_TERMS = [
    "refund", "onboarding", "security", "pricing", "enterprise", "encryption",
    "compliance", "csm", "sla", "api", "gdpr", "hipaa", "soc 2", "renewal",
    "provisioning", "escalation", "discount", "vector", "embedding", "orchestration",
    "warehouse", "pipeline", "retrieval",
]


def extract_terms(text: str) -> list[str]:
    text_lower = text.lower()
    return [term for term in KEY_TERMS if term in text_lower]


def build_knowledge_graph() -> nx.Graph:
    docs = load_documents()
    chunks = split_documents(docs)
    G = nx.Graph()

    for chunk in chunks:
        terms = set(extract_terms(chunk.page_content))
        for term in terms:
            G.add_node(term)
        for t1 in terms:
            for t2 in terms:
                if t1 < t2:
                    if G.has_edge(t1, t2):
                        G[t1][t2]["weight"] += 1
                    else:
                        G.add_edge(t1, t2, weight=1)
    return G


def top_central_concepts(G: nx.Graph, top_k: int = 8) -> list[tuple[str, float]]:
    centrality = nx.betweenness_centrality(G, weight="weight")
    return Counter(centrality).most_common(top_k)


def related_concepts(G: nx.Graph, term: str, top_k: int = 5) -> list[str]:
    if term not in G:
        return []
    neighbors = sorted(G[term].items(), key=lambda x: x[1]["weight"], reverse=True)
    return [n for n, _ in neighbors[:top_k]]


if __name__ == "__main__":
    G = build_knowledge_graph()
    print(f"[graph] nodes={G.number_of_nodes()} edges={G.number_of_edges()}")
    for term, score in top_central_concepts(G):
        print(f"  {term}: {score:.3f}")
