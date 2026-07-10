"""
Central configuration.

All the "which backend am I actually pointed at" decisions live here, so
swapping a local free component for its production/cloud counterpart is a
one-line change, not a rewrite. Every setting is overridable via
environment variables (.env).
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
SAMPLE_DOCS_DIR = DATA_DIR / "sample_docs"
ACCOUNTS_CSV = DATA_DIR / "accounts.csv"

# --- Warehouse (DuckDB locally / BigQuery in production) ---
DUCKDB_PATH = str(DATA_DIR / "warehouse.duckdb")

# --- Vector store (FAISS locally / AlloyDB Vector Search in production) ---
FAISS_INDEX_PATH = str(DATA_DIR / "faiss_index")

# --- Embeddings (BGE via sentence-transformers locally / Gemini embeddings in production) ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))  # bge-small-en-v1.5 dimension

# --- Reranker (BGE cross-encoder locally) ---
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() == "true"

# --- LLM (Ollama + Qwen2.5 locally / Gemini in production) ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

# --- Chunking ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))

# --- Retrieval ---
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "8"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "3"))

# --- MCP ---
MCP_SERVER_NAME = "enterprise-ai-platform"

# --- API ---
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
