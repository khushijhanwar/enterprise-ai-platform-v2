"""
Document loading, using LangChain's Document abstraction so every
downstream component (splitter, vectorstore, retriever) speaks the same
interface LangChain integrations expect -- swapping in a real
DirectoryLoader/PDF loader/S3 loader later is a one-line change here,
not a refactor of the rest of the pipeline.
"""
from __future__ import annotations

import pathlib

from langchain_core.documents import Document

from app.config import SAMPLE_DOCS_DIR


def load_documents(doc_dir: pathlib.Path = SAMPLE_DOCS_DIR) -> list[Document]:
    """Load all .txt documents in a directory as LangChain Documents.

    Production swap: replace this loop with
    `langchain_community.document_loaders.DirectoryLoader` (for a local
    folder), or `PyPDFLoader`/`S3DirectoryLoader`/`GCSDirectoryLoader` for
    real enterprise document sources -- all return the same `Document`
    type, so nothing downstream needs to change.
    """
    docs = []
    for path in sorted(doc_dir.glob("*.txt")):
        text = path.read_text()
        docs.append(
            Document(
                page_content=text,
                metadata={"source": path.name, "path": str(path)},
            )
        )
    return docs
