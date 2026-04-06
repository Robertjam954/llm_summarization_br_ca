"""
embed_chunks.py
Embed chunked text using sentence-transformers and store embeddings.
Supports FAISS and in-memory vector stores via LangChain abstractions.
"""

from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from src.workflows.extraction_state import Chunk

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedder(model_name: str = DEFAULT_EMBEDDING_MODEL) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def chunks_to_documents(chunks: List[Chunk]) -> List[Document]:
    docs = []
    for chunk in chunks:
        doc = Document(
            page_content=chunk["text"],
            metadata={
                "chunk_id": chunk["chunk_id"],
                "case_id": chunk["case_id"],
                "document_id": chunk["document_id"],
                "page_num": chunk["page_num"],
                "modality": chunk["modality"],
                "token_count": chunk["token_count"],
            },
        )
        docs.append(doc)
    return docs


def build_faiss_index(
    chunks: List[Chunk],
    embedder: Optional[HuggingFaceEmbeddings] = None,
) -> FAISS:
    if embedder is None:
        embedder = get_embedder()
    docs = chunks_to_documents(chunks)
    return FAISS.from_documents(docs, embedder)


def save_faiss_index(index: FAISS, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    index.save_local(str(path))


def load_faiss_index(
    path: Path,
    embedder: Optional[HuggingFaceEmbeddings] = None,
) -> FAISS:
    if embedder is None:
        embedder = get_embedder()
    return FAISS.load_local(
        str(path),
        embedder,
        allow_dangerous_deserialization=True,
    )


# =============================================================================
# MODULE SUMMARY
# =============================================================================
# File   : src/rag/embed_chunks.py
# Purpose: Embeds chunked OCR text using sentence-transformers via
#          HuggingFaceEmbeddings and builds / saves / loads FAISS indexes.
#
# Functions:
#   get_embedder(model_name) -> HuggingFaceEmbeddings
#     Returns a cached HuggingFaceEmbeddings instance.
#     Default model: sentence-transformers/all-mpnet-base-v2.
#
#   build_faiss_index(chunks, embedder) -> FAISS
#     Embeds a list of Chunk dicts and returns a FAISS index.
#     Input: List[Chunk] with 'text' field populated.
#     Output: LangChain FAISS VectorStore object.
#
#   save_faiss_index(index, path) -> None
#     Persists a FAISS index to disk at the given Path.
#
#   load_faiss_index(path, embedder) -> FAISS
#     Loads a previously saved FAISS index from disk.
#
# Consumed by:
#   src/rag/vector_store.py        (builds and caches indexes)
#   src/workflows/extraction_graph.py (node_index_case)
#   fabrication_analysis/01_langgraph_extraction_pipeline.ipynb
# =============================================================================
