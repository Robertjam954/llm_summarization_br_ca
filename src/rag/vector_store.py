"""
vector_store.py
Per-case vector store manager.
Builds, caches, and retrieves FAISS indexes for each case.
"""

from __future__ import annotations

from typing import Dict

from langchain_community.vectorstores import FAISS

from src.rag.embed_chunks import build_faiss_index, get_embedder
from src.workflows.extraction_state import Chunk

_CASE_INDEX_CACHE: Dict[str, FAISS] = {}


def get_or_build_index(
    case_id: str,
    chunks: list[Chunk],
    force_rebuild: bool = False,
) -> FAISS:
    if case_id in _CASE_INDEX_CACHE and not force_rebuild:
        return _CASE_INDEX_CACHE[case_id]
    embedder = get_embedder()
    index = build_faiss_index(chunks, embedder)
    _CASE_INDEX_CACHE[case_id] = index
    return index


def clear_index(case_id: str) -> None:
    _CASE_INDEX_CACHE.pop(case_id, None)


def clear_all_indexes() -> None:
    _CASE_INDEX_CACHE.clear()
