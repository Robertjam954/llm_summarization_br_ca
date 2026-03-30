"""
retrievers.py
Feature-specific retrieval from the per-case FAISS index.
Returns typed Chunk objects with retrieval scores populated.
"""

from typing import List

from langchain_community.vectorstores import FAISS

from src.rag.feature_queries import (
    FEATURES,
    RETRIEVAL_K_MAP,
    RETRIEVAL_K_SECOND_PASS_MAP,
)
from src.workflows.extraction_state import Chunk


def retrieve_for_feature(
    index: FAISS,
    case_id: str,
    feature_name: str,
    query: str,
    k: int,
) -> List[Chunk]:
    results = index.similarity_search_with_score(query, k=k)
    chunks: List[Chunk] = []
    for doc, score in results:
        meta = doc.metadata
        chunk = Chunk(
            chunk_id=meta.get("chunk_id", ""),
            case_id=meta.get("case_id", case_id),
            document_id=meta.get("document_id", case_id),
            page_num=meta.get("page_num", 0),
            modality=meta.get("modality", "other"),
            text=doc.page_content,
            token_count=meta.get("token_count", len(doc.page_content.split())),
            embedding_model=meta.get("embedding_model", ""),
            retrieval_score=float(score),
        )
        chunks.append(chunk)
    return chunks


def get_feature_k(feature_name: str, second_pass: bool = False) -> int:
    if second_pass:
        return RETRIEVAL_K_SECOND_PASS_MAP.get(feature_name, 7)
    return RETRIEVAL_K_MAP.get(feature_name, 3)


def get_feature_query(feature_name: str) -> str:
    feat = FEATURES.get(feature_name, {})
    return feat.get("query", feature_name)


def format_chunks_for_prompt(chunks: List[Chunk]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        score = chunk.get("retrieval_score") or 0.0
        parts.append(
            f"[Chunk {i} | page ~{chunk['page_num']} | "
            f"modality={chunk['modality']} | score={score:.3f}]\n"
            f"{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)


# =============================================================================
# MODULE SUMMARY
# =============================================================================
# File   : src/rag/retrievers.py
# Purpose: Feature-specific dense retrieval from the FAISS vector store,
#          returning typed Chunk objects with retrieval scores.
#
# Functions:
#   get_feature_query(feature_name) -> str
#     Returns the retrieval query string for a named feature.
#
#   get_feature_k(feature_name, second_pass) -> int
#     Returns the retrieval depth k (or k_second_pass) for a feature.
#
#   retrieve_for_feature(index, case_id, feature_name, query, k) -> List[Chunk]
#     Runs similarity search on the FAISS index for the given query.
#     Populates 'retrieval_score' field on each returned Chunk.
#     Returns up to k Chunk TypedDicts.
#
#   format_chunks_for_prompt(chunks) -> str
#     Formats a list of Chunks into a numbered text block for LLM prompts.
#     Output: Human-readable string with SOURCE, modality, score header per
#     chunk separated by horizontal rules.
#
# Consumed by:
#   src/workflows/extraction_graph.py (node_retrieve, node_retrieve_again)
#   fabrication_analysis/01_langgraph_extraction_pipeline.ipynb
# =============================================================================
