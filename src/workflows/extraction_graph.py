"""
extraction_graph.py
Full LangGraph StateGraph for feature-level RAG extraction + verification.

Graph topology:
  load_case → index_case → next_feature → retrieve → extract → verify
    verify → (adjudicate | rewrite_query | self_consistency)
    rewrite_query → retrieve_again → extract → verify
    self_consistency → adjudicate
    adjudicate → (next_feature | aggregate)
    aggregate → END
"""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, StateGraph

from src.agents.adjudicate_agent import adjudicate_result
from src.agents.extract_agent import extract_feature
from src.agents.rewrite_agent import rewrite_query
from src.agents.self_consistency_agent import self_consistency_check
from src.agents.verify_agent import verify_feature
from src.rag.feature_queries import (
    HIGH_RISK_SELF_CONSISTENCY,
    VERIFICATION_THRESHOLD_MAP,
)
from src.rag.retrievers import get_feature_k, get_feature_query, retrieve_for_feature
from src.rag.vector_store import get_or_build_index
from src.utils.logging_utils import get_logger
from src.workflows.extraction_state import ExtractionState
from src.preprocessing.chunk_text import chunk_ocr_text

logger = get_logger(__name__)

_MAX_RETRIEVAL_ATTEMPTS = 2


def node_load_case(state: ExtractionState) -> ExtractionState:
    logger.info(f"[load_case] case={state['case_id']}")
    return state


def node_index_case(state: ExtractionState) -> ExtractionState:
    logger.info(f"[index_case] case={state['case_id']}")
    chunks = chunk_ocr_text(state["ocr_text"], case_id=state["case_id"])
    get_or_build_index(state["case_id"], chunks, force_rebuild=True)
    return {**state, "chunk_index_ready": True}


def node_select_next_feature(state: ExtractionState) -> ExtractionState:
    queue = list(state["feature_queue"])
    if not queue:
        return {**state, "current_feature": None, "current_query": None}
    feature = queue.pop(0)
    query = get_feature_query(feature)
    logger.info(f"[next_feature] feature={feature}")
    return {
        **state,
        "feature_queue": queue,
        "current_feature": feature,
        "current_query": query,
    }


def node_retrieve(state: ExtractionState) -> ExtractionState:
    feature = state["current_feature"]
    if feature is None:
        return state
    query = state.get("current_query") or get_feature_query(feature)
    k = get_feature_k(feature)
    from src.rag.vector_store import _CASE_INDEX_CACHE
    index = _CASE_INDEX_CACHE.get(state["case_id"])
    if index is None:
        logger.warning(f"[retrieve] no index for case={state['case_id']}")
        node_index_case(state)
        index = _CASE_INDEX_CACHE.get(state["case_id"])
    chunks = retrieve_for_feature(index, state["case_id"], feature, query, k)
    logger.info(f"[retrieve] feature={feature} k={k} got={len(chunks)} chunks")
    return {**state, "retrieved_chunks": chunks, "current_query": query}


def node_retrieve_again(state: ExtractionState) -> ExtractionState:
    feature = state["current_feature"]
    if feature is None:
        return state
    query = state.get("current_query") or get_feature_query(feature)
    k = get_feature_k(feature, second_pass=True)
    from src.rag.vector_store import _CASE_INDEX_CACHE
    index = _CASE_INDEX_CACHE.get(state["case_id"])
    if index is None:
        return state
    chunks = retrieve_for_feature(index, state["case_id"], feature, query, k)
    logger.info(f"[retrieve_again] feature={feature} k={k} got={len(chunks)} chunks")
    return {**state, "retrieved_chunks": chunks}


def node_aggregate(state: ExtractionState) -> ExtractionState:
    logger.info(
        f"[aggregate] case={state['case_id']} "
        f"features_done={len(state['extracted_elements'])} "
        f"fabrications={len(state['fabrication_flags'])}"
    )
    return state


def route_after_verify(
    state: ExtractionState,
) -> Literal["adjudicate", "self_consistency", "rewrite_query"]:
    feature = state["current_feature"]
    if feature is None:
        return "adjudicate"

    result = state["extracted_elements"].get(feature, {})
    supported = result.get("supported")
    confidence = result.get("confidence", 0.0) or 0.0
    threshold = VERIFICATION_THRESHOLD_MAP.get(feature, 0.75)
    attempts = result.get("retrieval_attempts", 0)

    if supported is True and confidence >= threshold:
        return "adjudicate"

    if feature in HIGH_RISK_SELF_CONSISTENCY and confidence < 0.6:
        return "self_consistency"

    if attempts < _MAX_RETRIEVAL_ATTEMPTS:
        return "rewrite_query"

    return "adjudicate"


def route_after_adjudication(
    state: ExtractionState,
) -> Literal["next_feature", "aggregate"]:
    if state["feature_queue"]:
        return "next_feature"
    return "aggregate"


def build_extraction_graph() -> StateGraph:
    workflow = StateGraph(ExtractionState)

    workflow.add_node("load_case", node_load_case)
    workflow.add_node("index_case", node_index_case)
    workflow.add_node("next_feature", node_select_next_feature)
    workflow.add_node("retrieve", node_retrieve)
    workflow.add_node("extract", extract_feature)
    workflow.add_node("verify", verify_feature)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("retrieve_again", node_retrieve_again)
    workflow.add_node("self_consistency", self_consistency_check)
    workflow.add_node("adjudicate", adjudicate_result)
    workflow.add_node("aggregate", node_aggregate)

    workflow.set_entry_point("load_case")
    workflow.add_edge("load_case", "index_case")
    workflow.add_edge("index_case", "next_feature")
    workflow.add_edge("next_feature", "retrieve")
    workflow.add_edge("retrieve", "extract")
    workflow.add_edge("extract", "verify")

    workflow.add_conditional_edges(
        "verify",
        route_after_verify,
        {
            "adjudicate": "adjudicate",
            "self_consistency": "self_consistency",
            "rewrite_query": "rewrite_query",
        },
    )

    workflow.add_edge("rewrite_query", "retrieve_again")
    workflow.add_edge("retrieve_again", "extract")
    workflow.add_edge("self_consistency", "adjudicate")

    workflow.add_conditional_edges(
        "adjudicate",
        route_after_adjudication,
        {
            "next_feature": "next_feature",
            "aggregate": "aggregate",
        },
    )

    workflow.add_edge("aggregate", END)
    return workflow


def compile_graph():
    return build_extraction_graph().compile()


# =============================================================================
# MODULE SUMMARY
# =============================================================================
# File   : src/workflows/extraction_graph.py
# Purpose: Defines the full LangGraph StateGraph for feature-level RAG
#          extraction and verification across all 13 clinical features.
#
# Graph nodes:
#   load_case        - Validates OCR text; stores it in state.
#   index_case       - Chunks text and builds FAISS vector index.
#   next_feature     - Pops next feature from queue; sets current query.
#   retrieve         - Dense retrieval for current feature (k chunks).
#   extract          - Calls Claude to extract feature value (extract_agent).
#   verify           - Calls Claude to verify extraction (verify_agent).
#   rewrite_query    - Expands query with synonyms (rewrite_agent).
#   retrieve_again   - Second retrieval pass after query rewrite.
#   self_consistency - 3-pass consistency check on high-risk features.
#   adjudicate       - Assigns CORRECT/FABRICATION/OMISSION/UNCERTAIN verdict.
#   aggregate        - Collects all FeatureResults into final output dict.
#
# Routing edges:
#   route_after_verify() - Dispatches to adjudicate | self_consistency |
#                          rewrite_query based on support and confidence.
#   route_after_adjudicate() - Loops to next_feature or goes to aggregate.
#
# Functions:
#   build_extraction_graph() -> StateGraph
#     Assembles all nodes and edges; returns an uncompiled StateGraph.
#   compile_graph() -> CompiledGraph
#     Compiles the graph ready for .invoke() or .stream().
#
# Output: Compiled LangGraph app consumed by orchestration.py
# =============================================================================
