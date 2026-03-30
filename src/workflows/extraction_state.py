"""
extraction_state.py
Core TypedDict schemas for the LangGraph extraction + verification pipeline.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, TypedDict


class Chunk(TypedDict):
    chunk_id: str
    case_id: str
    document_id: str
    page_num: int
    modality: Literal["mammogram", "ultrasound", "mri", "pathology", "other"]
    text: str
    token_count: int
    embedding_model: str
    retrieval_score: Optional[float]


class FeatureResult(TypedDict):
    feature_name: str
    value: str
    evidence: str
    page_refs: List[int]
    confidence: float
    reasoning_for_confidence: str
    supported: Optional[bool]
    verification_quote: Optional[str]
    verification_confidence: Optional[float]
    verdict: Optional[Literal["CORRECT", "FABRICATION", "OMISSION", "UNCERTAIN"]]
    corrected_value: Optional[str]
    retrieval_attempts: int
    verification_method: Optional[str]


class ExtractionState(TypedDict):
    run_id: str
    case_id: str
    prompt_id: str
    model_id: str
    ocr_text: str
    chunk_index_ready: bool
    feature_queue: List[str]
    current_feature: Optional[str]
    current_query: Optional[str]
    retrieved_chunks: List[Chunk]
    extracted_elements: Dict[str, FeatureResult]
    fabrication_flags: List[str]
    omission_flags: List[str]
    high_risk_features: List[str]
    metadata: Dict


def make_empty_feature_result(feature_name: str) -> FeatureResult:
    return FeatureResult(
        feature_name=feature_name,
        value="Not reported",
        evidence="",
        page_refs=[],
        confidence=0.0,
        reasoning_for_confidence="",
        supported=None,
        verification_quote=None,
        verification_confidence=None,
        verdict=None,
        corrected_value=None,
        retrieval_attempts=0,
        verification_method=None,
    )


def make_initial_state(
    run_id: str,
    case_id: str,
    ocr_text: str,
    prompt_id: str = "default",
    model_id: str = "claude-sonnet",
    feature_queue: Optional[List[str]] = None,
    high_risk_features: Optional[List[str]] = None,
) -> ExtractionState:
    from src.rag.feature_queries import FEATURES, CRITICAL_FEATURES

    return ExtractionState(
        run_id=run_id,
        case_id=case_id,
        prompt_id=prompt_id,
        model_id=model_id,
        ocr_text=ocr_text,
        chunk_index_ready=False,
        feature_queue=feature_queue if feature_queue is not None else list(FEATURES.keys()),
        current_feature=None,
        current_query=None,
        retrieved_chunks=[],
        extracted_elements={},
        fabrication_flags=[],
        omission_flags=[],
        high_risk_features=high_risk_features if high_risk_features is not None else list(CRITICAL_FEATURES),
        metadata={"case_id": case_id, "run_id": run_id},
    )


# =============================================================================
# MODULE SUMMARY
# =============================================================================
# File   : src/workflows/extraction_state.py
# Purpose: Core TypedDict schemas and factory functions for all state flowing
#          through the LangGraph StateGraph pipeline.
#
# Types:
#   Chunk           - Single text window from OCR chunking. Fields: text,
#                     chunk_id, modality, page_num, token_count,
#                     retrieval_score.
#   FeatureResult   - Per-feature extraction record. Fields: value, evidence,
#                     page_refs, confidence, reasoning_for_confidence,
#                     supported, verification_quote, verification_confidence,
#                     verdict, retrieval_attempts, verification_method.
#   ExtractionState - Complete LangGraph pipeline state. Fields: run_id,
#                     case_id, prompt_id, model_id, ocr_text,
#                     chunk_index_ready, feature_queue, current_feature,
#                     current_query, retrieved_chunks, extracted_elements,
#                     fabrication_flags, omission_flags, high_risk_features,
#                     metadata.
#
# Functions:
#   make_empty_feature_result() -> FeatureResult
#     Returns a FeatureResult with all fields initialised to safe defaults.
#
#   make_initial_state(case_id, ocr_text, prompt_id, model_id, run_id,
#                      feature_queue, high_risk_features) -> ExtractionState
#     Constructs the starting ExtractionState for a new pipeline run.
#
# Consumed by: extraction_graph.py, orchestration.py, all src/agents/*.py
# =============================================================================
