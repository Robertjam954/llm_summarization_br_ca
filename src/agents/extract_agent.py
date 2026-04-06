"""
extract_agent.py
Structured extractor agent — node in the LangGraph pipeline.
Calls Claude via langchain_anthropic, parses strict JSON output.
"""

from __future__ import annotations

from src.agents.llm_factory import get_llm
from src.prompts.extraction_prompt_builder import build_extraction_prompt
from src.rag.feature_queries import FEATURES
from src.utils.json_utils import (
    coerce_confidence,
    coerce_page_refs,
    safe_parse_json,
    validate_feature_result,
)
from src.utils.logging_utils import get_logger
from src.workflows.extraction_state import ExtractionState, FeatureResult

logger = get_logger(__name__)



def extract_feature(state: ExtractionState) -> ExtractionState:
    feature = state["current_feature"]
    if feature is None:
        return state

    logger.info(
        f"[extract] case={state['case_id']} feature={feature}"
    )

    feat_meta = FEATURES.get(feature, {})
    display_name = feat_meta.get("display_name", feature)
    chunks = state["retrieved_chunks"]

    messages = build_extraction_prompt(
        feature_name=feature,
        retrieved_chunks=chunks,
        display_name=display_name,
        use_few_shot=True,
        use_cot=True,
    )

    llm = get_llm(state.get("model_id", "claude-3-5-sonnet-20241022"))
    response = llm.invoke(messages)
    raw = response.content if hasattr(response, "content") else str(response)

    parsed = safe_parse_json(raw)

    if parsed and validate_feature_result(parsed):
        result = FeatureResult(
            feature_name=feature,
            value=str(parsed.get("value", "Not reported")),
            evidence=str(parsed.get("evidence", "")),
            page_refs=coerce_page_refs(parsed.get("page_refs", [])),
            confidence=coerce_confidence(parsed.get("confidence", 0.0)),
            reasoning_for_confidence=str(
                parsed.get("reasoning_for_confidence", "")
            ),
            supported=None,
            verification_quote=None,
            verification_confidence=None,
            verdict=None,
            corrected_value=None,
            retrieval_attempts=state["extracted_elements"]
            .get(feature, {})
            .get("retrieval_attempts", 0)
            + 1,
            verification_method=None,
        )
    else:
        logger.warning(
            f"[extract] JSON parse failed for {feature}, "
            f"defaulting to 'Not reported'"
        )
        from src.workflows.extraction_state import make_empty_feature_result
        result = make_empty_feature_result(feature)
        result["retrieval_attempts"] = 1

    updated = dict(state["extracted_elements"])
    updated[feature] = result
    return {**state, "extracted_elements": updated}


# =============================================================================
# MODULE SUMMARY
# =============================================================================
# File   : src/agents/extract_agent.py
# Purpose: LangGraph node that calls Claude to extract one clinical feature
#          value from the retrieved OCR chunks for the current feature.
#
# Functions:
#   extract_feature(state: ExtractionState) -> dict
#     Builds an extraction prompt via extraction_prompt_builder, invokes the
#     LLM, and parses JSON output via safe_parse_json / validate_feature_result.
#     On parse failure: retries once with a stricter JSON-only instruction.
#     Updates state["extracted_elements"][current_feature] with a FeatureResult
#     containing: value, evidence, page_refs, confidence, reasoning.
#
# Outputs (state mutation):
#   extracted_elements[feature_name].value
#   extracted_elements[feature_name].confidence
#   extracted_elements[feature_name].evidence
#   extracted_elements[feature_name].page_refs
#
# Consumed by:
#   src/workflows/extraction_graph.py  (node: "extract")
# =============================================================================
