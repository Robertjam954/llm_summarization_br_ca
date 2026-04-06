"""
verify_agent.py
Retrieval-augmented verification agent — most critical anti-fabrication node.
Checks whether the extracted value is directly supported by source chunks.
"""

from __future__ import annotations

from src.agents.llm_factory import get_llm
from src.prompts.verification_prompt_builder import build_verification_prompt
from src.utils.json_utils import coerce_confidence, safe_parse_json
from src.utils.logging_utils import get_logger, log_fabrication_flag
from src.workflows.extraction_state import ExtractionState

logger = get_logger(__name__)



def verify_feature(state: ExtractionState) -> ExtractionState:
    feature = state["current_feature"]
    if feature is None:
        return state

    result = state["extracted_elements"].get(feature)
    if result is None:
        return state

    value = result["value"]
    if value in ("Not reported", "Indeterminate"):
        updated_result = {
            **result,
            "supported": None,
            "verification_method": "skipped_no_value",
        }
        updated = dict(state["extracted_elements"])
        updated[feature] = updated_result
        return {**state, "extracted_elements": updated}

    logger.info(
        f"[verify] case={state['case_id']} feature={feature} value='{value}'"
    )

    messages = build_verification_prompt(
        feature_name=feature,
        claimed_value=value,
        retrieved_chunks=state["retrieved_chunks"],
    )

    llm = get_llm(state.get("model_id", "claude-3-5-sonnet-20241022"))
    response = llm.invoke(messages)
    raw = response.content if hasattr(response, "content") else str(response)

    parsed = safe_parse_json(raw)
    supported = bool(parsed.get("supported", False)) if parsed else False
    v_confidence = coerce_confidence(
        parsed.get("verification_confidence", 0.0) if parsed else 0.0
    )
    v_quote = str(parsed.get("exact_support_quote", "")) if parsed else ""

    if not supported:
        log_fabrication_flag(
            logger, state["run_id"], state["case_id"], feature, value
        )

    updated_result = {
        **result,
        "supported": supported,
        "verification_quote": v_quote or None,
        "verification_confidence": v_confidence,
        "verification_method": "rag_verification",
        # If NOT_VERIFIABLE, zero out extraction confidence so adjudication
        # treats this as an unsubstantiated claim (reference: confidence=0.0)
        "confidence": v_confidence if not supported else result.get("confidence", 0.0),
    }

    updated = dict(state["extracted_elements"])
    updated[feature] = updated_result
    return {**state, "extracted_elements": updated}


# =============================================================================
# MODULE SUMMARY
# =============================================================================
# File   : src/agents/verify_agent.py
# Purpose: LangGraph node that verifies a claimed extraction value against
#          source chunks. Core anti-fabrication step of the pipeline.
#
# Functions:
#   verify_feature(state: ExtractionState) -> dict
#     Calls build_verification_prompt with the claimed value and retrieved
#     chunks. Parses LLM response for: supported (bool),
#     verification_confidence (float), exact_support_quote (str), page_ref.
#     Skips verification for "Not reported" / None values (avoids false flags).
#     Updates state["extracted_elements"][current_feature] with verification
#     fields: supported, verification_quote, verification_confidence,
#     verification_method.
#
# Outputs (state mutation):
#   extracted_elements[feature_name].supported
#   extracted_elements[feature_name].verification_quote
#   extracted_elements[feature_name].verification_confidence
#   extracted_elements[feature_name].verification_method
#
# Consumed by:
#   src/workflows/extraction_graph.py  (node: "verify")
# =============================================================================
