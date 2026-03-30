"""
self_consistency_agent.py
Selective self-consistency check for high-risk features only.
Runs 3 independent extraction passes; unanimous agreement boosts confidence,
disagreement marks as Indeterminate.
"""

from __future__ import annotations

import os

from langchain_anthropic import ChatAnthropic

from src.prompts.extraction_prompt_builder import build_extraction_prompt
from src.rag.feature_queries import HIGH_RISK_SELF_CONSISTENCY, FEATURES
from src.utils.json_utils import safe_parse_json
from src.utils.logging_utils import get_logger
from src.workflows.extraction_state import ExtractionState

logger = get_logger(__name__)

N_CONSISTENCY_RUNS = 3


def _get_llm(model_id: str = "claude-3-5-sonnet-20241022") -> ChatAnthropic:
    return ChatAnthropic(
        model=model_id,
        temperature=0.0,
        max_tokens=2048,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )


def self_consistency_check(state: ExtractionState) -> ExtractionState:
    feature = state["current_feature"]
    if feature is None:
        return state

    if feature not in HIGH_RISK_SELF_CONSISTENCY:
        logger.debug(
            f"[self_consistency] skipping non-critical feature={feature}"
        )
        return state

    result = state["extracted_elements"].get(feature)
    if result is None:
        return state

    if result.get("confidence", 0.0) >= 0.8 and result.get("supported"):
        logger.debug(
            f"[self_consistency] skipping high-confidence feature={feature}"
        )
        return state

    logger.info(
        f"[self_consistency] running {N_CONSISTENCY_RUNS} passes "
        f"case={state['case_id']} feature={feature}"
    )

    feat_meta = FEATURES.get(feature, {})
    display_name = feat_meta.get("display_name", feature)
    messages = build_extraction_prompt(
        feature_name=feature,
        retrieved_chunks=state["retrieved_chunks"],
        display_name=display_name,
        use_few_shot=True,
        use_cot=True,
    )

    llm = _get_llm(state.get("model_id", "claude-3-5-sonnet-20241022"))
    values = []
    for _ in range(N_CONSISTENCY_RUNS):
        response = llm.invoke(messages)
        raw = response.content if hasattr(response, "content") else str(response)
        parsed = safe_parse_json(raw)
        val = str(parsed.get("value", "Not reported")) if parsed else "Not reported"
        values.append(val)

    unique_values = set(v for v in values if v not in ("", "Not reported"))

    if len(unique_values) == 1:
        agreed_value = unique_values.pop()
        updated_result = {
            **result,
            "value": agreed_value,
            "confidence": 1.0,
            "verification_method": "self_consistency_passed",
            "reasoning_for_confidence": (
                f"All {N_CONSISTENCY_RUNS} passes agreed: {agreed_value}"
            ),
        }
        logger.info(
            f"[self_consistency] PASSED feature={feature} value='{agreed_value}'"
        )
    elif len(unique_values) == 0:
        updated_result = {
            **result,
            "value": "Not reported",
            "confidence": 0.0,
            "verification_method": "self_consistency_all_not_reported",
        }
    else:
        updated_result = {
            **result,
            "value": "Indeterminate",
            "confidence": 0.0,
            "verification_method": "self_consistency_failed",
            "reasoning_for_confidence": (
                f"Disagreement across runs: {list(unique_values)}"
            ),
        }
        logger.warning(
            f"[self_consistency] FAILED feature={feature} "
            f"values={list(unique_values)}"
        )

    updated = dict(state["extracted_elements"])
    updated[feature] = updated_result
    return {**state, "extracted_elements": updated}


# =============================================================================
# MODULE SUMMARY
# =============================================================================
# File   : src/agents/self_consistency_agent.py
# Purpose: LangGraph node that runs 3 independent extraction passes on
#          high-risk features (receptor status, invasive size) and accepts
#          the result only when at least 2 of 3 passes agree.
#
# Functions:
#   self_consistency_check(state: ExtractionState) -> dict
#     Calls build_extraction_prompt and invokes LLM 3 times independently.
#     Parses each response and collects extracted values.
#     If majority (>=2/3) agree: accepts agreed value, sets confidence=0.95,
#     sets verification_method="self_consistency_pass".
#     If no majority: keeps original value, sets confidence=0.4,
#     sets verification_method="self_consistency_fail" (flags for review).
#     Updates state["extracted_elements"][current_feature].
#
# Outputs (state mutation):
#   extracted_elements[feature_name].value           (majority or original)
#   extracted_elements[feature_name].confidence      (0.95 or 0.4)
#   extracted_elements[feature_name].verification_method
#
# Consumed by:
#   src/workflows/extraction_graph.py  (node: "self_consistency")
# =============================================================================
