"""
adjudicate_agent.py
Assigns CORRECT / FABRICATION / OMISSION / UNCERTAIN verdict
to each extracted feature result based on verification outcome.
"""

from __future__ import annotations

from src.rag.feature_queries import VERIFICATION_THRESHOLD_MAP
from src.utils.logging_utils import get_logger, log_verdict
from src.workflows.extraction_state import ExtractionState

logger = get_logger(__name__)

NOT_REPORTED_VALUES = {"Not reported", "not reported", ""}
INDETERMINATE_VALUES = {"Indeterminate", "indeterminate"}


def adjudicate_result(state: ExtractionState) -> ExtractionState:
    feature = state["current_feature"]
    if feature is None:
        return state

    result = state["extracted_elements"].get(feature)
    if result is None:
        return state

    value = result["value"]
    supported = result.get("supported")
    confidence = result.get("confidence", 0.0) or 0.0
    threshold = VERIFICATION_THRESHOLD_MAP.get(feature, 0.75)
    fabrication_flags = list(state["fabrication_flags"])
    omission_flags = list(state["omission_flags"])

    if value in NOT_REPORTED_VALUES:
        verdict = "OMISSION"
        if feature not in omission_flags:
            omission_flags.append(feature)
    elif value in INDETERMINATE_VALUES:
        verdict = "UNCERTAIN"
    elif supported is True and confidence >= threshold:
        verdict = "CORRECT"
    elif supported is False and value not in NOT_REPORTED_VALUES:
        verdict = "FABRICATION"
        if feature not in fabrication_flags:
            fabrication_flags.append(feature)
    elif confidence < threshold and value not in NOT_REPORTED_VALUES:
        verdict = "UNCERTAIN"
    else:
        verdict = "UNCERTAIN"

    log_verdict(
        logger,
        state["run_id"],
        state["case_id"],
        feature,
        verdict,
        confidence,
        result.get("retrieval_attempts", 0),
    )

    updated_result = {**result, "verdict": verdict}
    updated = dict(state["extracted_elements"])
    updated[feature] = updated_result

    return {
        **state,
        "extracted_elements": updated,
        "fabrication_flags": fabrication_flags,
        "omission_flags": omission_flags,
    }


# =============================================================================
# MODULE SUMMARY
# =============================================================================
# File   : src/agents/adjudicate_agent.py
# Purpose: LangGraph node that assigns a final CORRECT / FABRICATION /
#          OMISSION / UNCERTAIN verdict to each extracted feature result,
#          using verification confidence, support flag, and feature thresholds.
#
# Functions:
#   adjudicate_result(state: ExtractionState) -> dict
#     Reads the current feature's FeatureResult. Applies adjudication rules:
#       - supported=True + confidence >= threshold  -> CORRECT
#       - supported=False + value not None          -> FABRICATION
#       - value is None or "Not reported"           -> OMISSION
#       - confidence < threshold (uncertain)        -> UNCERTAIN
#     Appends to fabrication_flags or omission_flags lists as appropriate.
#     Updates state["extracted_elements"][feature].verdict.
#
# Outputs (state mutation):
#   extracted_elements[feature_name].verdict
#   fabrication_flags  (list of feature names with FABRICATION verdict)
#   omission_flags     (list of feature names with OMISSION verdict)
#
# Consumed by:
#   src/workflows/extraction_graph.py  (node: "adjudicate")
# =============================================================================
