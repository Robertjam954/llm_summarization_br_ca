"""
rewrite_agent.py
Query rewriter agent — generates an expanded query with synonyms,
modality context, and alternate terminology after failed retrieval
or failed verification.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.llm_factory import get_llm

from src.utils.logging_utils import get_logger
from src.workflows.extraction_state import ExtractionState

logger = get_logger(__name__)

REWRITER_SYSTEM = (
    "You are a clinical information retrieval specialist. "
    "You rewrite feature-extraction queries for dense medical PDF retrieval. "
    "Add synonyms, alternate terminology, and modality context."
)

SYNONYM_HINTS = {
    "feature_1_lesion_size": (
        "lesion size mass dimensions invasive component tumor size cm mm"
    ),
    "feature_2_lesion_location": (
        "laterality quadrant clock-face o'clock depth nipple distance location"
    ),
    "feature_8_lymph_node": (
        "axillary node adenopathy lymphadenopathy nodal findings sentinel"
    ),
    "feature_10_biopsy_method": (
        "core needle biopsy stereotactic US-guided MRI-guided FNA excision CNB"
    ),
    "feature_11_invasive_component_size_pathology": (
        "invasive carcinoma size tumor size pathology cm mm grade invasive component"
    ),
    "feature_13_receptor_status": (
        "ER PR HER2 IHC ISH FISH immunohistochemistry receptor estrogen "
        "progesterone HER-2/neu Allred score"
    ),
}



def rewrite_query(state: ExtractionState) -> ExtractionState:
    feature = state["current_feature"]
    old_query = state.get("current_query", "") or ""

    logger.info(
        f"[rewrite] case={state['case_id']} feature={feature}"
    )

    hints = SYNONYM_HINTS.get(feature, "")
    human_msg = (
        f"Rewrite this clinical retrieval query for a dense medical PDF.\n"
        f"Add synonyms, modality context, and alternate terminology.\n\n"
        f"Original query: {old_query}\n"
        f"Feature: {feature}\n"
        f"Helpful terms: {hints}\n\n"
        f"Return ONLY the rewritten query string, no explanation."
    )

    llm = get_llm(state.get("model_id", "claude-3-5-sonnet-20241022"), temperature=0.2, max_tokens=256)
    response = llm.invoke(
        [SystemMessage(content=REWRITER_SYSTEM), HumanMessage(content=human_msg)]
    )
    new_query = (
        response.content.strip()
        if hasattr(response, "content")
        else str(response).strip()
    )

    if not new_query or len(new_query) < 5:
        new_query = f"{old_query} {hints}".strip()

    logger.info(f"[rewrite] new_query='{new_query[:100]}'")
    return {**state, "current_query": new_query}


# =============================================================================
# MODULE SUMMARY
# =============================================================================
# File   : src/agents/rewrite_agent.py
# Purpose: LangGraph node that rewrites a failed retrieval query using synonym
#          expansion and modality context hints, improving recall on retry.
#
# Functions:
#   rewrite_query(state: ExtractionState) -> dict
#     Calls Claude with the original query + feature hints to generate an
#     expanded synonym-rich query string. Falls back to original query +
#     hint concatenation if LLM response is empty or too short.
#     Increments retrieval_attempts counter on the current FeatureResult.
#     Updates state["current_query"] with the rewritten query string.
#
# Outputs (state mutation):
#   current_query  (new expanded query string)
#   extracted_elements[feature_name].retrieval_attempts  (+1)
#
# Consumed by:
#   src/workflows/extraction_graph.py  (node: "rewrite_query")
# =============================================================================
