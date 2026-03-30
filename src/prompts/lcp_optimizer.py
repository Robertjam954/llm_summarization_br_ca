"""
lcp_optimizer.py
Lightweight Contrastive Prompt (LCP) optimizer.
Runs an outer optimization loop over the prompt library:
  1. Score each prompt variant on a dev set
  2. Contrast good vs bad prompts by failure mode
  3. Generate candidate revisions
  4. Re-evaluate and promote best performers
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pandas as pd

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def score_prompt_variant(
    results_df: pd.DataFrame,
    prompt_id: str,
    feature_name: Optional[str] = None,
) -> Dict[str, float]:
    mask = results_df["prompt_id"] == prompt_id
    if feature_name:
        mask &= results_df["feature_name"] == feature_name
    subset = results_df[mask]
    if subset.empty:
        return {"fabrication_rate": 1.0, "omission_rate": 1.0, "accuracy": 0.0}

    total = len(subset)
    fabrications = (subset["verdict"] == "FABRICATION").sum()
    omissions = (subset["verdict"] == "OMISSION").sum()
    correct = (subset["verdict"] == "CORRECT").sum()

    return {
        "prompt_id": prompt_id,
        "feature_name": feature_name or "all",
        "n": total,
        "fabrication_rate": fabrications / total,
        "omission_rate": omissions / total,
        "accuracy": correct / total,
        "verification_pass_rate": (
            subset["supported"].fillna(False).mean()
            if "supported" in subset.columns
            else 0.0
        ),
    }


def rank_prompt_variants(
    results_df: pd.DataFrame,
    prompt_ids: List[str],
    feature_name: Optional[str] = None,
) -> pd.DataFrame:
    scores = [
        score_prompt_variant(results_df, pid, feature_name)
        for pid in prompt_ids
    ]
    df = pd.DataFrame(scores)
    return df.sort_values("accuracy", ascending=False).reset_index(drop=True)


def build_contrastive_summary(
    top_prompts: List[Dict[str, Any]],
    bottom_prompts: List[Dict[str, Any]],
) -> str:
    top_str = json.dumps(top_prompts, indent=2, default=str)
    bottom_str = json.dumps(bottom_prompts, indent=2, default=str)
    return (
        f"TOP PERFORMING PROMPTS:\n{top_str}\n\n"
        f"BOTTOM PERFORMING PROMPTS:\n{bottom_str}"
    )


def generate_candidate_revision_prompt(
    feature_name: str,
    contrastive_summary: str,
    failure_examples: List[Dict[str, Any]],
) -> str:
    examples_str = json.dumps(failure_examples, indent=2, default=str)
    return f"""You are a prompt optimization system.

Feature being extracted: {feature_name}

CONTRASTIVE PROMPT PERFORMANCE SUMMARY:
{contrastive_summary}

FAILURE EXAMPLES (fabrications and omissions):
{examples_str}

Task:
Generate 3 improved prompt variants for extracting {feature_name}.
Focus on reducing fabrication rate while maintaining recall.
Return JSON list: [{{"prompt_id": "...", "system": "...", "instruction": "..."}}]"""


# =============================================================================
# MODULE SUMMARY
# =============================================================================
# File   : src/prompts/lcp_optimizer.py
# Purpose: Lightweight Contrastive Prompt (LCP) optimizer. Scores prompt
#          variants by fabrication and recall metrics, ranks them, and
#          generates LLM-assisted revision candidates.
#
# Functions:
#   score_prompt_variant(results_df, prompt_id_col, verdict_col,
#                        prompt_id) -> dict
#     Computes fabrication_rate, omission_rate, accuracy, and composite
#     score for a single prompt variant over a results DataFrame.
#
#   rank_prompt_variants(results_df, prompt_id_col, verdict_col) -> DataFrame
#     Scores all prompt variants and returns a ranked DataFrame.
#     Columns: prompt_id, fabrication_rate, omission_rate, accuracy, score.
#
#   build_contrastive_summary(best_df, worst_df, feature_name) -> str
#     Formats a human-readable contrastive summary of best vs worst
#     prompt examples for a given feature. Used as LLM revision context.
#
#   generate_candidate_revision_prompt(feature_name, success_examples,
#                                      failure_examples) -> str
#     Returns a meta-prompt string asking the LLM to generate 3 improved
#     prompt variants targeting lower fabrication for a feature.
#
# Outputs:
#   Ranked DataFrame; contrastive summary string; meta-prompt string.
#
# Consumed by:
#   Prompt engineering notebooks; LCP tuning workflows.
# =============================================================================
