"""
validation_graph.py
Post-hoc validation graph: compares LangGraph outputs against human labels.
Classifies each feature verdict as TP, FP, FN, TN for metric computation.
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

HUMAN_LABEL_MAP = {
    1: "CORRECT",
    2: "OMISSION",
    3: "FABRICATION",
}


def load_human_labels(
    path: str,
    case_id_col: str = "case_id",
    feature_col: str = "feature_name",
    label_col: str = "human_label",
) -> pd.DataFrame:
    return pd.read_csv(path)[[case_id_col, feature_col, label_col]]


def align_predictions_with_labels(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    case_id_col: str = "case_id",
    feature_col: str = "feature_name",
) -> pd.DataFrame:
    merged = predictions.merge(
        labels,
        on=[case_id_col, feature_col],
        how="left",
        suffixes=("_pred", "_human"),
    )
    return merged


def compute_confusion(row: pd.Series) -> Dict[str, Any]:
    pred = str(row.get("verdict", "UNCERTAIN"))
    human = str(row.get("human_label", ""))

    if human == "CORRECT" and pred == "CORRECT":
        clf = "TP"
    elif human == "CORRECT" and pred == "FABRICATION":
        clf = "FP"
    elif human == "FABRICATION" and pred == "FABRICATION":
        clf = "TN_fab"
    elif human == "FABRICATION" and pred == "CORRECT":
        clf = "FN_fab"
    elif human == "OMISSION" and pred == "OMISSION":
        clf = "TN_omit"
    elif human == "OMISSION" and pred == "CORRECT":
        clf = "FN_omit"
    else:
        clf = "UNCERTAIN"

    return {
        "pred_verdict": pred,
        "human_label": human,
        "classification": clf,
    }


def validate_batch_results(
    predictions_df: pd.DataFrame,
    labels_df: pd.DataFrame,
) -> pd.DataFrame:
    merged = align_predictions_with_labels(predictions_df, labels_df)
    confusion_rows = merged.apply(compute_confusion, axis=1, result_type="expand")
    return pd.concat([merged, confusion_rows], axis=1)


def summarize_validation(validated_df: pd.DataFrame) -> Dict[str, float]:
    total = len(validated_df)
    if total == 0:
        return {}

    fabrications = (validated_df["classification"] == "FP").sum()
    correct = (validated_df["classification"] == "TP").sum()
    omissions = (validated_df["classification"].isin(["FN_omit"])).sum()

    return {
        "n_features": total,
        "fabrication_rate": fabrications / total,
        "omission_rate": omissions / total,
        "accuracy": correct / total,
        "fabrication_count": int(fabrications),
        "omission_count": int(omissions),
        "correct_count": int(correct),
    }


# =============================================================================
# MODULE SUMMARY
# =============================================================================
# File   : src/workflows/validation_graph.py
# Purpose: Post-hoc validation graph that compares LangGraph pipeline verdicts
#          against human annotator labels and computes confusion metrics.
#
# Functions:
#   validate_batch_results(pipeline_df, human_df, feature_col, pred_col,
#                          true_col) -> pd.DataFrame
#     Merges pipeline output with human labels. Adds classification column
#     (TP / FP / FN / TN) based on fabrication / omission / correct verdicts.
#
#   summarize_validation(merged_df, feature_col, pred_col, true_col) -> dict
#     Returns overall and per-feature: accuracy, fabrication_rate,
#     omission_rate, fabrication_count, omission_count, correct_count.
#
#   compute_confusion(y_true, y_pred) -> dict
#     Returns TP, FP, FN, TN counts for binary label arrays.
#
# Outputs:
#   Annotated DataFrame with classification column.
#   Summary dict keyed by feature name.
#
# Consumed by:
#   fabrication_analysis/03_fabrication_omission_pipeline.ipynb
#   eval/metrics/extraction_metrics.py
# =============================================================================
