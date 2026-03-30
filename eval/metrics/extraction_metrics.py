"""
extraction_metrics.py
Feature-level extraction accuracy, sensitivity, precision, and F1.
Aligned against human annotator labels (1=Correct, 2=Omission, 3=Fabrication).
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


HUMAN_LABEL_DECODE = {1: "CORRECT", 2: "OMISSION", 3: "FABRICATION"}
VERDICT_CLASSES = ["CORRECT", "FABRICATION", "OMISSION", "UNCERTAIN"]


def decode_human_labels(df: pd.DataFrame, label_col: str = "human_label") -> pd.DataFrame:
    df = df.copy()
    df["human_label_str"] = df[label_col].map(HUMAN_LABEL_DECODE).fillna("UNCERTAIN")
    return df


def compute_overall_metrics(
    df: pd.DataFrame,
    pred_col: str = "verdict",
    true_col: str = "human_label_str",
) -> Dict[str, Any]:
    df = df.dropna(subset=[pred_col, true_col])
    if df.empty:
        return {}

    y_true = df[true_col].tolist()
    y_pred = df[pred_col].tolist()

    labels = [c for c in VERDICT_CLASSES if c in set(y_true + y_pred)]

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "precision_macro": precision_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        ),
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, zero_division=0
        ),
        "n": len(df),
    }


def compute_feature_metrics(
    df: pd.DataFrame,
    feature_col: str = "feature_name",
    pred_col: str = "verdict",
    true_col: str = "human_label_str",
) -> pd.DataFrame:
    rows = []
    for feat, group in df.groupby(feature_col):
        group = group.dropna(subset=[pred_col, true_col])
        if group.empty:
            continue
        y_true = group[true_col].tolist()
        y_pred = group[pred_col].tolist()
        acc = accuracy_score(y_true, y_pred)
        rows.append({
            "feature_name": feat,
            "n": len(group),
            "accuracy": acc,
            "fabrication_count": (group[pred_col] == "FABRICATION").sum(),
            "omission_count": (group[pred_col] == "OMISSION").sum(),
            "correct_count": (group[pred_col] == "CORRECT").sum(),
        })
    return pd.DataFrame(rows).sort_values("accuracy").reset_index(drop=True)


# =============================================================================
# MODULE SUMMARY
# =============================================================================
# File   : eval/metrics/extraction_metrics.py
# Purpose: Feature-level extraction accuracy metrics comparing pipeline
#          verdicts against human annotator labels.
#
# Functions:
#   decode_human_labels(series) -> pd.Series
#     Maps integer human labels (1/2/3) to verdict strings
#     (CORRECT/OMISSION/FABRICATION).
#
#   compute_overall_metrics(df, true_col, pred_col) -> dict
#     Returns overall accuracy, macro F1, precision, recall, and
#     a full sklearn classification_report string.
#
#   compute_feature_metrics(df, feature_col, true_col,
#                           pred_col) -> pd.DataFrame
#     Per-feature accuracy, fabrication_count, omission_count,
#     correct_count. Sorted ascending by accuracy.
#
# Consumed by:
#   eval/metrics/hcat_metrics.py
#   fabrication_analysis/03_fabrication_omission_pipeline.ipynb
# =============================================================================
