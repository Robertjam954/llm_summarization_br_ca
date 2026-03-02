"""
metrics_utils.py

Canonical utilities for:
- TP/FP/FN/TN computation
- Diagnostic + classification metrics
- Confidence intervals
- McNemar tests
- ROC/PR utilities
- Plotting
- Table export helpers
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binomtest, norm
from sklearn.metrics import roc_curve, precision_recall_curve, auc


# ============================================================
# Safe saving helpers
# ============================================================


def safe_save_dataframe(df: pd.DataFrame, path: Path) -> None:
    """Save a dataframe to CSV only if parent directory exists."""
    out_dir = path.parent
    if not out_dir.exists():
        print(f"Warning: directory does not exist, skipping CSV save: {path}")
        return
    try:
        df.to_csv(path, index=False)
        print(f"Saved CSV to: {path}")
    except Exception as exc:
        print(f"Failed saving CSV {path}: {exc}")


def safe_save_plt(fig, path: Path, **savefig_kwargs) -> None:
    """Save a matplotlib figure only if parent directory exists."""
    out_dir = path.parent
    if not out_dir.exists():
        print(f"Warning: directory does not exist, skipping figure save: {path}")
        return
    try:
        fig.savefig(path, **savefig_kwargs)
        print(f"Saved figure to: {path}")
    except Exception as exc:
        print(f"Failed saving figure {path}: {exc}")


# ============================================================
# Confusion counts and metrics
# ============================================================


def compute_confusion_counts(
    data: pd.DataFrame, source_col: str, annotator_col: str
) -> Dict[str, int]:
    """
    Compute TP, FP, FN, TN using the canonical encoding:

    source column: 1 = present, 0 = absent
    annotator column:
      1 = TP (present and correctly captured)
    2 = FN (present but missed)
    3 = FP (source present, annotator labeled 3)
      'N/A' = TN (absent and correctly absent)
    """
    annot = data[annotator_col]
    annot_numeric = pd.to_numeric(annot, errors="coerce")

    TP = ((data[source_col] == 1) & (annot_numeric == 1)).sum()
    FN = ((data[source_col] == 1) & (annot_numeric == 2)).sum()
    FP = ((data[source_col] == 1) & (annot_numeric == 3)).sum()
    TN = ((data[source_col] == 0) & (annot == "N/A")).sum()

    return {
        "TP": int(TP),
        "FP": int(FP),
        "FN": int(FN),
        "TN": int(TN),
    }


def compute_metrics_from_counts(TP: int, FP: int, FN: int, TN: int) -> Dict[str, float]:
    """Compute diagnostic + classification metrics from confusion counts."""
    TP = float(TP)
    FP = float(FP)
    FN = float(FN)
    TN = float(TN)

    total = TP + FP + FN + TN
    sens_den = TP + FN
    spec_den = TN + FP
    ppv_den = TP + FP
    npv_den = TN + FN

    sensitivity = TP / sens_den if sens_den > 0 else np.nan
    specificity = TN / spec_den if spec_den > 0 else np.nan
    ppv = TP / ppv_den if ppv_den > 0 else np.nan
    npv = TN / npv_den if npv_den > 0 else np.nan
    accuracy = (TP + TN) / total if total > 0 else np.nan
    precision = ppv
    recall = sensitivity
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = np.nan

    fabrication_rate = FP / ppv_den if ppv_den > 0 else 0.0

    prevalence = sens_den / total if total > 0 else np.nan
    balanced_accuracy = (
        (sensitivity + specificity) / 2
        if not np.isnan(sensitivity) and not np.isnan(specificity)
        else np.nan
    )

    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fabrication_rate": fabrication_rate,
        "prevalence": prevalence,
        "balanced_accuracy": balanced_accuracy,
    }


# ============================================================
# Confidence intervals
# ============================================================


def wilson_ci(x: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (np.nan, np.nan)
    z = norm.ppf(1 - alpha / 2)
    p_hat = x / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    half_width = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
    return (center - half_width, center + half_width)


def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    random_state: int | None = None,
) -> Tuple[float, float]:
    """Nonparametric bootstrap CI for a 1D array of values."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return (np.nan, np.nan)

    rng = np.random.default_rng(random_state)
    boot_stats = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=values.size, replace=True)
        boot_stats.append(np.mean(sample))
    lower = np.percentile(boot_stats, 100 * alpha / 2)
    upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    return (float(lower), float(upper))


# ============================================================
# McNemar tests
# ============================================================


def annotator_correct(
    d: pd.DataFrame, source_col: str, annotator_col: str
) -> pd.Series:
    """
    Binary correctness for accuracy over all cases:
      - if source=1, correct if annotator==1
      - if source=0, correct if annotator=='N/A'
    """
    return ((d[source_col] == 1) & (d[annotator_col] == 1)) | (
        (d[source_col] == 0) & (d[annotator_col] == "N/A")
    )


def mcnemar_exact_from_masks(human_ok: pd.Series, ai_ok: pd.Series) -> float:
    """
    Exact McNemar p-value using binomial test on discordant pairs.
    Tests H0: p_human = p_ai vs H1: p_ai > p_human (one-sided).
    """
    human_ok = human_ok.astype(bool)
    ai_ok = ai_ok.astype(bool)
    b = int((human_ok & ~ai_ok).sum())
    c = int((~human_ok & ai_ok).sum())
    n_disc = b + c
    if n_disc == 0:
        return np.nan
    pval = binomtest(k=c, n=n_disc, p=0.5, alternative="greater").pvalue
    return float(pval)


def metric_correct_masks(
    data: pd.DataFrame,
    source_col: str,
    human_col: str,
    ai_col: str,
    metric_name: str,
) -> Tuple[pd.Series, pd.Series]:
    """
    Build correctness masks for human and AI for a given metric.
    """
    if metric_name == "accuracy":
        human_ok = annotator_correct(data, source_col, human_col)
        ai_ok = annotator_correct(data, source_col, ai_col)

    elif metric_name == "sensitivity":
        mask = data[source_col] == 1
        human_ok = (data.loc[mask, human_col] == 1).reindex(
            data.index, fill_value=False
        )
        ai_ok = (data.loc[mask, ai_col] == 1).reindex(data.index, fill_value=False)

    elif metric_name == "specificity":
        mask = data[source_col] == 0
        human_ok = (data.loc[mask, human_col] == "N/A").reindex(
            data.index, fill_value=False
        )
        ai_ok = (data.loc[mask, ai_col] == "N/A").reindex(data.index, fill_value=False)

    elif metric_name == "ppv":
        common = data[human_col].isin([1, 3]) & data[ai_col].isin([1, 3])
        human_ok = (
            (data.loc[common, source_col] == 1) & (data.loc[common, human_col] == 1)
        ).reindex(data.index, fill_value=False)
        ai_ok = (
            (data.loc[common, source_col] == 1) & (data.loc[common, ai_col] == 1)
        ).reindex(data.index, fill_value=False)

    elif metric_name == "npv":
        common = (data[human_col] == "N/A") & (data[ai_col] == "N/A")
        human_ok = (
            (data.loc[common, source_col] == 0) & (data.loc[common, human_col] == "N/A")
        ).reindex(data.index, fill_value=False)
        ai_ok = (
            (data.loc[common, source_col] == 0) & (data.loc[common, ai_col] == "N/A")
        ).reindex(data.index, fill_value=False)

    else:
        raise ValueError(f"Unsupported metric_name for McNemar: {metric_name}")

    return human_ok.astype(bool), ai_ok.astype(bool)


def element_metric_pvalue(
    data: pd.DataFrame,
    source_col: str,
    human_col: str,
    ai_col: str,
    metric_name: str,
) -> float:
    """Wrapper: compute McNemar-style p-value for a given metric."""
    human_ok, ai_ok = metric_correct_masks(
        data, source_col, human_col, ai_col, metric_name
    )
    return mcnemar_exact_from_masks(human_ok, ai_ok)


# ============================================================
# ROC / PR utilities (binary from categorical predictions)
# ============================================================


def binary_predictions_from_annotator(
    data: pd.DataFrame, source_col: str, annotator_col: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build binary ground truth and predictions from categorical encoding.

    y_true: 1 if source==1, 0 if source==0
    y_pred: 1 if annotator in {1,3}, 0 if annotator in {2,'N/A'}
    """
    mask_valid = data[source_col].isin([0, 1])
    d = data.loc[mask_valid].copy()

    y_true = d[source_col].astype(int).to_numpy()
    y_pred = np.where(d[annotator_col].isin([1, 3]), 1, 0)

    return y_true, y_pred


def roc_pr_from_binary(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute pseudo ROC/PR metrics from binary predictions.

    Since predictions are deterministic, we build:
      - ROC curve with 3 points: always negative, actual, always positive
      - PR curve similarly
    and compute AUCs.
    """
    unique_classes = np.unique(y_true)
    if unique_classes.size < 2:
        return {
            "roc_auc": np.nan,
            "pr_auc": np.nan,
        }
    # Use predicted probabilities as y_pred for sklearn; here we only have 0/1,
    # so treat them as scores.
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)

    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
    }


# ============================================================
# Plotting helpers
# ============================================================


def plot_confusion_heatmap(
    cm_df: pd.DataFrame,
    title: str,
    figsize: Tuple[int, int] = (5, 4),
):
    """
    Plot a 2x2 confusion matrix heatmap with counts and percentages.
        cm_df is a 2x2 DataFrame with:
            rows: Predicted Positive/Negative
            cols: True Positive/True Negative
    """
    fig, ax = plt.subplots(figsize=figsize)
    matrix = cm_df.values.astype(int)
    total = matrix.sum()
    annot = [
        [
            f"{int(matrix[i, j])}\n{(matrix[i, j] / total * 100 if total > 0 else 0):.1f}%"
            for j in range(2)
        ]
        for i in range(2)
    ]
    sns.heatmap(
        matrix,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=False,
        ax=ax,
        xticklabels=cm_df.columns.tolist(),
        yticklabels=cm_df.index.tolist(),
        annot_kws={"size": 11},
    )
    ax.set_title(title)
    fig.tight_layout()
    return fig


def build_confusion_df_from_counts(TP: int, FP: int, FN: int, TN: int) -> pd.DataFrame:
    """
    Build a 2x2 confusion matrix DataFrame from counts.
    Rows: Predicted Positive/Negative
    Cols: True Positive/True Negative
    """
    data = {
        "True Positive": [TP, FN],
        "True Negative": [FP, TN],
    }
    index = ["Predicted Positive", "Predicted Negative"]
    return pd.DataFrame(data, index=index)


# ============================================================
# Pretty formatting helpers
# ============================================================


def format_metric_with_ci(
    estimate: float, ci_low: float, ci_high: float, digits: int = 3
) -> str:
    if np.isnan(estimate):
        return "NA"
    return f"{estimate:.{digits}f} ({ci_low:.{digits}f}, {ci_high:.{digits}f})"
