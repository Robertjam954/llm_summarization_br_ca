"""
main_analysis.py

Element-level and domain-level diagnostic + classification metrics
for Human vs AI, using canonical TP/FP/FN/TN definitions, with
fabrication rates reported alongside dedicated tables/plots rather than
inside the core classification outputs.

- Element-level metrics and tables
- Domain-level metrics (aggregated confusion + element-balanced)
- McNemar p-values
- ROC/PR pseudo-curves
- Plots and CSV exports
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List
from math import ceil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from great_tables import GT
from sklearn.metrics import roc_curve, auc, precision_recall_curve

from metrics_utils import (
    compute_confusion_counts,
    compute_metrics_from_counts,
    bootstrap_ci,
    element_metric_pvalue,
    binary_predictions_from_annotator,
    roc_pr_from_binary,
    build_confusion_df_from_counts,
)

# ============================================================
# Data load and output configuration
# ============================================================

data = pd.read_excel(
    "data/raw/merged_llm_summary_validation_datasheet_deidentified.xlsx"
)
OUTPUT_DIR = Path("/Users/robertjames/Documents/llm_summarization/data reports")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

directory_path = Path("/Users/robertjames/Documents/llm_summarization")
if not directory_path.exists():
    print(f"Warning: expected project directory does not exist: {directory_path}")
else:
    try:
        os.chdir(directory_path)
        print(f"Set working directory to project root: {directory_path}")
    except Exception as exc:
        print(f"Warning: failed to set working directory to {directory_path}: {exc}")

if not OUTPUT_DIR.exists():
    print(f"Warning: OUTPUT_DIR does not exist and saves may fail: {OUTPUT_DIR}")

# ============================================================
# Assumptions about your encoding
# ============================================================
# source column: 1 = present, 0 = absent
# annotator column values:
#   1 = TP (present and correctly captured)
#   2 = FN (present but missed)
#   3 = FP (source present, annotator labeled 3)
#  'N/A' = TN (absent and correctly absent)
# ============================================================

string_data = ["NA", "na", "n/a", "N/A", "NA ", " na", " n/a", " N/A"]
string_null = "N/A"
data = data.replace(string_data, string_null)

# ----------------------------
# Reusable confusion-matrix fn
# ----------------------------


def compute_confusion_matrix(
    data: pd.DataFrame, source_col: str, annotator_col: str
) -> dict:
    """
    Computes confusion matrix counts and rates for one element and one annotator.

    Returns a dict with TP, FP, FN, TN and derived rates (TPR/FNR/FPR/TNR).
    """
    TP = ((data[source_col] == 1) & (data[annotator_col] == 1)).sum()
    FN = ((data[source_col] == 1) & (data[annotator_col] == 2)).sum()
    FP = ((data[source_col] == 1) & (data[annotator_col] == 3)).sum()
    TN = ((data[source_col] == 0) & (data[annotator_col] == "N/A")).sum()

    P = TP + FN
    N = TN + FP

    return {
        "TP": int(TP),
        "FP": int(FP),
        "FN": int(FN),
        "TN": int(TN),
        "TPR": (TP / P) if P > 0 else np.nan,  # sensitivity/recall
        "FNR": (FN / P) if P > 0 else np.nan,
        "FPR": (FP / N) if N > 0 else np.nan,
        "TNR": (TN / N) if N > 0 else np.nan,  # specificity
        "accuracy": (TP + TN) / (P + N) if (P + N) > 0 else np.nan,
        "precision": (TP / (TP + FP)) if (TP + FP) > 0 else np.nan,
        "recall": (TP / (TP + FN)) if (TP + FN) > 0 else np.nan,
        "f1": (2 * TP / (2 * TP + FP + FN)) if (2 * TP + FP + FN) > 0 else np.nan,
        "PPV": (TP / (TP + FP)) if (TP + FP) > 0 else np.nan,
        "NPV": (TN / (TN + FN)) if (TN + FN) > 0 else np.nan,
        "Observations": int(P + N),
        "fabrication_rate": (FP / (TP + FP)) if (TP + FP) > 0 else 0.0,
    }


# ----------------------------
# Helper save wrappers
# ----------------------------


def safe_save_dataframe(df: pd.DataFrame, path: Path) -> None:
    """Save a dataframe to CSV only if OUTPUT_DIR exists, else warn and skip."""
    if not OUTPUT_DIR.exists():
        print(f"Warning: OUTPUT_DIR does not exist, skipping CSV save: {path}")
        return
    try:
        df.to_csv(path)
        print(f"Saved CSV to: {path}")
    except Exception as exc:
        print(f"Failed saving CSV {path}: {exc}")


def safe_save_plt(fig, path: Path, **savefig_kwargs) -> None:
    """Save a matplotlib figure only if OUTPUT_DIR exists, else warn and skip."""
    if not OUTPUT_DIR.exists():
        print(f"Warning: OUTPUT_DIR does not exist, skipping figure save: {path}")
        return
    try:
        fig.savefig(path, **savefig_kwargs)
        print(f"Saved figure to: {path}")
    except Exception as exc:
        print(f"Failed saving figure {path}: {exc}")


def safe_save_plotly(fig, path: Path, write_kw: dict | None = None) -> None:
    """Save a Plotly figure to an image (via kaleido) only if OUTPUT_DIR exists.
    If Kaleido/Chrome cannot render, prints a helpful message and skips.
    """
    if not OUTPUT_DIR.exists():
        print(f"Warning: OUTPUT_DIR does not exist, skipping Plotly save: {path}")
        return
    try:
        if write_kw is None:
            write_kw = {}
        fig.write_image(path, **write_kw)
        print(f"Saved plotly image to: {path}")
    except Exception as exc:
        print(
            "Skipping Plotly static export because Kaleido could not render the figure. "
            "Install Chrome or run `kaleido.get_chrome()` to enable PNG export."
        )
        print(f"Kaleido error: {exc}")


# ============================================================
# Element and domain definitions (aliases preserved)
# ============================================================

ELEMENTS: Dict[str, Dict[str, str]] = {
    "Lesion Size": {
        "source": "lesion_size_status_source",
        "human": "lesion_size_status_human",
        "ai": "lesion_size_status_ai",
    },
    "Lesion Laterality": {
        "source": "laterality_status_source",
        "human": "laterality_status_human",
        "ai": "laterality_status_ai",
    },
    "Lesion Location": {
        "source": "lesion_location_status_source",
        "human": "lesion_location_status_human",
        "ai": "lesion_location_status_ai",
    },
    "Calcifications / Asymmetry": {
        "source": "calcifications_asymmetry_status_source",
        "human": "calcifications_asymmetry_status_human",
        "ai": "calcifications_asymmetry_status_ai",
    },
    "Additional Enhancement (MRI)": {
        "source": "additional_enhancement_mri_status_source",
        "human": "additional_enhancement_mri_status_human",
        "ai": "additional_enhancement_mri_status_ai",
    },
    "Extent": {
        "source": "extent_status_source",
        "human": "extent_status_human",
        "ai": "extent_status_ai",
    },
    "Accurate Clip Placement": {
        "source": "accurate_clip_placement_status_source",
        "human": "accurate_clip_placement_status_human",
        "ai": "accurate_clip_placement_status_ai",
    },
    "Workup Recommendation": {
        "source": "workup_recommendation_status_source",
        "human": "workup_recommendation_status_human",
        "ai": "workup_recommendation_status_ai",
    },
    "Lymph Node": {
        "source": "Lymph node_status_source",
        "human": "Lymph node_status_human",
        "ai": "Lymph node_status_ai",
    },
    "Chronology Preserved": {
        "source": "chronology_preserved_status_source",
        "human": "chronology_preserved_status_human",
        "ai": "chronology_preserved_status_ai",
    },
    "Biopsy Method": {
        "source": "biopsy_method_status_source",
        "human": "biopsy_method_status_human",
        "ai": "biopsy_method_status_ai",
    },
    "Invasive Component Size (Pathology)": {
        "source": "invasive_component_size_pathology_status_source",
        "human": "invasive_component_size_pathology_status_human",
        "ai": "invasive_component_size_pathology_status_ai",
    },
    "Histologic Diagnosis": {
        "source": "histologic_diagnosis_status_source",
        "human": "histologic_diagnosis_status_human",
        "ai": "histologic_diagnosis_status_ai",
    },
    "Receptor Status": {
        "source": "receptor_status_source",
        "human": "receptor_status_human",
        "ai": "receptor_status_ai",
    },
}
for col in ELEMENTS.values():
    source_col = col["source"]  # example to access source column for lesion_size
    human_col = col["human"]  # example to access human column for lesion_size
    ai_col = col["ai"]  # example to access ai column for lesion_size


DOMAINS: Dict[str, List[str]] = {
    "Radiology": [
        "Lesion Size",
        "Lesion Laterality",
        "Lesion Location",
        "Calcifications / Asymmetry",
        "Additional Enhancement (MRI)",
        "Extent",
        "Accurate Clip Placement",
        "Workup Recommendation",
        "Lymph Node",
        "Chronology Preserved",
        "Biopsy Method",
    ],
    "Pathology": [
        "Biopsy Method",
        "Invasive Component Size (Pathology)",
        "Histologic Diagnosis",
        "Receptor Status",
    ],
}
# ----------------------------
# Element-level totals (non-null + null) for Human and AI
element_total_rows = []
for element_name, cols in ELEMENTS.items():
    human_col = cols.get("human")
    ai_col = cols.get("ai")

    if human_col and human_col in data.columns:
        element_total_rows.append(
            {
                "Element": element_name,
                "Annotator": "Human",
                "Non-Null Count": int(data[human_col].notna().sum()),
                "Null Count": int(data[human_col].isna().sum()),
            }
        )
    if ai_col and ai_col in data.columns:
        element_total_rows.append(
            {
                "Element": element_name,
                "Annotator": "AI",
                "Non-Null Count": int(data[ai_col].notna().sum()),
                "Null Count": int(data[ai_col].isna().sum()),
            }
        )

df_element_totals = pd.DataFrame(element_total_rows)
element_order_list = sorted(df_element_totals["Element"].unique())
annotator_order = ["Human", "AI"]
df_element_totals["Element"] = pd.Categorical(
    df_element_totals["Element"], categories=element_order_list, ordered=True
)
df_element_totals["Annotator"] = pd.Categorical(
    df_element_totals["Annotator"], categories=annotator_order, ordered=True
)
df_element_totals = df_element_totals.sort_values(["Element", "Annotator"]).reset_index(
    drop=True
)

try:
    gt_element_totals = (
        GT(df_element_totals, rowname_col="Annotator", groupname_col="Element")
        .tab_header(
            title="Element-Level Non-Null and Null Counts",
            subtitle="Human vs AI counts per element",
        )
        .tab_stubhead(label="Element")
        .tab_spanner(label="Counts", columns=["Non-Null Count", "Null Count"])
        .tab_options(
            table_border_top_color="#004D80",
            table_border_bottom_color="#004D80",
            heading_border_bottom_color="#0076BA",
            column_labels_border_top_color="#0076BA",
            column_labels_border_bottom_color="#0076BA",
            column_labels_background_color="#FFFFFF",
            row_group_border_top_color="#0076BA",
            row_group_border_bottom_color="#0076BA",
            stub_background_color="#0076BA",
            stub_border_style="solid",
            stub_border_color="#0076BA",
            table_body_border_top_color="#0076BA",
            table_body_border_bottom_color="#0076BA",
            table_body_hlines_style="none",
            table_body_vlines_style="none",
        )
    )

    element_totals_png = OUTPUT_DIR / "element_level_non_null_null_counts.png"
    GT.save(gt_element_totals, file=str(element_totals_png))
except Exception as exc:
    print(f"Failed generating element totals table: {exc}")


# ============================================================
# Element-level metrics
# ============================================================

BOOTSTRAP_METRICS = [
    "accuracy",
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
    "precision",
    "recall",
    "fabrication_rate",
    "f1",
]


def _safe_ci_from_samples(values: np.ndarray) -> tuple[float, float]:
    if np.all(np.isnan(values)):
        return (np.nan, np.nan)
    return (
        float(np.nanpercentile(values, 2.5)),
        float(np.nanpercentile(values, 97.5)),
    )


def _bootstrap_metric_cis(
    d: pd.DataFrame,
    source_col: str,
    annot_col: str,
    n_boot: int = 2000,
    seed: int = 123,
) -> Dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    n = len(d)
    if n == 0:
        return {metric: (np.nan, np.nan) for metric in BOOTSTRAP_METRICS}

    samples = {metric: np.empty(n_boot, dtype=float) for metric in BOOTSTRAP_METRICS}
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        bs = d.iloc[idx]
        counts = compute_confusion_counts(bs, source_col, annot_col)
        metrics = compute_metrics_from_counts(**counts)
        for metric in BOOTSTRAP_METRICS:
            samples[metric][i] = metrics[metric]

    cis: Dict[str, tuple[float, float]] = {}
    for metric, values in samples.items():
        cis[metric] = _safe_ci_from_samples(values)
    return cis


def _bootstrap_domain_metric_cis(
    d: pd.DataFrame,
    element_names: List[str],
    annot_key: str,
    n_boot: int = 2000,
    seed: int = 123,
) -> Dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    n = len(d)
    if n == 0:
        return {metric: (np.nan, np.nan) for metric in BOOTSTRAP_METRICS}

    valid_elements = []
    for element_name in element_names:
        cols = ELEMENTS.get(element_name)
        if not cols:
            continue
        source_col = cols["source"]
        annot_col = cols[annot_key]
        if source_col not in d.columns or annot_col not in d.columns:
            continue
        valid_elements.append((source_col, annot_col))

    if not valid_elements:
        return {metric: (np.nan, np.nan) for metric in BOOTSTRAP_METRICS}

    samples = {metric: np.empty(n_boot, dtype=float) for metric in BOOTSTRAP_METRICS}
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        bs = d.iloc[idx]
        total_counts = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
        for source_col, annot_col in valid_elements:
            counts = compute_confusion_counts(bs, source_col, annot_col)
            for key in total_counts:
                total_counts[key] += counts[key]
        metrics = compute_metrics_from_counts(**total_counts)
        for metric in BOOTSTRAP_METRICS:
            samples[metric][i] = metrics[metric]

    cis: Dict[str, tuple[float, float]] = {}
    for metric, values in samples.items():
        cis[metric] = _safe_ci_from_samples(values)
    return cis


element_rows = []

for element_name, cols in ELEMENTS.items():
    source_col = cols["source"]
    human_col = cols["human"]
    ai_col = cols["ai"]

    # Sanity check
    missing_cols = [c for c in [source_col, human_col, ai_col] if c not in data.columns]
    if missing_cols:
        print(f"Warning: missing columns for element '{element_name}': {missing_cols}")
        continue

    # Human
    human_counts = compute_confusion_counts(data, source_col, human_col)
    human_metrics = compute_metrics_from_counts(**human_counts)
    human_total = sum(human_counts.values())

    # AI
    ai_counts = compute_confusion_counts(data, source_col, ai_col)
    ai_metrics = compute_metrics_from_counts(**ai_counts)
    ai_total = sum(ai_counts.values())

    # Bootstrap CIs for all metrics
    ci_h = _bootstrap_metric_cis(data, source_col, human_col)
    ci_ai = _bootstrap_metric_cis(data, source_col, ai_col)

    acc_ci_h = ci_h["accuracy"]
    sens_ci_h = ci_h["sensitivity"]
    spec_ci_h = ci_h["specificity"]
    ppv_ci_h = ci_h["ppv"]
    npv_ci_h = ci_h["npv"]
    fabrication_ci_h = ci_h["fabrication_rate"]

    acc_ci_ai = ci_ai["accuracy"]
    sens_ci_ai = ci_ai["sensitivity"]
    spec_ci_ai = ci_ai["specificity"]
    ppv_ci_ai = ci_ai["ppv"]
    npv_ci_ai = ci_ai["npv"]
    fabrication_ci_ai = ci_ai["fabrication_rate"]

    # McNemar p-values (accuracy, sensitivity, specificity, PPV, NPV)
    p_acc = element_metric_pvalue(
        data, source_col, human_col, ai_col, metric_name="accuracy"
    )
    p_sens = element_metric_pvalue(
        data, source_col, human_col, ai_col, metric_name="sensitivity"
    )
    p_spec = element_metric_pvalue(
        data, source_col, human_col, ai_col, metric_name="specificity"
    )
    p_ppv = element_metric_pvalue(
        data, source_col, human_col, ai_col, metric_name="ppv"
    )
    p_npv = element_metric_pvalue(
        data, source_col, human_col, ai_col, metric_name="npv"
    )

    # ROC/PR (binary from categorical)
    y_true_h, y_pred_h = binary_predictions_from_annotator(data, source_col, human_col)
    y_true_ai, y_pred_ai = binary_predictions_from_annotator(data, source_col, ai_col)

    rocpr_h = roc_pr_from_binary(y_true_h, y_pred_h)
    rocpr_ai = roc_pr_from_binary(y_true_ai, y_pred_ai)

    # Store rows (one per annotator)
    element_rows.append(
        {
            "Element": element_name,
            "Annotator": "Human",
            "TP": human_counts["TP"],
            "FP": human_counts["FP"],
            "FN": human_counts["FN"],
            "TN": human_counts["TN"],
            "accuracy": human_metrics["accuracy"],
            "accuracy_ci_low": acc_ci_h[0],
            "accuracy_ci_high": acc_ci_h[1],
            "sensitivity": human_metrics["sensitivity"],
            "sensitivity_ci_low": sens_ci_h[0],
            "sensitivity_ci_high": sens_ci_h[1],
            "specificity": human_metrics["specificity"],
            "specificity_ci_low": spec_ci_h[0],
            "specificity_ci_high": spec_ci_h[1],
            "ppv": human_metrics["ppv"],
            "ppv_ci_low": ppv_ci_h[0],
            "ppv_ci_high": ppv_ci_h[1],
            "fabrication_rate": human_metrics["fabrication_rate"],
            "fabrication_rate_ci_low": fabrication_ci_h[0],
            "fabrication_rate_ci_high": fabrication_ci_h[1],
            "npv": human_metrics["npv"],
            "npv_ci_low": npv_ci_h[0],
            "npv_ci_high": npv_ci_h[1],
            "precision": human_metrics["precision"],
            "recall": human_metrics["recall"],
            "f1": human_metrics["f1"],
            "roc_auc": rocpr_h["roc_auc"],
            "pr_auc": rocpr_h["pr_auc"],
            "p_mcnemar_accuracy": p_acc,
            "p_mcnemar_sensitivity": p_sens,
            "p_mcnemar_specificity": p_spec,
            "p_mcnemar_ppv": p_ppv,
            "p_mcnemar_npv": p_npv,
        }
    )

    element_rows.append(
        {
            "Element": element_name,
            "Annotator": "AI",
            "TP": ai_counts["TP"],
            "FP": ai_counts["FP"],
            "FN": ai_counts["FN"],
            "TN": ai_counts["TN"],
            "accuracy": ai_metrics["accuracy"],
            "accuracy_ci_low": acc_ci_ai[0],
            "accuracy_ci_high": acc_ci_ai[1],
            "sensitivity": ai_metrics["sensitivity"],
            "sensitivity_ci_low": sens_ci_ai[0],
            "sensitivity_ci_high": sens_ci_ai[1],
            "specificity": ai_metrics["specificity"],
            "specificity_ci_low": spec_ci_ai[0],
            "specificity_ci_high": spec_ci_ai[1],
            "ppv": ai_metrics["ppv"],
            "ppv_ci_low": ppv_ci_ai[0],
            "ppv_ci_high": ppv_ci_ai[1],
            "fabrication_rate": ai_metrics["fabrication_rate"],
            "fabrication_rate_ci_low": fabrication_ci_ai[0],
            "fabrication_rate_ci_high": fabrication_ci_ai[1],
            "npv": ai_metrics["npv"],
            "npv_ci_low": npv_ci_ai[0],
            "npv_ci_high": npv_ci_ai[1],
            "precision": ai_metrics["precision"],
            "recall": ai_metrics["recall"],
            "f1": ai_metrics["f1"],
            "roc_auc": rocpr_ai["roc_auc"],
            "pr_auc": rocpr_ai["pr_auc"],
            "p_mcnemar_accuracy": p_acc,
            "p_mcnemar_sensitivity": p_sens,
            "p_mcnemar_specificity": p_spec,
            "p_mcnemar_ppv": p_ppv,
            "p_mcnemar_npv": p_npv,
        }
    )

df_elements = pd.DataFrame(element_rows)
safe_save_dataframe(df_elements, OUTPUT_DIR / "element_level_metrics.csv")

# Debug: confirm FP counts (value==3) and fabrication rate by element/annotator
fp_debug_cols = ["Element", "Annotator", "TP", "FP", "FN", "TN", "fabrication_rate"]
df_fp_debug = df_elements[fp_debug_cols].copy()
safe_save_dataframe(df_fp_debug, OUTPUT_DIR / "fabrication_rate_debug_counts.csv")


# ============================================================
# Element-level confusion matrices (aggregated by annotator)
# ============================================================

agg_counts = df_elements.groupby("Annotator")[["TP", "FP", "FN", "TN"]].sum()

cm_human_df = None
cm_ai_df = None

if "Human" in agg_counts.index:
    c = agg_counts.loc["Human"]
    cm_human_df = build_confusion_df_from_counts(
        TP=int(c["TP"].item()),
        FP=int(c["FP"].item()),
        FN=int(c["FN"].item()),
        TN=int(c["TN"].item()),
    )
    safe_save_dataframe(cm_human_df.reset_index(), OUTPUT_DIR / "confusion_human.csv")

if "AI" in agg_counts.index:
    c = agg_counts.loc["AI"]
    cm_ai_df = build_confusion_df_from_counts(
        TP=int(c["TP"].item()),
        FP=int(c["FP"].item()),
        FN=int(c["FN"].item()),
        TN=int(c["TN"].item()),
    )
    safe_save_dataframe(cm_ai_df.reset_index(), OUTPUT_DIR / "confusion_ai.csv")

# Side-by-side confusion heatmaps
if cm_human_df is not None and cm_ai_df is not None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, cm_df, title in [
        (axes[0], cm_human_df, "Human - Confusion Heatmap"),
        (axes[1], cm_ai_df, "AI - Confusion Heatmap"),
    ]:
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

    safe_save_plt(
        fig, OUTPUT_DIR / "confusion_heatmaps.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)


# ============================================================
# Domain-level metrics
#   System A: aggregated confusion
#   System B: element-balanced (mean of element metrics)
# ============================================================

domain_rows_agg = []
domain_rows_balanced = []

for domain_name, element_list in DOMAINS.items():
    # Filter to elements that exist in df_elements
    domain_elements = [e for e in element_list if e in df_elements["Element"].unique()]
    if not domain_elements:
        continue

    df_dom = df_elements[df_elements["Element"].isin(domain_elements)].copy()

    # -----------------------------
    # System A: aggregated confusion
    # -----------------------------
    for annot in ["Human", "AI"]:
        df_a = df_dom[df_dom["Annotator"] == annot]
        if df_a.empty:
            continue

        TP = int(df_a["TP"].sum())
        FP = int(df_a["FP"].sum())
        FN = int(df_a["FN"].sum())
        TN = int(df_a["TN"].sum())

        metrics = compute_metrics_from_counts(TP, FP, FN, TN)
        total = TP + FP + FN + TN

        ci_dom = _bootstrap_domain_metric_cis(
            data,
            domain_elements,
            "human" if annot == "Human" else "ai",
        )
        acc_ci = ci_dom["accuracy"]
        sens_ci = ci_dom["sensitivity"]
        spec_ci = ci_dom["specificity"]
        ppv_ci = ci_dom["ppv"]
        npv_ci = ci_dom["npv"]
        fabrication_ci = ci_dom["fabrication_rate"]

        domain_rows_agg.append(
            {
                "Domain": domain_name,
                "Annotator": annot,
                "TP": TP,
                "FP": FP,
                "FN": FN,
                "TN": TN,
                "accuracy": metrics["accuracy"],
                "accuracy_ci_low": acc_ci[0],
                "accuracy_ci_high": acc_ci[1],
                "sensitivity": metrics["sensitivity"],
                "sensitivity_ci_low": sens_ci[0],
                "sensitivity_ci_high": sens_ci[1],
                "specificity": metrics["specificity"],
                "specificity_ci_low": spec_ci[0],
                "specificity_ci_high": spec_ci[1],
                "ppv": metrics["ppv"],
                "ppv_ci_low": ppv_ci[0],
                "ppv_ci_high": ppv_ci[1],
                "fabrication_rate": metrics["fabrication_rate"],
                "fabrication_rate_ci_low": fabrication_ci[0],
                "fabrication_rate_ci_high": fabrication_ci[1],
                "npv": metrics["npv"],
                "npv_ci_low": npv_ci[0],
                "npv_ci_high": npv_ci[1],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
            }
        )

    # -----------------------------------
    # System B: element-balanced (means)
    # -----------------------------------
    for annot in ["Human", "AI"]:
        df_b = df_dom[df_dom["Annotator"] == annot]
        if df_b.empty:
            continue

        # Mean of element-level metrics
        metrics_mean = {
            "accuracy": df_b["accuracy"].mean(),
            "sensitivity": df_b["sensitivity"].mean(),
            "specificity": df_b["specificity"].mean(),
            "ppv": df_b["ppv"].mean(),
            "npv": df_b["npv"].mean(),
            "precision": df_b["precision"].mean(),
            "recall": df_b["recall"].mean(),
            "f1": df_b["f1"].mean(),
            "fabrication_rate": df_b["fabrication_rate"].mean(),
        }

        # Bootstrap CIs across elements
        acc_ci = bootstrap_ci(df_b["accuracy"].to_numpy())
        sens_ci = bootstrap_ci(df_b["sensitivity"].to_numpy())
        spec_ci = bootstrap_ci(df_b["specificity"].to_numpy())
        ppv_ci = bootstrap_ci(df_b["ppv"].to_numpy())
        npv_ci = bootstrap_ci(df_b["npv"].to_numpy())
        f1_ci = bootstrap_ci(df_b["f1"].to_numpy())
        fabrication_ci = bootstrap_ci(df_b["fabrication_rate"].to_numpy())

        domain_rows_balanced.append(
            {
                "Domain": domain_name,
                "Annotator": annot,
                "accuracy": metrics_mean["accuracy"],
                "accuracy_ci_low": acc_ci[0],
                "accuracy_ci_high": acc_ci[1],
                "sensitivity": metrics_mean["sensitivity"],
                "sensitivity_ci_low": sens_ci[0],
                "sensitivity_ci_high": sens_ci[1],
                "specificity": metrics_mean["specificity"],
                "specificity_ci_low": spec_ci[0],
                "specificity_ci_high": spec_ci[1],
                "ppv": metrics_mean["ppv"],
                "ppv_ci_low": ppv_ci[0],
                "ppv_ci_high": ppv_ci[1],
                "fabrication_rate": metrics_mean["fabrication_rate"],
                "fabrication_rate_ci_low": fabrication_ci[0],
                "fabrication_rate_ci_high": fabrication_ci[1],
                "npv": metrics_mean["npv"],
                "npv_ci_low": npv_ci[0],
                "npv_ci_high": npv_ci[1],
                "precision": metrics_mean["precision"],
                "recall": metrics_mean["recall"],
                "f1": metrics_mean["f1"],
                "f1_ci_low": f1_ci[0],
                "f1_ci_high": f1_ci[1],
            }
        )

df_domain_agg = pd.DataFrame(domain_rows_agg)
df_domain_balanced = pd.DataFrame(domain_rows_balanced)

safe_save_dataframe(df_domain_agg, OUTPUT_DIR / "domain_level_aggregated_metrics.csv")
safe_save_dataframe(
    df_domain_balanced, OUTPUT_DIR / "domain_level_element_balanced_metrics.csv"
)


# ============================================================
# Optional: pretty-print summary tables (wide format)
# ============================================================


def build_element_summary_table(df_elements: pd.DataFrame) -> pd.DataFrame:
    """Wide table: Element x (accuracy, sensitivity, specificity, ppv, npv, f1) for Human vs AI."""
    cols = [
        "accuracy",
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "f1",
    ]
    df = df_elements.set_index(["Element", "Annotator"])[cols].unstack("Annotator")
    df.columns = [f"{metric}_{annotator.lower()}" for metric, annotator in df.columns]
    df = df.reset_index()
    return df


element_summary = build_element_summary_table(df_elements)
safe_save_dataframe(element_summary, OUTPUT_DIR / "element_level_summary_wide.csv")


# ============================================================
# Element-level non-null/null totals (Human vs AI)
# ============================================================

element_total_rows = []
for element_name, cols in ELEMENTS.items():
    human_col = cols["human"]
    ai_col = cols["ai"]

    if human_col in data.columns:
        element_total_rows.append(
            {
                "Element": element_name,
                "Annotator": "Human",
                "Non-Null Count": int(data[human_col].notna().sum()),
                "Null Count": int(data[human_col].isna().sum()),
            }
        )
    if ai_col in data.columns:
        element_total_rows.append(
            {
                "Element": element_name,
                "Annotator": "AI",
                "Non-Null Count": int(data[ai_col].notna().sum()),
                "Null Count": int(data[ai_col].isna().sum()),
            }
        )

df_element_totals = pd.DataFrame(element_total_rows)
element_order_list = sorted(df_element_totals["Element"].unique())
annotator_order = ["Human", "AI"]
df_element_totals["Element"] = pd.Categorical(
    df_element_totals["Element"], categories=element_order_list, ordered=True
)
df_element_totals["Annotator"] = pd.Categorical(
    df_element_totals["Annotator"], categories=annotator_order, ordered=True
)
df_element_totals = df_element_totals.sort_values(["Element", "Annotator"]).reset_index(
    drop=True
)

try:
    gt_element_totals = (
        GT(df_element_totals, rowname_col="Annotator", groupname_col="Element")
        .tab_header(
            title="Element-Level Non-Null and Null Counts",
            subtitle="Human vs AI counts per element",
        )
        .tab_stubhead(label="Element")
        .tab_spanner(label="Counts", columns=["Non-Null Count", "Null Count"])
        .tab_options(
            table_border_top_color="#004D80",
            table_border_bottom_color="#004D80",
            heading_border_bottom_color="#0076BA",
            column_labels_border_top_color="#0076BA",
            column_labels_border_bottom_color="#0076BA",
            column_labels_background_color="#FFFFFF",
            row_group_border_top_color="#0076BA",
            row_group_border_bottom_color="#0076BA",
            stub_background_color="#0076BA",
            stub_border_style="solid",
            stub_border_color="#0076BA",
            table_body_border_top_color="#0076BA",
            table_body_border_bottom_color="#0076BA",
            table_body_hlines_style="none",
            table_body_vlines_style="none",
        )
    )
    element_totals_png = OUTPUT_DIR / "element_level_non_null_null_counts.png"
    GT.save(gt_element_totals, file=str(element_totals_png))
except Exception as exc:
    print(f"Failed generating element totals table: {exc}")


# ============================================================
# Confusion matrix tables (side-by-side text tables)
# ============================================================

try:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax in axes:
        ax.axis("off")

    if cm_human_df is not None:
        axes[0].set_title("Human - Confusion Matrix")
        tbl = axes[0].table(
            cellText=cm_human_df.values.tolist(),
            colLabels=cm_human_df.columns.tolist(),
            rowLabels=cm_human_df.index.tolist(),
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1, 1.4)
    else:
        axes[0].text(0.5, 0.5, "No Human rows", ha="center", va="center")

    if cm_ai_df is not None:
        axes[1].set_title("AI - Confusion Matrix")
        tbl = axes[1].table(
            cellText=cm_ai_df.values.tolist(),
            colLabels=cm_ai_df.columns.tolist(),
            rowLabels=cm_ai_df.index.tolist(),
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1, 1.4)
    else:
        axes[1].text(0.5, 0.5, "No AI rows", ha="center", va="center")

    fig.tight_layout()
    safe_save_plt(
        fig, OUTPUT_DIR / "confusion_tables.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)
except Exception as exc:
    print(f"Failed saving confusion tables PNG: {exc}")


# ============================================================
# Helper formatting for tables and plots
# ============================================================


def _sig_star(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _fmt_cell(est: float, lo: float, hi: float, p: float, decimals: int = 3) -> str:
    if pd.isna(est):
        return ""
    return f"{est:.{decimals}f} ({lo:.{decimals}f}, {hi:.{decimals}f}){_sig_star(p)}"


def _fmt_pct_cell(est: float, lo: float, hi: float, p: float, decimals: int = 1) -> str:
    if pd.isna(est):
        return ""
    return f"{est * 100:.{decimals}f}% ({lo * 100:.{decimals}f}, {hi * 100:.{decimals}f}){_sig_star(p)}"


def _bootstrap_f1_ci_for_element(
    d: pd.DataFrame,
    source_col: str,
    annot_col: str,
    n_boot: int = 2000,
    seed: int = 123,
):
    rng = np.random.default_rng(seed)
    n = len(d)
    if n == 0:
        return (np.nan, np.nan, np.nan)
    f1s = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        bs = d.iloc[idx]
        counts = compute_confusion_counts(bs, source_col, annot_col)
        metrics = compute_metrics_from_counts(**counts)
        f1s[i] = metrics["f1"]
    point = float(np.nanmean(f1s))
    lo, hi = _safe_ci_from_samples(f1s)
    return point, lo, hi


def _bootstrap_f1_diff_ci_for_element(
    d: pd.DataFrame,
    source_col: str,
    human_col: str,
    ai_col: str,
    n_boot: int = 2000,
    seed: int = 123,
):
    rng = np.random.default_rng(seed)
    n = len(d)
    if n == 0:
        return (np.nan, np.nan)
    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        bs = d.iloc[idx]
        h_counts = compute_confusion_counts(bs, source_col, human_col)
        a_counts = compute_confusion_counts(bs, source_col, ai_col)
        h_f1 = compute_metrics_from_counts(**h_counts)["f1"]
        a_f1 = compute_metrics_from_counts(**a_counts)["f1"]
        diffs[i] = h_f1 - a_f1
    lo, hi = _safe_ci_from_samples(diffs)
    return lo, hi


def _fabrication_rate_from_counts(tp: int, fp: int) -> float:
    denom = tp + fp
    return fp / denom if denom > 0 else np.nan


def _bootstrap_fabrication_ci_element(
    d: pd.DataFrame,
    source_col: str,
    annot_col: str,
    n_boot: int = 2000,
    seed: int = 123,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(d)
    if n == 0:
        return (np.nan, np.nan)
    samples = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        bs = d.iloc[idx]
        counts = compute_confusion_counts(bs, source_col, annot_col)
        samples[i] = _fabrication_rate_from_counts(counts["TP"], counts["FP"])
    return _safe_ci_from_samples(samples)


def _bootstrap_fabrication_ci_aggregate(
    d: pd.DataFrame,
    element_names: List[str],
    annot_key: str,
    n_boot: int = 2000,
    seed: int = 123,
) -> tuple[float, float]:
    cis = _bootstrap_domain_metric_cis(
        d,
        element_names,
        annot_key,
        n_boot=n_boot,
        seed=seed,
    )
    return cis["fabrication_rate"]


def _bootstrap_fabrication_diff_p_one_sided_element(
    d: pd.DataFrame,
    source_col: str,
    human_col: str,
    ai_col: str,
    n_boot: int = 5000,
    seed: int = 123,
) -> float:
    rng = np.random.default_rng(seed)
    n = len(d)
    if n == 0:
        return np.nan

    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        bs = d.iloc[idx]
        h_counts = compute_confusion_counts(bs, source_col, human_col)
        a_counts = compute_confusion_counts(bs, source_col, ai_col)
        h_rate = _fabrication_rate_from_counts(h_counts["TP"], h_counts["FP"])
        a_rate = _fabrication_rate_from_counts(a_counts["TP"], a_counts["FP"])
        if np.isnan(h_rate) or np.isnan(a_rate):
            continue
        diffs.append(h_rate - a_rate)

    if not diffs:
        return np.nan

    diffs_arr = np.asarray(diffs, dtype=float)
    return float((np.sum(diffs_arr <= 0) + 1) / (len(diffs_arr) + 1))


def _bootstrap_fabrication_diff_p_one_sided_aggregate(
    d: pd.DataFrame,
    element_names: List[str],
    n_boot: int = 5000,
    seed: int = 123,
) -> float:
    rng = np.random.default_rng(seed)
    n = len(d)
    if n == 0:
        return np.nan

    valid_elements = []
    for element_name in element_names:
        cols = ELEMENTS.get(element_name)
        if not cols:
            continue
        required = [cols["source"], cols["human"], cols["ai"]]
        if not all(col in d.columns for col in required):
            continue
        valid_elements.append(cols)

    if not valid_elements:
        return np.nan

    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        bs = d.iloc[idx]
        h_tp = h_fp = a_tp = a_fp = 0
        for cols in valid_elements:
            h_counts = compute_confusion_counts(bs, cols["source"], cols["human"])
            a_counts = compute_confusion_counts(bs, cols["source"], cols["ai"])
            h_tp += int(h_counts["TP"])
            h_fp += int(h_counts["FP"])
            a_tp += int(a_counts["TP"])
            a_fp += int(a_counts["FP"])
        h_rate = _fabrication_rate_from_counts(h_tp, h_fp)
        a_rate = _fabrication_rate_from_counts(a_tp, a_fp)
        if np.isnan(h_rate) or np.isnan(a_rate):
            continue
        diffs.append(h_rate - a_rate)

    if not diffs:
        return np.nan

    diffs_arr = np.asarray(diffs, dtype=float)
    return float((np.sum(diffs_arr <= 0) + 1) / (len(diffs_arr) + 1))


def _fabrication_ylim(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    ymax = float(finite.max())
    return max(0.05, min(1.0, ymax * 1.15))


# ============================================================
# Element-level diagnostic + classification tables (Great Tables)
# ============================================================

try:
    diag_rows = []
    class_rows = []

    for element_name, cols in ELEMENTS.items():
        source_col = cols["source"]
        human_col = cols["human"]
        ai_col = cols["ai"]

        if element_name not in df_elements["Element"].unique():
            continue

        df_el = df_elements[df_elements["Element"] == element_name]
        for annot in ["Human", "AI"]:
            r = df_el[df_el["Annotator"] == annot].iloc[0]

            diag_rows.append(
                {
                    "Element": element_name,
                    "Annotator": annot,
                    "Sensitivity": _fmt_cell(
                        r["sensitivity"],
                        r["sensitivity_ci_low"],
                        r["sensitivity_ci_high"],
                        r["p_mcnemar_sensitivity"],
                    ),
                    "Specificity": _fmt_cell(
                        r["specificity"],
                        r["specificity_ci_low"],
                        r["specificity_ci_high"],
                        r["p_mcnemar_specificity"],
                    ),
                    "PPV": _fmt_cell(
                        r["ppv"],
                        r["ppv_ci_low"],
                        r["ppv_ci_high"],
                        r["p_mcnemar_ppv"],
                    ),
                    "NPV": _fmt_cell(
                        r["npv"],
                        r["npv_ci_low"],
                        r["npv_ci_high"],
                        r["p_mcnemar_npv"],
                    ),
                }
            )

            f1_point, f1_lo, f1_hi = _bootstrap_f1_ci_for_element(
                data, source_col, cols["human"] if annot == "Human" else cols["ai"]
            )

            class_rows.append(
                {
                    "Element": element_name,
                    "Annotator": annot,
                    "Accuracy": _fmt_pct_cell(
                        r["accuracy"],
                        r["accuracy_ci_low"],
                        r["accuracy_ci_high"],
                        r["p_mcnemar_accuracy"],
                    ),
                    "Precision": _fmt_pct_cell(
                        r["precision"],
                        r["ppv_ci_low"],
                        r["ppv_ci_high"],
                        r["p_mcnemar_ppv"],
                    ),
                    "Recall": _fmt_pct_cell(
                        r["recall"],
                        r["sensitivity_ci_low"],
                        r["sensitivity_ci_high"],
                        r["p_mcnemar_sensitivity"],
                    ),
                    "F1": _fmt_pct_cell(
                        f1_point,
                        f1_lo,
                        f1_hi,
                        np.nan,
                    ),
                }
            )

    df_elem_diag_table = pd.DataFrame(diag_rows)
    df_elem_class_table = pd.DataFrame(class_rows)

    df_elem_diag_table["Element"] = pd.Categorical(
        df_elem_diag_table["Element"], categories=element_order_list, ordered=True
    )
    df_elem_diag_table["Annotator"] = pd.Categorical(
        df_elem_diag_table["Annotator"], categories=annotator_order, ordered=True
    )
    df_elem_diag_table = df_elem_diag_table.sort_values(
        ["Element", "Annotator"]
    ).reset_index(drop=True)

    df_elem_class_table["Element"] = pd.Categorical(
        df_elem_class_table["Element"], categories=element_order_list, ordered=True
    )
    df_elem_class_table["Annotator"] = pd.Categorical(
        df_elem_class_table["Annotator"], categories=annotator_order, ordered=True
    )
    df_elem_class_table = df_elem_class_table.sort_values(
        ["Element", "Annotator"]
    ).reset_index(drop=True)

    gt_elem_diag = (
        GT(df_elem_diag_table, rowname_col="Annotator", groupname_col="Element")
        .tab_header(
            title="Element-Level Diagnostic Metrics",
            subtitle="LLM vs Human Oncologic Summaries (Humans as Judge)",
        )
        .tab_stubhead(label="Element")
        .tab_spanner(
            label="Diagnostic Metrics",
            columns=["Sensitivity", "Specificity", "PPV", "NPV"],
        )
        .fmt_markdown(columns=["Sensitivity", "Specificity", "PPV", "NPV"])
    )
    GT.save(
        gt_elem_diag,
        file=str(OUTPUT_DIR / "element_level_diagnostic_metrics_table.png"),
    )

    gt_elem_class = (
        GT(df_elem_class_table, rowname_col="Annotator", groupname_col="Element")
        .tab_header(
            title="Element-Level Classification Metrics",
            subtitle="LLM vs Human Oncologic Summaries (Humans as Judge)",
        )
        .tab_stubhead(label="Element")
        .tab_spanner(
            label="Classification Metrics",
            columns=["Accuracy", "Precision", "Recall", "F1"],
        )
        .fmt_markdown(columns=["Accuracy", "Precision", "Recall", "F1"])
    )
    GT.save(
        gt_elem_class,
        file=str(OUTPUT_DIR / "element_level_classification_metrics_table.png"),
    )
except Exception as exc:
    print(f"Failed generating element-level tables: {exc}")


# ============================================================
# Domain-level tables (diagnostic + classification)
# ============================================================

try:
    domain_diag_rows = []
    domain_class_rows = []

    for domain_name, _ in DOMAINS.items():
        df_dom = df_domain_agg[df_domain_agg["Domain"] == domain_name]
        if df_dom.empty:
            continue

        p_domain = df_elements[df_elements["Element"].isin(DOMAINS[domain_name])]
        p_map = {
            "accuracy": p_domain["p_mcnemar_accuracy"].min(),
            "sensitivity": p_domain["p_mcnemar_sensitivity"].min(),
            "specificity": p_domain["p_mcnemar_specificity"].min(),
            "ppv": p_domain["p_mcnemar_ppv"].min(),
            "npv": p_domain["p_mcnemar_npv"].min(),
        }

        for annot in ["Human", "AI"]:
            r = df_dom[df_dom["Annotator"] == annot].iloc[0]
            domain_diag_rows.append(
                {
                    "Domain": domain_name,
                    "Annotator": annot,
                    "Sensitivity": _fmt_cell(
                        r["sensitivity"],
                        r["sensitivity_ci_low"],
                        r["sensitivity_ci_high"],
                        p_map["sensitivity"],
                    ),
                    "Specificity": _fmt_cell(
                        r["specificity"],
                        r["specificity_ci_low"],
                        r["specificity_ci_high"],
                        p_map["specificity"],
                    ),
                    "PPV": _fmt_cell(
                        r["ppv"],
                        r["ppv_ci_low"],
                        r["ppv_ci_high"],
                        p_map["ppv"],
                    ),
                    "NPV": _fmt_cell(
                        r["npv"],
                        r["npv_ci_low"],
                        r["npv_ci_high"],
                        p_map["npv"],
                    ),
                }
            )

        df_dom_bal = df_domain_balanced[df_domain_balanced["Domain"] == domain_name]
        for annot in ["Human", "AI"]:
            r = df_dom_bal[df_dom_bal["Annotator"] == annot].iloc[0]
            domain_class_rows.append(
                {
                    "Domain": domain_name,
                    "Annotator": annot,
                    "Accuracy": _fmt_pct_cell(
                        r["accuracy"],
                        r["accuracy_ci_low"],
                        r["accuracy_ci_high"],
                        p_map["accuracy"],
                    ),
                    "Precision": _fmt_pct_cell(
                        r["precision"],
                        r["ppv_ci_low"],
                        r["ppv_ci_high"],
                        p_map["ppv"],
                    ),
                    "Recall": _fmt_pct_cell(
                        r["recall"],
                        r["sensitivity_ci_low"],
                        r["sensitivity_ci_high"],
                        p_map["sensitivity"],
                    ),
                    "F1": _fmt_pct_cell(
                        r["f1"],
                        r["f1_ci_low"],
                        r["f1_ci_high"],
                        np.nan,
                    ),
                }
            )

    df_domain_diag_table = pd.DataFrame(domain_diag_rows)
    df_domain_class_table = pd.DataFrame(domain_class_rows)

    df_domain_diag_table["Domain"] = pd.Categorical(
        df_domain_diag_table["Domain"], categories=list(DOMAINS.keys()), ordered=True
    )
    df_domain_diag_table["Annotator"] = pd.Categorical(
        df_domain_diag_table["Annotator"], categories=annotator_order, ordered=True
    )
    df_domain_diag_table = df_domain_diag_table.sort_values(
        ["Domain", "Annotator"]
    ).reset_index(drop=True)

    df_domain_class_table["Domain"] = pd.Categorical(
        df_domain_class_table["Domain"], categories=list(DOMAINS.keys()), ordered=True
    )
    df_domain_class_table["Annotator"] = pd.Categorical(
        df_domain_class_table["Annotator"], categories=annotator_order, ordered=True
    )
    df_domain_class_table = df_domain_class_table.sort_values(
        ["Domain", "Annotator"]
    ).reset_index(drop=True)

    gt_domain_diag = (
        GT(df_domain_diag_table, rowname_col="Annotator", groupname_col="Domain")
        .tab_header(
            title="Domain-Level Diagnostic Metrics",
            subtitle="LLM vs Human Oncologic Summaries (Humans as Judge)",
        )
        .tab_stubhead(label="Domain")
        .tab_spanner(
            label="Diagnostic Metrics",
            columns=["Sensitivity", "Specificity", "PPV", "NPV"],
        )
        .fmt_markdown(columns=["Sensitivity", "Specificity", "PPV", "NPV"])
    )
    GT.save(
        gt_domain_diag,
        file=str(OUTPUT_DIR / "domain_level_diagnostic_metrics_table.png"),
    )

    gt_domain_class = (
        GT(df_domain_class_table, rowname_col="Annotator", groupname_col="Domain")
        .tab_header(
            title="Domain-Level Classification Metrics",
            subtitle="LLM vs Human Oncologic Summaries (Humans as Judge)",
        )
        .tab_stubhead(label="Domain")
        .tab_spanner(
            label="Classification Metrics",
            columns=["Accuracy", "Precision", "Recall", "F1"],
        )
        .fmt_markdown(columns=["Accuracy", "Precision", "Recall", "F1"])
    )
    GT.save(
        gt_domain_class,
        file=str(OUTPUT_DIR / "domain_level_classification_metrics_table.png"),
    )
except Exception as exc:
    print(f"Failed generating domain-level tables: {exc}")


# ============================================================
# Element-level diagnostic metrics plot (facet)
# ============================================================

try:
    metrics = ["sensitivity", "specificity", "ppv", "npv"]
    metric_labels = {
        "sensitivity": "Sensitivity",
        "specificity": "Specificity",
        "ppv": "PPV",
        "npv": "NPV",
    }

    plot_rows = []
    for _, r in df_elements.iterrows():
        metric_map = {
            "sensitivity": (
                r["sensitivity"],
                r["sensitivity_ci_low"],
                r["sensitivity_ci_high"],
                r["p_mcnemar_sensitivity"],
            ),
            "specificity": (
                r["specificity"],
                r["specificity_ci_low"],
                r["specificity_ci_high"],
                r["p_mcnemar_specificity"],
            ),
            "ppv": (r["ppv"], r["ppv_ci_low"], r["ppv_ci_high"], r["p_mcnemar_ppv"]),
            "npv": (r["npv"], r["npv_ci_low"], r["npv_ci_high"], r["p_mcnemar_npv"]),
        }
        for metric, (val, lo, hi, pval) in metric_map.items():
            plot_rows.append(
                {
                    "Element": r["Element"],
                    "Annotator": r["Annotator"],
                    "Metric": metric,
                    "value": val,
                    "ci_low": lo,
                    "ci_high": hi,
                    "p_value": pval,
                }
            )

    df_diag_plot = pd.DataFrame(plot_rows)
    element_order = sorted(df_diag_plot["Element"].unique())

    n_metrics = len(metrics)
    n_cols = 2
    n_rows = ceil(n_metrics / n_cols)
    fig_height = 5.5 * n_rows
    fig_element, axes = plt.subplots(
        n_rows, n_cols, figsize=(20, fig_height), sharey=True
    )
    axes = np.atleast_1d(axes).flatten()
    bar_width = 0.38
    y_pad = 0.02
    cap = 3

    for ax_idx, metric in enumerate(metrics):
        ax = axes[ax_idx]
        df_metric = df_diag_plot[df_diag_plot["Metric"] == metric].copy()

        human = (
            df_metric[df_metric["Annotator"] == "Human"]
            .set_index("Element")
            .reindex(element_order)
            .reset_index()
        )
        ai = (
            df_metric[df_metric["Annotator"] == "AI"]
            .set_index("Element")
            .reindex(element_order)
            .reset_index()
        )

        x = np.arange(len(element_order))
        human_vals = human["value"].to_numpy()
        ai_vals = ai["value"].to_numpy()

        human_yerr = np.vstack(
            [
                np.maximum(human_vals - human["ci_low"].to_numpy(), 0),
                np.maximum(human["ci_high"].to_numpy() - human_vals, 0),
            ]
        )
        ai_yerr = np.vstack(
            [
                np.maximum(ai_vals - ai["ci_low"].to_numpy(), 0),
                np.maximum(ai["ci_high"].to_numpy() - ai_vals, 0),
            ]
        )

        ax.bar(
            x - bar_width / 2, human_vals, width=bar_width, label="Human", color="black"
        )
        ax.bar(x + bar_width / 2, ai_vals, width=bar_width, label="AI", color="gray")

        ax.errorbar(
            x - bar_width / 2,
            human_vals,
            yerr=human_yerr,
            fmt="none",
            ecolor="black",
            capsize=cap,
            linewidth=1,
        )
        ax.errorbar(
            x + bar_width / 2,
            ai_vals,
            yerr=ai_yerr,
            fmt="none",
            ecolor="black",
            capsize=cap,
            linewidth=1,
        )

        ax.set_title(metric_labels[metric], fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(element_order, rotation=45, ha="right", fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Value" if ax_idx % n_cols == 0 else "")
        ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.6)

        for i, element in enumerate(element_order):
            star = _sig_star(human.iloc[i]["p_value"])
            if star == "":
                continue
            if human_vals[i] >= ai_vals[i]:
                y = human_vals[i] + human_yerr[1, i] + y_pad
                x_pos = x[i] - bar_width / 2
            else:
                y = ai_vals[i] + ai_yerr[1, i] + y_pad
                x_pos = x[i] + bar_width / 2
            ax.text(x_pos, min(y, 1.08), star, ha="center", va="bottom", fontsize=12)

    for ax in axes[n_metrics:]:
        ax.remove()
    handles, labels = axes[0].get_legend_handles_labels()
    fig_element.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.98),
        fontsize=12,
    )
    fig_element.suptitle(
        "Element-Level Diagnostic Metrics: Human vs AI (with 95% CIs)",
        y=0.99,
        fontsize=16,
        fontweight="bold",
    )
    fig_element.tight_layout(rect=(0, 0, 1, 0.96))
    safe_save_plt(
        fig_element,
        OUTPUT_DIR / "element_level_diagnostic_metrics_human_vs_ai.png",
        dpi=600,
        bbox_inches="tight",
    )
    plt.close(fig_element)
except Exception as exc:
    print(f"Failed generating element-level diagnostic plots: {exc}")


# ============================================================
# Domain aggregated diagnostic metrics plot
# ============================================================

try:
    metric_labels = ["Sensitivity", "Specificity", "PPV", "NPV"]
    fig_domain, axes_domain = plt.subplots(
        1, len(metric_labels), figsize=(16, 5), sharey=True
    )
    bar_width = 0.38
    y_pad = 0.02
    cap = 3

    for ax_idx, metric in enumerate(metric_labels):
        ax = axes_domain[ax_idx]
        human = df_domain_agg[df_domain_agg["Annotator"] == "Human"].set_index("Domain")
        ai = df_domain_agg[df_domain_agg["Annotator"] == "AI"].set_index("Domain")

        domain_order = list(DOMAINS.keys())
        x = np.arange(len(domain_order))

        metric_col = metric.lower()

        human_vals = human[metric_col].reindex(domain_order).to_numpy()
        ai_vals = ai[metric_col].reindex(domain_order).to_numpy()

        human_lo = human[f"{metric_col}_ci_low"].reindex(domain_order).to_numpy()
        human_hi = human[f"{metric_col}_ci_high"].reindex(domain_order).to_numpy()
        ai_lo = ai[f"{metric_col}_ci_low"].reindex(domain_order).to_numpy()
        ai_hi = ai[f"{metric_col}_ci_high"].reindex(domain_order).to_numpy()

        human_yerr = np.vstack(
            [
                np.maximum(human_vals - human_lo, 0),
                np.maximum(human_hi - human_vals, 0),
            ]
        )
        ai_yerr = np.vstack(
            [
                np.maximum(ai_vals - ai_lo, 0),
                np.maximum(ai_hi - ai_vals, 0),
            ]
        )

        ax.bar(
            x - bar_width / 2, human_vals, width=bar_width, label="Human", color="black"
        )
        ax.bar(x + bar_width / 2, ai_vals, width=bar_width, label="AI", color="gray")

        ax.errorbar(
            x - bar_width / 2,
            human_vals,
            yerr=human_yerr,
            fmt="none",
            ecolor="black",
            capsize=cap,
            linewidth=1,
        )
        ax.errorbar(
            x + bar_width / 2,
            ai_vals,
            yerr=ai_yerr,
            fmt="none",
            ecolor="black",
            capsize=cap,
            linewidth=1,
        )

        ax.set_title(metric, fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(domain_order, rotation=45, ha="right", fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Value" if ax_idx == 0 else "")
        ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.6)

    handles, labels = axes_domain[0].get_legend_handles_labels()
    fig_domain.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.98),
        fontsize=12,
    )
    fig_domain.suptitle(
        "Domain-Aggregated Diagnostic Metrics: Human vs AI (with 95% CIs)",
        y=0.99,
        fontsize=16,
        fontweight="bold",
    )
    fig_domain.tight_layout(rect=(0, 0, 1, 0.95))
    safe_save_plt(
        fig_domain,
        OUTPUT_DIR / "domain_aggregated_diagnostic_metrics_human_vs_ai.png",
        dpi=600,
        bbox_inches="tight",
    )
    plt.close(fig_domain)
except Exception as exc:
    print(f"Failed generating domain aggregated diagnostic plot: {exc}")


# ============================================================
# Element-level classification metrics plot (facet)
# ============================================================

try:
    metrics = ["accuracy", "precision", "recall", "f1"]
    metric_titles = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1 score",
    }

    plot_rows = []
    for element_name, cols in ELEMENTS.items():
        source_col = cols["source"]
        human_col = cols["human"]
        ai_col = cols["ai"]
        if element_name not in df_elements["Element"].unique():
            continue

        d_el = data[[source_col, human_col, ai_col]].dropna()
        h_f1, h_f1_lo, h_f1_hi = _bootstrap_f1_ci_for_element(
            d_el, source_col, human_col
        )
        a_f1, a_f1_lo, a_f1_hi = _bootstrap_f1_ci_for_element(d_el, source_col, ai_col)
        f1_diff_lo, f1_diff_hi = _bootstrap_f1_diff_ci_for_element(
            d_el, source_col, human_col, ai_col
        )

        rows = df_elements[df_elements["Element"] == element_name]
        for annot in ["Human", "AI"]:
            r = rows[rows["Annotator"] == annot].iloc[0]
            plot_rows.extend(
                [
                    {
                        "Element": element_name,
                        "Annotator": annot,
                        "metric": "accuracy",
                        "value": r["accuracy"],
                        "ci_low": r["accuracy_ci_low"],
                        "ci_high": r["accuracy_ci_high"],
                        "p_value": r["p_mcnemar_accuracy"],
                    },
                    {
                        "Element": element_name,
                        "Annotator": annot,
                        "metric": "precision",
                        "value": r["precision"],
                        "ci_low": r["ppv_ci_low"],
                        "ci_high": r["ppv_ci_high"],
                        "p_value": r["p_mcnemar_ppv"],
                    },
                    {
                        "Element": element_name,
                        "Annotator": annot,
                        "metric": "recall",
                        "value": r["recall"],
                        "ci_low": r["sensitivity_ci_low"],
                        "ci_high": r["sensitivity_ci_high"],
                        "p_value": r["p_mcnemar_sensitivity"],
                    },
                ]
            )

        plot_rows.append(
            {
                "Element": element_name,
                "Annotator": "Human",
                "metric": "f1",
                "value": h_f1,
                "ci_low": h_f1_lo,
                "ci_high": h_f1_hi,
                "p_value": np.nan,
                "f1_diff_ci_low": f1_diff_lo,
                "f1_diff_ci_high": f1_diff_hi,
            }
        )
        plot_rows.append(
            {
                "Element": element_name,
                "Annotator": "AI",
                "metric": "f1",
                "value": a_f1,
                "ci_low": a_f1_lo,
                "ci_high": a_f1_hi,
                "p_value": np.nan,
                "f1_diff_ci_low": f1_diff_lo,
                "f1_diff_ci_high": f1_diff_hi,
            }
        )

    df_ci_plot = pd.DataFrame(plot_rows)
    n_metrics = len(metrics)
    n_cols = 2
    n_rows = ceil(n_metrics / n_cols)
    fig_height = 5.5 * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, fig_height), sharey=True)
    axes = np.atleast_1d(axes).flatten()
    bar_width = 0.38
    y_pad = 0.02
    cap = 3

    for ax_idx, metric in enumerate(metrics):
        ax = axes[ax_idx]
        order = element_order_list
        sub = df_ci_plot[df_ci_plot["metric"] == metric].copy()
        sub["Element"] = pd.Categorical(sub["Element"], categories=order, ordered=True)
        sub = sub.sort_values(["Element", "Annotator"])

        x = np.arange(len(order))
        human = sub[sub["Annotator"] == "Human"].set_index("Element").reindex(order)
        ai = sub[sub["Annotator"] == "AI"].set_index("Element").reindex(order)

        human_vals = human["value"].to_numpy()
        ai_vals = ai["value"].to_numpy()

        human_yerr = np.vstack(
            [
                np.maximum(human_vals - human["ci_low"].to_numpy(), 0),
                np.maximum(human["ci_high"].to_numpy() - human_vals, 0),
            ]
        )
        ai_yerr = np.vstack(
            [
                np.maximum(ai_vals - ai["ci_low"].to_numpy(), 0),
                np.maximum(ai["ci_high"].to_numpy() - ai_vals, 0),
            ]
        )

        ax.bar(
            x - bar_width / 2, human_vals, width=bar_width, label="Human", color="black"
        )
        ax.bar(x + bar_width / 2, ai_vals, width=bar_width, label="AI", color="gray")

        ax.errorbar(
            x - bar_width / 2,
            human_vals,
            yerr=human_yerr,
            fmt="none",
            ecolor="black",
            capsize=cap,
            linewidth=1,
        )
        ax.errorbar(
            x + bar_width / 2,
            ai_vals,
            yerr=ai_yerr,
            fmt="none",
            ecolor="black",
            capsize=cap,
            linewidth=1,
        )

        ax.set_title(metric_titles[metric])
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=35, ha="right")
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Value" if ax_idx % n_cols == 0 else "")
        ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.6)

        for i, element in enumerate(order):
            star = ""
            if metric == "f1":
                lo = human.iloc[i].get("f1_diff_ci_low", np.nan)
                hi = human.iloc[i].get("f1_diff_ci_high", np.nan)
                if pd.notna(lo) and pd.notna(hi) and ((lo > 0) or (hi < 0)):
                    star = "*"
            else:
                star = _sig_star(human.iloc[i]["p_value"])
            if not star:
                continue
            if human_vals[i] >= ai_vals[i]:
                y = human_vals[i] + human_yerr[1, i] + y_pad
                x_pos = x[i] - bar_width / 2
            else:
                y = ai_vals[i] + ai_yerr[1, i] + y_pad
                x_pos = x[i] + bar_width / 2
            ax.text(x_pos, min(y, 1.08), star, ha="center", va="bottom", fontsize=12)

    for ax in axes[n_metrics:]:
        ax.set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.suptitle("Human vs AI performance by element and metric (with 95% CIs)", y=1.06)
    fig.tight_layout()
    safe_save_plt(
        fig, OUTPUT_DIR / "human_ai_metrics_facet_ci.png", dpi=600, bbox_inches="tight"
    )
    plt.close(fig)
except Exception as exc:
    print(f"Failed generating element-level classification plot: {exc}")


# ============================================================
# Domain-level classification plot (aggregated)
# ============================================================

try:
    metric_labels = ["Accuracy", "Precision", "Recall", "F1"]
    x = np.arange(len(metric_labels))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax, domain_name in zip(axes, DOMAINS.keys()):
        d = df_domain_balanced[df_domain_balanced["Domain"] == domain_name]
        if d.empty:
            continue

        human = d[d["Annotator"] == "Human"].iloc[0]
        ai = d[d["Annotator"] == "AI"].iloc[0]

        human_vals = np.array(
            [
                human["accuracy"],
                human["precision"],
                human["recall"],
                human["f1"],
            ]
        )
        ai_vals = np.array(
            [
                ai["accuracy"],
                ai["precision"],
                ai["recall"],
                ai["f1"],
            ]
        )

        human_err = np.array(
            [
                [
                    human["accuracy"] - human["accuracy_ci_low"],
                    human["precision"] - human["ppv_ci_low"],
                    human["recall"] - human["sensitivity_ci_low"],
                    human["f1"] - human["f1_ci_low"],
                ],
                [
                    human["accuracy_ci_high"] - human["accuracy"],
                    human["ppv_ci_high"] - human["precision"],
                    human["sensitivity_ci_high"] - human["recall"],
                    human["f1_ci_high"] - human["f1"],
                ],
            ]
        )
        ai_err = np.array(
            [
                [
                    ai["accuracy"] - ai["accuracy_ci_low"],
                    ai["precision"] - ai["ppv_ci_low"],
                    ai["recall"] - ai["sensitivity_ci_low"],
                    ai["f1"] - ai["f1_ci_low"],
                ],
                [
                    ai["accuracy_ci_high"] - ai["accuracy"],
                    ai["ppv_ci_high"] - ai["precision"],
                    ai["sensitivity_ci_high"] - ai["recall"],
                    ai["f1_ci_high"] - ai["f1"],
                ],
            ]
        )

        ax.bar(x - width / 2, human_vals, width, label="Human", color="black")
        ax.bar(x + width / 2, ai_vals, width, label="AI", color="gray")

        ax.errorbar(
            x - width / 2,
            human_vals,
            yerr=human_err,
            fmt="none",
            ecolor="black",
            capsize=4,
        )
        ax.errorbar(
            x + width / 2,
            ai_vals,
            yerr=ai_err,
            fmt="none",
            ecolor="black",
            capsize=4,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, rotation=35, ha="right")
        ax.set_ylim(0, 1.1)
        ax.set_title(domain_name)
        ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.6)

    handles = [mpatches.Patch(color="black"), mpatches.Patch(color="gray")]
    labels = ["Human", "AI"]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.05),
    )
    fig.suptitle(
        "Average performance across elements by domain (95% bootstrap CIs across elements)"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    safe_save_plt(
        fig,
        OUTPUT_DIR / "avg_metrics_rad_vs_path_grouped_ci_stars.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
except Exception as exc:
    print(f"Failed generating domain classification plot: {exc}")


# ============================================================
# Fabrication rate tables and plots (separate from classification)
# ============================================================

fabrication_element_rows: list[dict] = []
valid_elements = [e for e in element_order_list if e in df_elements["Element"].unique()]

for element_name in valid_elements:
    cols = ELEMENTS.get(element_name)
    if not cols:
        continue
    required = [cols["source"], cols["human"], cols["ai"]]
    if not all(col in data.columns for col in required):
        continue

    element_subset = data[required]
    p_one_sided = _bootstrap_fabrication_diff_p_one_sided_element(
        element_subset,
        cols["source"],
        cols["human"],
        cols["ai"],
    )

    for annot, annot_col in [("Human", cols["human"]), ("AI", cols["ai"])]:
        row = df_elements[
            (df_elements["Element"] == element_name)
            & (df_elements["Annotator"] == annot)
        ]
        if row.empty:
            continue
        r = row.iloc[0]
        fabrication_element_rows.append(
            {
                "Element": element_name,
                "Annotator": annot,
                "TP": int(r["TP"]),
                "FP": int(r["FP"]),
                "fabrication_rate": r["fabrication_rate"],
                "fabrication_rate_ci_low": r["fabrication_rate_ci_low"],
                "fabrication_rate_ci_high": r["fabrication_rate_ci_high"],
                "p_one_sided_ai_ge_human": p_one_sided,
            }
        )

fabrication_element = pd.DataFrame(fabrication_element_rows)
safe_save_dataframe(
    fabrication_element, OUTPUT_DIR / "fabrication_rate_element_level.csv"
)

fabrication_aggregate_rows: list[dict] = []
all_elements = [e for e in ELEMENTS.keys() if e in df_elements["Element"].unique()]
agg_p = _bootstrap_fabrication_diff_p_one_sided_aggregate(data, all_elements)

for annot, annot_key in [("Human", "human"), ("AI", "ai")]:
    subset = df_elements[df_elements["Annotator"] == annot]
    if subset.empty:
        continue
    TP = int(subset["TP"].sum())
    FP = int(subset["FP"].sum())
    ci_low, ci_high = _bootstrap_fabrication_ci_aggregate(
        data,
        all_elements,
        annot_key,
    )
    fabrication_aggregate_rows.append(
        {
            "Level": "Overall",
            "Annotator": annot,
            "TP": TP,
            "FP": FP,
            "fabrication_rate": _fabrication_rate_from_counts(TP, FP),
            "fabrication_rate_ci_low": ci_low,
            "fabrication_rate_ci_high": ci_high,
            "p_one_sided_ai_ge_human": agg_p,
        }
    )

fabrication_aggregate = pd.DataFrame(fabrication_aggregate_rows)
safe_save_dataframe(
    fabrication_aggregate, OUTPUT_DIR / "fabrication_rate_aggregate.csv"
)

fabrication_domain_rows: list[dict] = []

for domain_name, element_list in DOMAINS.items():
    domain_elements = [e for e in element_list if e in df_elements["Element"].unique()]
    if not domain_elements:
        continue
    p_dom = _bootstrap_fabrication_diff_p_one_sided_aggregate(data, domain_elements)
    for annot, annot_key in [("Human", "human"), ("AI", "ai")]:
        dom_row = df_domain_agg[
            (df_domain_agg["Domain"] == domain_name)
            & (df_domain_agg["Annotator"] == annot)
        ]
        if dom_row.empty:
            continue
        r = dom_row.iloc[0]
        ci_low, ci_high = _bootstrap_fabrication_ci_aggregate(
            data,
            domain_elements,
            annot_key,
        )
        fabrication_domain_rows.append(
            {
                "Domain": domain_name,
                "Annotator": annot,
                "TP": int(r["TP"]),
                "FP": int(r["FP"]),
                "fabrication_rate": r["fabrication_rate"],
                "fabrication_rate_ci_low": ci_low,
                "fabrication_rate_ci_high": ci_high,
                "p_one_sided_ai_ge_human": p_dom,
            }
        )

fabrication_domain = pd.DataFrame(fabrication_domain_rows)
safe_save_dataframe(
    fabrication_domain, OUTPUT_DIR / "fabrication_rate_domain_aggregate.csv"
)

try:
    if not fabrication_element.empty:
        gt_elem_fab = (
            GT(fabrication_element, rowname_col="Annotator", groupname_col="Element")
            .tab_header(
                title="Fabrication Rate by Element",
                subtitle="Reported independently from classification metrics",
            )
            .fmt_percent(
                columns=[
                    "fabrication_rate",
                    "fabrication_rate_ci_low",
                    "fabrication_rate_ci_high",
                ],
                decimals=1,
            )
        )
        GT.save(
            gt_elem_fab, file=str(OUTPUT_DIR / "fabrication_rate_element_table.png")
        )

    if not fabrication_aggregate.empty:
        gt_agg_fab = (
            GT(fabrication_aggregate, rowname_col="Annotator", groupname_col="Level")
            .tab_header(
                title="Fabrication Rate (Aggregate)",
                subtitle="Overall Human vs AI",
            )
            .fmt_percent(
                columns=[
                    "fabrication_rate",
                    "fabrication_rate_ci_low",
                    "fabrication_rate_ci_high",
                ],
                decimals=1,
            )
        )
        GT.save(
            gt_agg_fab, file=str(OUTPUT_DIR / "fabrication_rate_aggregate_table.png")
        )

    if not fabrication_domain.empty:
        gt_dom_fab = (
            GT(fabrication_domain, rowname_col="Annotator", groupname_col="Domain")
            .tab_header(
                title="Fabrication Rate (Aggregate by Domain)",
                subtitle="Human vs AI",
            )
            .fmt_percent(
                columns=[
                    "fabrication_rate",
                    "fabrication_rate_ci_low",
                    "fabrication_rate_ci_high",
                ],
                decimals=1,
            )
        )
        GT.save(gt_dom_fab, file=str(OUTPUT_DIR / "fabrication_rate_domain_table.png"))
except Exception as exc:
    print(f"Fabrication rate tables skipped: {exc}")

try:
    if not fabrication_element.empty:
        order = [
            e
            for e in element_order_list
            if e in fabrication_element["Element"].unique()
        ]
        fig, ax = plt.subplots(figsize=(18, 7))
        human = (
            fabrication_element[fabrication_element["Annotator"] == "Human"]
            .set_index("Element")
            .reindex(order)
        )
        ai = (
            fabrication_element[fabrication_element["Annotator"] == "AI"]
            .set_index("Element")
            .reindex(order)
        )

        x = np.arange(len(order))
        width = 0.38
        h_vals = human["fabrication_rate"].to_numpy()
        a_vals = ai["fabrication_rate"].to_numpy()
        h_yerr = np.vstack(
            [
                np.maximum(h_vals - human["fabrication_rate_ci_low"].to_numpy(), 0),
                np.maximum(
                    human["fabrication_rate_ci_high"].to_numpy() - h_vals,
                    0,
                ),
            ]
        )
        a_yerr = np.vstack(
            [
                np.maximum(a_vals - ai["fabrication_rate_ci_low"].to_numpy(), 0),
                np.maximum(
                    ai["fabrication_rate_ci_high"].to_numpy() - a_vals,
                    0,
                ),
            ]
        )

        ax.bar(x - width / 2, h_vals, width=width, color="black", label="Human")
        ax.bar(x + width / 2, a_vals, width=width, color="gray", label="AI")
        ax.errorbar(
            x - width / 2,
            h_vals,
            yerr=h_yerr,
            fmt="none",
            ecolor="black",
            capsize=4,
        )
        ax.errorbar(
            x + width / 2,
            a_vals,
            yerr=a_yerr,
            fmt="none",
            ecolor="black",
            capsize=4,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=35, ha="right")
        ax.set_ylabel("Fabrication Rate")
        ax.set_title("Fabrication Rate by Element (Human vs AI)")
        ax.set_ylim(0, _fabrication_ylim(np.concatenate([h_vals, a_vals])))
        ax.grid(axis="y", linestyle=":", alpha=0.5)

        for i, element in enumerate(order):
            p_raw = human.loc[element, "p_one_sided_ai_ge_human"]
            p_conv = pd.to_numeric(p_raw, errors="coerce")
            p_val = float(p_conv) if pd.notna(p_conv) else np.nan
            star = _sig_star(p_val)
            if not star:
                continue
            top = max(h_vals[i] + h_yerr[1, i], a_vals[i] + a_yerr[1, i]) + 0.01
            ax.text(x[i], min(top, 1.08), star, ha="center", va="bottom", fontsize=12)

        ax.legend(frameon=False)
        fig.tight_layout()
        safe_save_plt(
            fig,
            OUTPUT_DIR / "fabrication_rate_element_plot.png",
            dpi=600,
            bbox_inches="tight",
        )
        plt.close(fig)

    if not fabrication_aggregate.empty or not fabrication_domain.empty:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        if not fabrication_aggregate.empty:
            ax0 = axes[0]
            positions = np.arange(len(fabrication_aggregate))
            colors = [
                "black" if annot == "Human" else "gray"
                for annot in fabrication_aggregate["Annotator"]
            ]
            agg_vals = fabrication_aggregate["fabrication_rate"].to_numpy()
            agg_ci_low = fabrication_aggregate["fabrication_rate_ci_low"].to_numpy()
            agg_ci_high = fabrication_aggregate["fabrication_rate_ci_high"].to_numpy()
            ax0.bar(positions, agg_vals, width=0.5, color=colors)
            ax0.errorbar(
                positions,
                agg_vals,
                yerr=np.vstack(
                    [
                        np.maximum(agg_vals - agg_ci_low, 0),
                        np.maximum(agg_ci_high - agg_vals, 0),
                    ]
                ),
                fmt="none",
                ecolor="black",
                capsize=4,
            )
            ax0.set_xticks(positions)
            ax0.set_xticklabels(fabrication_aggregate["Annotator"].tolist())
            ax0.set_ylabel("Fabrication Rate")
            ax0.set_title("Overall Aggregate")
            ax0.set_ylim(0, _fabrication_ylim(agg_vals))
            ax0.grid(axis="y", linestyle=":", alpha=0.5)
            agg_star = _sig_star(
                fabrication_aggregate["p_one_sided_ai_ge_human"].iloc[0]
            )
            if agg_star:
                y_text = float(np.nanmax(agg_ci_high)) + 0.01
                ax0.text(
                    0.5,
                    min(y_text, 1.08),
                    agg_star,
                    ha="center",
                    va="bottom",
                    fontsize=12,
                )
        else:
            axes[0].axis("off")

        if not fabrication_domain.empty:
            ax1 = axes[1]
            dom_order = [
                d for d in DOMAINS.keys() if d in fabrication_domain["Domain"].unique()
            ]
            dom_x = np.arange(len(dom_order))
            width = 0.38
            dh = (
                fabrication_domain[fabrication_domain["Annotator"] == "Human"]
                .set_index("Domain")
                .reindex(dom_order)
            )
            da = (
                fabrication_domain[fabrication_domain["Annotator"] == "AI"]
                .set_index("Domain")
                .reindex(dom_order)
            )
            dh_vals = dh["fabrication_rate"].to_numpy()
            da_vals = da["fabrication_rate"].to_numpy()
            dh_yerr = np.vstack(
                [
                    np.maximum(dh_vals - dh["fabrication_rate_ci_low"].to_numpy(), 0),
                    np.maximum(dh["fabrication_rate_ci_high"].to_numpy() - dh_vals, 0),
                ]
            )
            da_yerr = np.vstack(
                [
                    np.maximum(da_vals - da["fabrication_rate_ci_low"].to_numpy(), 0),
                    np.maximum(da["fabrication_rate_ci_high"].to_numpy() - da_vals, 0),
                ]
            )

            ax1.bar(
                dom_x - width / 2, dh_vals, width=width, color="black", label="Human"
            )
            ax1.bar(dom_x + width / 2, da_vals, width=width, color="gray", label="AI")
            ax1.errorbar(
                dom_x - width / 2,
                dh_vals,
                yerr=dh_yerr,
                fmt="none",
                ecolor="black",
                capsize=4,
            )
            ax1.errorbar(
                dom_x + width / 2,
                da_vals,
                yerr=da_yerr,
                fmt="none",
                ecolor="black",
                capsize=4,
            )
            ax1.set_xticks(dom_x)
            ax1.set_xticklabels(dom_order)
            ax1.set_title("Aggregate by Domain")
            ax1.set_ylim(0, _fabrication_ylim(np.concatenate([dh_vals, da_vals])))
            ax1.grid(axis="y", linestyle=":", alpha=0.5)

            for i, dom in enumerate(dom_order):
                p_raw = dh.loc[dom, "p_one_sided_ai_ge_human"]
                p_conv = pd.to_numeric(p_raw, errors="coerce")
                p_val = float(p_conv) if pd.notna(p_conv) else np.nan
                star = _sig_star(p_val)
                if not star:
                    continue
                y = max(dh_vals[i] + dh_yerr[1, i], da_vals[i] + da_yerr[1, i]) + 0.01
                ax1.text(
                    dom_x[i], min(y, 1.08), star, ha="center", va="bottom", fontsize=12
                )

            handles = [
                mpatches.Patch(color="black", label="Human"),
                mpatches.Patch(color="gray", label="AI"),
            ]
            fig.legend(
                handles,
                ["Human", "AI"],
                loc="upper center",
                ncol=2,
                frameon=False,
            )
        else:
            axes[1].axis("off")

        fig.suptitle("Fabrication Rate (Separate Reporting)")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        safe_save_plt(
            fig,
            OUTPUT_DIR / "fabrication_rate_aggregate_plots.png",
            dpi=600,
            bbox_inches="tight",
        )
        plt.close(fig)
except Exception as exc:
    print(f"Fabrication rate plotting skipped: {exc}")


# ============================================================
# Specificity pies
# ============================================================

try:
    if cm_human_df is not None:
        tn = float(
            pd.to_numeric(
                cm_human_df.loc["Predicted Negative", "True Negative"],
                errors="coerce",
            )
        )
        fp = float(
            pd.to_numeric(
                cm_human_df.loc["Predicted Positive", "True Negative"],
                errors="coerce",
            )
        )
        if tn + fp > 0:
            fig = plt.figure(figsize=(6, 4))
            plt.pie(
                [tn, fp],
                labels=["True Negative", "False Positive"],
                autopct="%1.1f%%",
                startangle=90,
            )
            plt.title(
                "Human Specificity: Proportion of Actual Negatives Correctly Identified"
            )
            plt.axis("equal")
            safe_save_plt(fig, OUTPUT_DIR / "specificity_human.png", dpi=300)
            plt.close(fig)

    if cm_ai_df is not None:
        tn = float(
            pd.to_numeric(
                cm_ai_df.loc["Predicted Negative", "True Negative"], errors="coerce"
            )
        )
        fp = float(
            pd.to_numeric(
                cm_ai_df.loc["Predicted Positive", "True Negative"], errors="coerce"
            )
        )
        if tn + fp > 0:
            fig = plt.figure(figsize=(6, 4))
            plt.pie(
                [tn, fp],
                labels=["True Negative", "False Positive"],
                autopct="%1.1f%%",
                startangle=90,
            )
            plt.title(
                "AI Specificity: Proportion of Actual Negatives Correctly Identified"
            )
            plt.axis("equal")
            safe_save_plt(fig, OUTPUT_DIR / "specificity_ai.png", dpi=300)
            plt.close(fig)
except Exception as exc:
    print(f"Failed generating specificity pies: {exc}")


# ============================================================
# Synthetic ROC/PR, error, bias-variance, CV plots
# ============================================================

try:
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 1000)
    y_scores = np.random.rand(1000)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    fig = plt.figure(figsize=(8, 6))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    safe_save_plt(fig, OUTPUT_DIR / "roc_example.png", dpi=300)
    plt.close(fig)

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    fig = plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="blue", lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    safe_save_plt(fig, OUTPUT_DIR / "pr_example.png", dpi=300)
    plt.close(fig)
except Exception as exc:
    print(f"Failed generating ROC/PR examples: {exc}")

try:
    np.random.seed(42)
    features = np.random.rand(100, 3)
    true_labels = np.random.choice(["A", "B", "C"], 100)
    predicted_labels = np.random.choice(["A", "B", "C"], 100)
    df_err = pd.DataFrame(features, columns=["Feature 1", "Feature 2", "Feature 3"])
    df_err["True Label"] = true_labels
    df_err["Predicted Label"] = predicted_labels
    df_err["Error"] = df_err["True Label"] != df_err["Predicted Label"]
    fig = plt.figure(figsize=(10, 6))
    plt.scatter(
        df_err[~df_err["Error"]]["Feature 1"],
        df_err[~df_err["Error"]]["Feature 2"],
        c="green",
        label="Correct",
        alpha=0.7,
    )
    plt.scatter(
        df_err[df_err["Error"]]["Feature 1"],
        df_err[df_err["Error"]]["Feature 2"],
        c="red",
        label="Error",
        alpha=0.7,
    )
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Error Distribution in Feature Space")
    plt.legend()
    safe_save_plt(fig, OUTPUT_DIR / "error_scatter.png", dpi=300)
    plt.close(fig)
except Exception as exc:
    print(f"Failed generating error analysis plots: {exc}")

try:
    np.random.seed(42)
    X = np.linspace(0, 10, 100)
    y = 3 * X + 2 + np.random.normal(0, 2, 100)
    degrees = [1, 3, 15]
    fig = plt.figure(figsize=(12, 4))
    for i, degree in enumerate(degrees):
        plt.subplot(1, 3, i + 1)
        coeffs = np.polyfit(X, y, degree)
        y_pred = np.polyval(coeffs, X)
        plt.scatter(X, y, alpha=0.6)
        plt.plot(X, y_pred, color="r")
        plt.title(f"Degree {degree} Polynomial")
        plt.xlabel("X")
        plt.ylabel("y")
    plt.tight_layout()
    safe_save_plt(fig, OUTPUT_DIR / "bias_variance_demo.png", dpi=300)
    plt.close(fig)
except Exception as exc:
    print(f"Failed generating bias-variance demo: {exc}")

try:
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification

    X_cv, y_cv = make_classification(
        n_samples=1000, n_features=20, n_classes=2, random_state=42
    )
    model = LogisticRegression(max_iter=1000)
    cv_scores = cross_val_score(model, X_cv, y_cv, cv=5)

    fig = plt.figure(figsize=(8, 6))
    plt.boxplot(cv_scores)
    plt.title("5-Fold Cross-Validation Results")
    plt.ylabel("Accuracy")
    safe_save_plt(fig, OUTPUT_DIR / "cv_boxplot.png", dpi=300)
    plt.close(fig)
except Exception as exc:
    print(f"Failed generating cross-validation example: {exc}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  📊 Plots (PNG):")
    print("     - confusion_tables.png")
    print("     - confusion_heatmaps.png")
    print("     - human_ai_metrics_facet_ci.png")
    print("     - element_level_diagnostic_metrics_human_vs_ai.png")
    print("     - domain_aggregated_diagnostic_metrics_human_vs_ai.png")
    print("     - avg_metrics_rad_vs_path_grouped_ci_stars.png")
    print("     - fabrication_rate_element_plot.png")
    print("     - fabrication_rate_aggregate_plots.png")
    print("     - specificity_human.png")
    print("     - specificity_ai.png")
    print("     - roc_example.png")
    print("     - pr_example.png")
    print("     - error_scatter.png")
    print("     - bias_variance_demo.png")
    print("     - cv_boxplot.png")
    print("  📋 Tables (PNG):")
    print("     - element_level_non_null_null_counts.png")
    print("     - element_level_diagnostic_metrics_table.png")
    print("     - element_level_classification_metrics_table.png")
    print("     - domain_level_diagnostic_metrics_table.png")
    print("     - domain_level_classification_metrics_table.png")
    print("     - fabrication_rate_element_table.png")
    print("     - fabrication_rate_aggregate_table.png")
    print("     - fabrication_rate_domain_table.png")
    print("  📄 Data (CSV):")
    print("     - element_level_metrics.csv")
    print("     - fabrication_rate_debug_counts.csv")
    print("     - confusion_human.csv")
    print("     - confusion_ai.csv")
    print("     - domain_level_aggregated_metrics.csv")
    print("     - domain_level_element_balanced_metrics.csv")
    print("     - element_level_summary_wide.csv")
    print("     - diagnostic_tests_with_p.csv")
    print("     - domain_agg_metrics_with_p.csv")
    print("     - domain_class_table.csv")
    print("     - fabrication_rate_element_level.csv")
    print("     - fabrication_rate_aggregate.csv")
    print("     - fabrication_rate_domain_aggregate.csv")
    print("\n" + "=" * 70)
