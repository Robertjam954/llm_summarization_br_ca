"""
main_analysis.py

Element-level and domain-level diagnostic + classification metrics
for Human vs AI, using canonical TP/FP/FN/TN definitions.

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

import numpy as np
import pandas as pd

from metrics_utils import (
    compute_confusion_counts,
    compute_metrics_from_counts,
    wilson_ci,
    bootstrap_ci,
    element_metric_pvalue,
    binary_predictions_from_annotator,
    roc_pr_from_binary,
    build_confusion_df_from_counts,
    plot_confusion_heatmap,
    safe_save_dataframe,
    safe_save_plt,
    format_metric_with_ci,
)

# ============================================================
# Paths and data loading
# ============================================================

PROJECT_ROOT = Path("/Users/robertjames/Documents/llm_summarization")
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "merged_llm_summary_validation_datasheet_deidentified.xlsx"
)
OUTPUT_DIR = PROJECT_ROOT / "data reports"

if not PROJECT_ROOT.exists():
    print(f"Warning: expected project directory does not exist: {PROJECT_ROOT}")
else:
    try:
        os.chdir(PROJECT_ROOT)
        print(f"Set working directory to project root: {PROJECT_ROOT}")
    except Exception as exc:
        print(f"Warning: failed to set working directory to {PROJECT_ROOT}: {exc}")

if not OUTPUT_DIR.exists():
    print(f"Warning: OUTPUT_DIR does not exist and saves may fail: {OUTPUT_DIR}")

data = pd.read_excel(DATA_PATH)

# Normalize NA-like strings to "N/A"
string_data = ["NA", "na", "n/a", "N/A", "NA ", " na", " n/a", " N/A"]
data = data.replace(string_data, "N/A")


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
        "source": "lesion_laterality_status_source",
        "human": "lesion_laterality_status_human",
        "ai": "lesion_laterality_status_ai",
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


# ============================================================
# Element-level metrics
# ============================================================

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

    # CIs (Wilson for accuracy, sensitivity, specificity, PPV, NPV)
    # Human
    acc_ci_h = wilson_ci(
        x=human_counts["TP"] + human_counts["TN"],
        n=human_total,
    )
    sens_ci_h = (
        wilson_ci(
            x=human_counts["TP"],
            n=human_counts["TP"] + human_counts["FN"],
        )
        if (human_counts["TP"] + human_counts["FN"]) > 0
        else (np.nan, np.nan)
    )
    spec_ci_h = (
        wilson_ci(
            x=human_counts["TN"],
            n=human_counts["TN"] + human_counts["FP"],
        )
        if (human_counts["TN"] + human_counts["FP"]) > 0
        else (np.nan, np.nan)
    )
    ppv_ci_h = (
        wilson_ci(
            x=human_counts["TP"],
            n=human_counts["TP"] + human_counts["FP"],
        )
        if (human_counts["TP"] + human_counts["FP"]) > 0
        else (np.nan, np.nan)
    )
    npv_ci_h = (
        wilson_ci(
            x=human_counts["TN"],
            n=human_counts["TN"] + human_counts["FN"],
        )
        if (human_counts["TN"] + human_counts["FN"]) > 0
        else (np.nan, np.nan)
    )

    # AI
    acc_ci_ai = wilson_ci(
        x=ai_counts["TP"] + ai_counts["TN"],
        n=ai_total,
    )
    sens_ci_ai = (
        wilson_ci(
            x=ai_counts["TP"],
            n=ai_counts["TP"] + ai_counts["FN"],
        )
        if (ai_counts["TP"] + ai_counts["FN"]) > 0
        else (np.nan, np.nan)
    )
    spec_ci_ai = (
        wilson_ci(
            x=ai_counts["TN"],
            n=ai_counts["TN"] + ai_counts["FP"],
        )
        if (ai_counts["TN"] + ai_counts["FP"]) > 0
        else (np.nan, np.nan)
    )
    ppv_ci_ai = (
        wilson_ci(
            x=ai_counts["TP"],
            n=ai_counts["TP"] + ai_counts["FP"],
        )
        if (ai_counts["TP"] + ai_counts["FP"]) > 0
        else (np.nan, np.nan)
    )
    npv_ci_ai = (
        wilson_ci(
            x=ai_counts["TN"],
            n=ai_counts["TN"] + ai_counts["FN"],
        )
        if (ai_counts["TN"] + ai_counts["FN"]) > 0
        else (np.nan, np.nan)
    )

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


# ============================================================
# Element-level confusion matrices (aggregated by annotator)
# ============================================================

agg_counts = df_elements.groupby("Annotator")[["TP", "FP", "FN", "TN"]].sum()

cm_human_df = None
cm_ai_df = None

if "Human" in agg_counts.index:
    c = agg_counts.loc["Human"]
    cm_human_df = build_confusion_df_from_counts(
        TP=int(c["TP"]), FP=int(c["FP"]), FN=int(c["FN"]), TN=int(c["TN"])
    )
    safe_save_dataframe(cm_human_df.reset_index(), OUTPUT_DIR / "confusion_human.csv")

if "AI" in agg_counts.index:
    c = agg_counts.loc["AI"]
    cm_ai_df = build_confusion_df_from_counts(
        TP=int(c["TP"]), FP=int(c["FP"]), FN=int(c["FN"]), TN=int(c["TN"])
    )
    safe_save_dataframe(cm_ai_df.reset_index(), OUTPUT_DIR / "confusion_ai.csv")

# Side-by-side confusion heatmaps
if cm_human_df is not None and cm_ai_df is not None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    # Human
    fig_h = plot_confusion_heatmap(cm_human_df, "Human - Confusion Heatmap")
    axes[0].imshow(fig_h.axes[0].images[0].get_array())
    axes[0].set_axis_off()
    axes[0].set_title("Human - Confusion Heatmap")
    plt.close(fig_h)

    # AI
    fig_ai = plot_confusion_heatmap(cm_ai_df, "AI - Confusion Heatmap")
    axes[1].imshow(fig_ai.axes[0].images[0].get_array())
    axes[1].set_axis_off()
    axes[1].set_title("AI - Confusion Heatmap")
    plt.close(fig_ai)

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

        acc_ci = wilson_ci(TP + TN, total) if total > 0 else (np.nan, np.nan)
        sens_ci = wilson_ci(TP, TP + FN) if (TP + FN) > 0 else (np.nan, np.nan)
        spec_ci = wilson_ci(TN, TN + FP) if (TN + FP) > 0 else (np.nan, np.nan)
        ppv_ci = wilson_ci(TP, TP + FP) if (TP + FP) > 0 else (np.nan, np.nan)
        npv_ci = wilson_ci(TN, TN + FN) if (TN + FN) > 0 else (np.nan, np.nan)

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
        }

        # Bootstrap CIs across elements
        acc_ci = bootstrap_ci(df_b["accuracy"].to_numpy())
        sens_ci = bootstrap_ci(df_b["sensitivity"].to_numpy())
        spec_ci = bootstrap_ci(df_b["specificity"].to_numpy())
        ppv_ci = bootstrap_ci(df_b["ppv"].to_numpy())
        npv_ci = bootstrap_ci(df_b["npv"].to_numpy())
        f1_ci = bootstrap_ci(df_b["f1"].to_numpy())

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


if __name__ == "__main__":
    print("Analysis complete.")
    print(f"Element-level metrics: {OUTPUT_DIR / 'element_level_metrics.csv'}")
    print(
        f"Domain-level aggregated metrics: {OUTPUT_DIR / 'domain_level_aggregated_metrics.csv'}"
    )
    print(
        f"Domain-level element-balanced metrics: {OUTPUT_DIR / 'domain_level_element_balanced_metrics.csv'}"
    )
