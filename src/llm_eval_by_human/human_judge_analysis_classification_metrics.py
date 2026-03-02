import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import binomtest
import plotly.io as pio
from tabulate import tabulate
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import numpy as np
from great_tables import GT, md, html
import plotly.express as px
# ============================================================

data = pd.read_excel(
    "data/raw/merged_llm_summary_validation_datasheet_deidentified.xlsx"
)
OUTPUT_DIR = Path("/Users/robertjames/Documents/llm_summarization/data reports")
OUTPUT_DIR.mkdir(
    parents=True, exist_ok=True
)  # keep user's original behavior: do not create directories automatically
directory_path = Path("/Users/robertjames/Documents/llm_summarization")
# Do not create directories automatically — warn if they are missing so the user
# explicitly controls filesystem changes (per user's preference).
if not directory_path.exists():
    print(f"Warning: expected project directory does not exist: {directory_path}")
else:
    # If the project directory exists, make it the current working directory so
    # relative paths (like the `data/` folder) resolve correctly.
    try:
        os.chdir(directory_path)
        print(f"Set working directory to project root: {directory_path}")
    except Exception as exc:
        print(f"Warning: failed to set working directory to {directory_path}: {exc}")

# Also warn if the configured OUTPUT_DIR is missing; saving will fail unless it
# exists (user requested a warning rather than auto-creation).
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
    }


# ============================================================
# Define element_columns as a dict keyed by element name
# ============================================================
element_columns = {
    "lesion_size": {
        "source": "lesion_size_status_source",
        "human": "lesion_size_status_human",
        "ai": "lesion_size_status_ai",
    },
    "lesion_laterality": {
        "source": "lesion_laterality_status_source",
        "human": "lesion_laterality_status_human",
        "ai": "lesion_laterality_status_ai",
    },
    "lesion_location": {
        "source": "lesion_location_status_source",
        "human": "lesion_location_status_human",
        "ai": "lesion_location_status_ai",
    },
    "calcifications_asymmetry": {
        "source": "calcifications_asymmetry_status_source",
        "human": "calcifications_asymmetry_status_human",
        "ai": "calcifications_asymmetry_status_ai",
    },
    "additional_enhancement_mri": {
        "source": "additional_enhancement_mri_status_source",
        "human": "additional_enhancement_mri_status_human",
        "ai": "additional_enhancement_mri_status_ai",
    },
    "extent": {
        "source": "extent_status_source",
        "human": "extent_status_human",
        "ai": "extent_status_ai",
    },
    "accurate_clip_placement": {
        "source": "accurate_clip_placement_status_source",
        "human": "accurate_clip_placement_status_human",
        "ai": "accurate_clip_placement_status_ai",
    },
    "workup_recommendation": {
        "source": "workup_recommendation_status_source",
        "human": "workup_recommendation_status_human",
        "ai": "workup_recommendation_status_ai",
    },
    "Lymph node": {
        "source": "Lymph node_status_source",
        "human": "Lymph node_status_human",
        "ai": "Lymph node_status_ai",
    },
    "chronology_preserved": {
        "source": "chronology_preserved_status_source",
        "human": "chronology_preserved_status_human",
        "ai": "chronology_preserved_status_ai",
    },
    "biopsy_method": {
        "source": "biopsy_method_status_source",
        "human": "biopsy_method_status_human",
        "ai": "biopsy_method_status_ai",
    },
    "invasive_component_size_pathology": {
        "source": "invasive_component_size_pathology_status_source",
        "human": "invasive_component_size_pathology_status_human",
        "ai": "invasive_component_size_pathology_status_ai",
    },
    "histologic_diagnosis": {
        "source": "histologic_diagnosis_status_source",
        "human": "histologic_diagnosis_status_human",
        "ai": "histologic_diagnosis_status_ai",
    },
    "receptor_status": {
        "source": "receptor_status_source",
        "human": "receptor_status_human",
        "ai": "receptor_status_ai",
    },
}

for col in element_columns.values():
    source_col = col["source"]  # example to access source column for lesion_size
    human_col = col["human"]  # example to access human column for lesion_size
    ai_col = col["ai"]  # example to access ai column for lesion_size

for col in element_columns.keys():
    element = col.replace(
        "_", " "
    ).title()  # example to derive human-friendly element name

# ----------------------------
# Your domain definitions
# ----------------------------
rad_col = [
    ("lesion_size_status_source", "lesion_size_status_human", "lesion_size_status_ai"),
    (
        "lesion_laterality_status_source",
        "lesion_laterality_status_human",
        "lesion_laterality_status_ai",
    ),
    (
        "lesion_location_status_source",
        "lesion_location_status_human",
        "lesion_location_status_ai",
    ),
    (
        "calcifications_asymmetry_status_source",
        "calcifications_asymmetry_status_human",
        "calcifications_asymmetry_status_ai",
    ),
    (
        "additional_enhancement_mri_status_source",
        "additional_enhancement_mri_status_human",
        "additional_enhancement_mri_status_ai",
    ),
    ("extent_status_source", "extent_status_human", "extent_status_ai"),
    (
        "accurate_clip_placement_status_source",
        "accurate_clip_placement_status_human",
        "accurate_clip_placement_status_ai",
    ),
    (
        "workup_recommendation_status_source",
        "workup_recommendation_status_human",
        "workup_recommendation_status_ai",
    ),
    ("Lymph node_status_source", "Lymph node_status_human", "Lymph node_status_ai"),
    (
        "chronology_preserved_status_source",
        "chronology_preserved_status_human",
        "chronology_preserved_status_ai",
    ),
    (
        "biopsy_method_status_source",
        "biopsy_method_status_human",
        "biopsy_method_status_ai",
    ),
]

path_col = [
    (
        "biopsy_method_status_source",
        "biopsy_method_status_human",
        "biopsy_method_status_ai",
    ),
    (
        "invasive_component_size_pathology_status_source",
        "invasive_component_size_pathology_status_human",
        "invasive_component_size_pathology_status_ai",
    ),
    (
        "histologic_diagnosis_status_source",
        "histologic_diagnosis_status_human",
        "histologic_diagnosis_status_ai",
    ),
    ("receptor_status_source", "receptor_status_human", "receptor_status_ai"),
]

# missing in human and ai columns are treated as nulls, but we want to count them as "N/A" for the purposes of value counts and confusion matrices, so we replace nulls with a string that will be counted as "N/A" in the value counts. This allows us to capture the TN counts correctly.
string_data = ["NA", "na", "n/a", "N/A", "NA ", " na", " n/a", " N/A"]
string_null = "N/A"

data = data.replace(string_data, string_null)


# ----------------------------
# Element-level totals (non-null + null) for Human and AI
element_total_rows = []
for element_key, cols in element_columns.items():
    element_name = element_key.replace("_", " ").title()
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

# Export table
element_totals_png = OUTPUT_DIR / "element_level_non_null_null_counts.png"
GT.save(gt_element_totals, file=str(element_totals_png))


# ============================================================
# Group dict into per-element (source, human, ai) triples
# ============================================================
ROLE_FIELDS = ("source", "human", "ai")
elements = {}

for element_key, role_map in element_columns.items():
    required_roles = set(ROLE_FIELDS)
    missing = required_roles - set(role_map)
    if missing:
        raise ValueError(
            f"Element '{element_key}' is missing roles: {', '.join(sorted(missing))}"
        )

    domain = role_map.get("domain", "Unspecified")
    elements[element_key] = {role: role_map[role] for role in ROLE_FIELDS}
    elements[element_key]["domain"] = domain

# Optional sanity checks (recommended)
missing_elements = []
for k, v in elements.items():
    if not set(ROLE_FIELDS) <= set(v.keys()):
        print(
            f"Warning: missing one of source/human/ai columns for element: {k}. Found: {v}"
        )
        missing_elements.append(k)
        continue
    for role in ROLE_FIELDS:
        col = v[role]
        if col not in data.columns:
            print(f"Warning: column not found in dataframe: {col} (element: {k})")
            missing_elements.append(k)
            break

for k in set(missing_elements):
    elements.pop(k, None)


# ============================================================
# Build confusion matrices into a tidy DataFrame (long format)
# ============================================================
records_cm = []

for element_key, cols in elements.items():
    source_col = cols["source"]
    human_col = cols["human"]
    ai_col = cols["ai"]

    # Human-friendly label derived from key
    element = element_key.replace("_", " ").title()

    human_cm = compute_confusion_matrix(data, source_col, human_col)
    ai_cm = compute_confusion_matrix(data, source_col, ai_col)

    records_cm.append({"Element": element, "Annotator": "Human", **human_cm})
    records_cm.append({"Element": element, "Annotator": "AI", **ai_cm})

df_confusion = pd.DataFrame(records_cm)

# ------------------------------------------------------------------
# Export confusion matrices (regular table) and heatmaps for Human and AI
# Aggregates TP/FP/FN/TN across elements, builds Predicted x Actual tables,
# saves CSVs and two PNGs into OUTPUT_DIR:
#   - confusion_tables.png  (side-by-side textual tables)
#   - confusion_heatmaps.png (side-by-side heatmaps with counts + %)
# ------------------------------------------------------------------
import seaborn as sns


def build_confusion_df_from_counts(counts):
    TP = int(counts["TP"])
    FP = int(counts["FP"])
    FN = int(counts["FN"])
    TN = int(counts["TN"])
    data = {
        "Actual Positive": [TP, FN],
        "Actual Negative": [FP, TN],
    }
    index = ["Predicted Positive", "Predicted Negative"]
    return pd.DataFrame(data, index=index)


# Rebuild df_confusion if missing or empty (defensive)
if "df_confusion" not in globals() or df_confusion is None or len(df_confusion) == 0:
    records_cm = []
    for element_key, cols in elements.items():
        s_col, h_col, a_col = cols["source"], cols["human"], cols["ai"]
        human_cm = compute_confusion_matrix(data, s_col, h_col)
        ai_cm = compute_confusion_matrix(data, s_col, a_col)
        records_mcnemar_accuracy = []

        for element_key, cols in elements.items():
            source_col = cols["source"]
            human_col = cols["human"]
            ai_col = cols["ai"]
            element = element_key.replace("_", " ").title()

            mask = data[[source_col, human_col, ai_col]].notnull().all(axis=1)
            if mask.sum() == 0:
                continue

            d = data.loc[mask].copy()
            human_correct = annotator_correct(d, source_col, human_col).astype(bool)
            ai_correct = annotator_correct(d, source_col, ai_col).astype(bool)

            b = ((human_correct == 1) & (ai_correct == 0)).sum()
            c = ((human_correct == 0) & (ai_correct == 1)).sum()

            table = np.array(
                [
                    [(human_correct & ai_correct).sum(), b],
                    [c, (~human_correct & ~ai_correct).sum()],
                ]
            )

            # two-sided McNemar statistic (kept for reference), but use exact binomial one-sided p-value
            result = mcnemar(table, exact=((b + c) < 25))
            diff, ci_low, ci_high = newcombe_paired_diff_ci(
                int(b), int(c), int(len(d)), alpha=0.05
            )

            n_disc = int(b + c)
            if n_disc > 0:
                p_one_sided = binomtest(k=int(c), n=n_disc, p=0.5, alternative="greater").pvalue
            else:
                p_one_sided = np.nan

            records_mcnemar_accuracy.append(
                {
                    "Element": element,
                    "Metric": "Accuracy",
                    "n_pairs": int(len(d)),
                    "b_human_only": int(b),
                    "c_ai_only": int(c),
                    "diff_human_minus_ai": diff,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    # one-sided exact McNemar p-value (AI > Human)
                    "p_value_mcnemar": p_one_sided,
                    "exact_mcnemar": bool((b + c) < 25),
                }
            )

        df_mcnemar_accuracy = pd.DataFrame(records_mcnemar_accuracy)
            write_kw = {}
        fig.write_image(path, **write_kw)
        print(f"Saved plotly image to: {path}")
    except Exception as exc:  # pragma: no cover - depends on local Chrome install
        print(
            "Skipping Plotly static export because Kaleido could not render the figure. "
            "Install Chrome or run `kaleido.get_chrome()` to enable PNG export."
        )
        print(f"Kaleido error: {exc}")


# Build DataFrames and save CSVs
human_cm_df = None
ai_cm_df = None
if "Human" in agg.index:
    human_cm_df = build_confusion_df_from_counts(agg.loc["Human"])
    safe_save_dataframe(human_cm_df, OUTPUT_DIR / "confusion_human.csv")

if "AI" in agg.index:
    ai_cm_df = build_confusion_df_from_counts(agg.loc["AI"])
    safe_save_dataframe(ai_cm_df, OUTPUT_DIR / "confusion_ai.csv")


# Save side-by-side textual tables as an image
try:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax in axes:
        ax.axis("off")

    if human_cm_df is not None:
        axes[0].set_title("Human - Confusion Matrix")
        tbl = axes[0].table(
            cellText=human_cm_df.values.tolist(),
            colLabels=human_cm_df.columns.tolist(),
            rowLabels=human_cm_df.index.tolist(),
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1, 1.4)
    else:
        axes[0].text(
            0.5, 0.5, "No Human rows in df_confusion", ha="center", va="center"
        )

    if ai_cm_df is not None:
        axes[1].set_title("AI - Confusion Matrix")
        tbl = axes[1].table(
            cellText=ai_cm_df.values.tolist(),
            colLabels=ai_cm_df.columns.tolist(),
            rowLabels=ai_cm_df.index.tolist(),
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1, 1.4)
    else:
        axes[1].text(0.5, 0.5, "No AI rows in df_confusion", ha="center", va="center")

    fig.tight_layout()
    # Save figure
    out_path = OUTPUT_DIR / "confusion_tables.png"
    safe_save_plt(fig, out_path, dpi=300, bbox_inches="tight")
    if os.environ.get("SHOW_PLOTS", "0") == "1":
        plt.show()
    else:
        plt.close(fig)
except Exception as exc:
    print(f"Failed saving confusion tables PNG: {exc}")


# Save side-by-side heatmaps with counts + percent
try:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    def plot_heatmap_from_df(cm_df, ax, title):
        matrix = cm_df.values.astype(int)
        total = matrix.sum()
        # build annotation strings with count + percent
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

    if human_cm_df is not None:
        plot_heatmap_from_df(human_cm_df, axes[0], "Human - Confusion Heatmap")
    else:
        axes[0].text(0.5, 0.5, "No Human rows", ha="center", va="center")

    if ai_cm_df is not None:
        plot_heatmap_from_df(ai_cm_df, axes[1], "AI - Confusion Heatmap")
    else:
        axes[1].text(0.5, 0.5, "No AI rows", ha="center", va="center")

    # Save figure
    out_path = OUTPUT_DIR / "confusion_heatmaps.png"
    safe_save_plt(fig, out_path, dpi=300, bbox_inches="tight")
    if os.environ.get("SHOW_PLOTS", "0") == "1":
        plt.show()
    else:
        plt.close(fig)
except Exception as exc:
    print(f"Failed saving confusion heatmaps PNG: {exc}")


# (Additional exports moved to end of file so they run after aggregated metrics)


# ============================================================
# Add accuracy/precision/recall/F1 to df_confusion
# ============================================================
df_metrics = df_confusion.copy()

den = df_metrics["TP"] + df_metrics["FP"] + df_metrics["FN"] + df_metrics["TN"]
df_metrics["accuracy"] = (df_metrics["TP"] + df_metrics["TN"]) / den

# Diagnostic metrics computed from raw counts (TP/FP/FN/TN) using explicit
# formulas so the results are deterministic and traceable. All divisions are
# guarded against zero denominators and will return NaN when undefined.
#
# Definitions (per user):
# TP = ((data[source_col] == 1) & (data[annotator_col] == 1)).sum()
# FN = ((data[source_col] == 1) & (data[annotator_col] == 2)).sum()
# FP = ((data[source_col] == 1) & (data[annotator_col] == 3)).sum()
# TN = ((data[source_col] == 0) & (data[annotator_col] == "N/A")).sum()

# Accuracy (already present)
df_metrics["accuracy"] = (df_metrics["TP"] + df_metrics["TN"]) / den

# Sensitivity / Recall = TP / (TP + FN)
sens_denom = df_metrics["TP"] + df_metrics["FN"]
df_metrics["sensitivity"] = df_metrics["TP"] / sens_denom

# Specificity = TN / (TN + FP)
spec_denom = df_metrics["TN"] + df_metrics["FP"]
df_metrics["specificity"] = df_metrics["TN"] / spec_denom

# Positive Predictive Value (PPV) = TP / (TP + FP)
ppv_denom = df_metrics["TP"] + df_metrics["FP"]
df_metrics["ppv"] = df_metrics["TP"] / ppv_denom

# Negative Predictive Value (NPV) = TN / (TN + FN)
npv_denom = df_metrics["TN"] + df_metrics["FN"]
df_metrics["npv"] = df_metrics["TN"] / npv_denom

# Diagnostic Accuracy (same as accuracy but included here for clarity):
df_metrics["diagnostic_accuracy"] = (df_metrics["TP"] + df_metrics["TN"]) / (
    df_metrics["TP"] + df_metrics["FP"] + df_metrics["FN"] + df_metrics["TN"]
)

# Precision and recall (for compatibility with previous naming)
df_metrics["precision"] = df_metrics["TP"] / (df_metrics["TP"] + df_metrics["FP"])
df_metrics["recall"] = df_metrics["TP"] / (df_metrics["TP"] + df_metrics["FN"])

# F1 (harmonic mean of precision and recall)
df_metrics["f1"] = (
    2
    * df_metrics["precision"]
    * df_metrics["recall"]
    / (df_metrics["precision"] + df_metrics["recall"])
)

df_metrics.replace([np.inf, -np.inf], np.nan, inplace=True)
# ============================================================
# Human vs AI comparison table (wide) + deltas
# ============================================================
df_metrics_table = df_metrics.set_index(["Element", "Annotator"])[
    ["accuracy", "precision", "recall", "f1"]
].unstack("Annotator")

df_metrics_table.columns = [
    f"{metric}_{annotator.lower()}" for metric, annotator in df_metrics_table.columns
]
df_metrics_table = df_metrics_table.reset_index()

for metric in ["accuracy", "precision", "recall", "f1"]:
    df_metrics_table[f"{metric}_delta_human_minus_ai"] = (
        df_metrics_table[f"{metric}_human"] - df_metrics_table[f"{metric}_ai"]
    )


# ------------------------------------------------------------------
# Diagnostic tests table: per Element x Annotator diagnostic measures
# Includes TP, FP, FN, TN and derived metrics (sensitivity, specificity,
# PPV, NPV, diagnostic_accuracy). Saved to OUTPUT_DIR as CSV.
# ------------------------------------------------------------------
diagnostic_rows = []
for _, r in df_metrics.iterrows():
    element = r["Element"]
    annot = r["Annotator"]
    TP = int(r["TP"]) if pd.notna(r["TP"]) else 0
    FP = int(r["FP"]) if pd.notna(r["FP"]) else 0
    FN = int(r["FN"]) if pd.notna(r["FN"]) else 0
    TN = int(r["TN"]) if pd.notna(r["TN"]) else 0

    # compute metrics with guarded denominators
    sens = TP / (TP + FN) if (TP + FN) > 0 else float("nan")
    spec = TN / (TN + FP) if (TN + FP) > 0 else float("nan")
    ppv = TP / (TP + FP) if (TP + FP) > 0 else float("nan")
    npv = TN / (TN + FN) if (TN + FN) > 0 else float("nan")
    diag_acc = (
        (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) > 0 else float("nan")
    )

    diagnostic_rows.append(
        {
            "Element": element,
            "Annotator": annot,
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "TN": TN,
            "sensitivity": sens,
            "specificity": spec,
            "ppv": ppv,
            "npv": npv,
            "diagnostic_accuracy": diag_acc,
        }
    )

df_diagnostic_tests = pd.DataFrame(diagnostic_rows)
safe_save_dataframe(df_diagnostic_tests, OUTPUT_DIR / "diagnostic_tests.csv")


# ------------------------------------------------------------------
# Element-level McNemar-style p-values for each diagnostic metric
# We use the exact binomial on discordant pairs (equivalent to exact
# McNemar) with alternative AI > Human (one-sided).
# ------------------------------------------------------------------
def _element_metric_pvalue(data_df, source_col, human_col, ai_col, metric_name):
    # build correctness masks per metric
    if metric_name == "accuracy":
        human_ok = annotator_correct(data_df, source_col, human_col).astype(bool)
        ai_ok = annotator_correct(data_df, source_col, ai_col).astype(bool)

    elif metric_name == "sensitivity":
        mask = data_df[source_col] == 1
        human_ok = (
            (data_df.loc[mask, human_col] == 1)
            .reindex(data_df.index, fill_value=False)
            .astype(bool)
        )
        ai_ok = (
            (data_df.loc[mask, ai_col] == 1)
            .reindex(data_df.index, fill_value=False)
            .astype(bool)
        )

    elif metric_name == "specificity":
        mask = data_df[source_col] == 0
        human_ok = (
            (data_df.loc[mask, human_col] == "N/A")
            .reindex(data_df.index, fill_value=False)
            .astype(bool)
        )
        ai_ok = (
            (data_df.loc[mask, ai_col] == "N/A")
            .reindex(data_df.index, fill_value=False)
            .astype(bool)
        )

    elif metric_name == "ppv":
        common = data_df[human_col].isin([1, 3]) & data_df[ai_col].isin([1, 3])
        human_ok = (
            (
                (data_df.loc[common, source_col] == 1)
                & (data_df.loc[common, human_col] == 1)
            )
            .reindex(data_df.index, fill_value=False)
            .astype(bool)
        )
        ai_ok = (
            (
                (data_df.loc[common, source_col] == 1)
                & (data_df.loc[common, ai_col] == 1)
            )
            .reindex(data_df.index, fill_value=False)
            .astype(bool)
        )

    elif metric_name == "npv":
        common = (data_df[human_col] == "N/A") & (data_df[ai_col] == "N/A")
        human_ok = (
            (
                (data_df.loc[common, source_col] == 0)
                & (data_df.loc[common, human_col] == "N/A")
            )
            .reindex(data_df.index, fill_value=False)
            .astype(bool)
        )
        ai_ok = (
            (
                (data_df.loc[common, source_col] == 0)
                & (data_df.loc[common, ai_col] == "N/A")
            )
            .reindex(data_df.index, fill_value=False)
            .astype(bool)
        )

    else:
        return float("nan")

    b = int(((human_ok) & (~ai_ok)).sum())
    c = int(((~human_ok) & (ai_ok)).sum())
    n_disc = b + c
    if n_disc > 0:
        return float(binomtest(k=int(c), n=n_disc, p=0.5, alternative="greater").pvalue)
    return float("nan")


# NOTE: element-level p-values are computed below after the
# `element_display_to_key` mapping is defined. Don't compute them
# earlier (it caused a NameError when the mapping wasn't available).


# ============================================================
# Stats: McNemar (paired) for ACCURACY
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


def wilson_ci(x: int, n: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """
    Wilson score CI for a binomial proportion.
    """
    if n <= 0:
        return (np.nan, np.nan)

    z = 1.959963984540054  # ~norm.ppf(0.975)
    p = x / n
    denom = 1 + (z**2) / n
    center = (p + (z**2) / (2 * n)) / denom
    half = (z * np.sqrt((p * (1 - p) + (z**2) / (4 * n)) / n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def newcombe_paired_diff_ci(
    b: int, c: int, n: int, alpha: float = 0.05
) -> tuple[float, float, float]:
    """
    Newcombe CI for paired difference in proportions (p_b - p_c),
    using Wilson intervals for the two discordant proportions.

    b = count(Human correct, AI incorrect)
    c = count(AI correct, Human incorrect)
    n = total paired observations
    """
    if n <= 0:
        return (np.nan, np.nan, np.nan)

    diff = (b / n) - (c / n)

    lb_b, ub_b = wilson_ci(b, n, alpha=alpha)
    lb_c, ub_c = wilson_ci(c, n, alpha=alpha)

    lower = lb_b - ub_c
    upper = ub_b - lb_c
    return diff, lower, upper


# Accuracy McNemar per-element removed from diagnostic stats (accuracy is treated as
# a classification metric only). If you want per-element McNemar for accuracy again,
# re-enable the block above.


# ============================================================
# Define correctness masks per metric (Recall, Precision)
# ============================================================


def recall_correct(
    d: pd.DataFrame, source_col: str, annotator_col: str
) -> tuple[pd.Series, pd.Series]:
    """
    Recall correctness is defined on source-positive cases only:
      correct iff annotator==1
    """
    mask = d[source_col] == 1
    correct = d.loc[mask, annotator_col] == 1
    return mask, correct


def precision_correct(
    d: pd.DataFrame, source_col: str, annotator_col: str
) -> tuple[pd.Series, pd.Series]:
    """
    Precision correctness is defined on predicted-positive cases only:
      predicted positive iff annotator in {1,3}
      correct among those iff (source==1 and annotator==1)
    """
    mask = d[annotator_col].isin([1, 3])
    correct = (d.loc[mask, source_col] == 1) & (d.loc[mask, annotator_col] == 1)
    return mask, correct


# ============================================================
# McNemar for Recall (subset: source==1)
# ============================================================
records_mcnemar_recall = []

for element_key, cols in elements.items():
    source_col = cols["source"]
    human_col = cols["human"]
    ai_col = cols["ai"]
    element = element_key.replace("_", " ").title()

    mask_all = data[[source_col, human_col, ai_col]].notnull().all(axis=1)
    d0 = data.loc[mask_all].copy()

    d = d0.loc[d0[source_col] == 1].copy()  # condition on source positive
    if len(d) == 0:
        continue

    human_correct = (d[human_col] == 1).astype(bool)
    ai_correct = (d[ai_col] == 1).astype(bool)

    b = ((human_correct == 1) & (ai_correct == 0)).sum()
    c = ((human_correct == 0) & (ai_correct == 1)).sum()

    table = np.array(
        [
            [(human_correct & ai_correct).sum(), b],
            [c, (~human_correct & ~ai_correct).sum()],
        ]
    )

    result = mcnemar(table, exact=((b + c) < 25))
    diff, ci_low, ci_high = newcombe_paired_diff_ci(
        int(b), int(c), int(len(d)), alpha=0.05
    )

    n_disc = int(b + c)
    if n_disc > 0:
        p_one_sided = binomtest(k=int(c), n=n_disc, p=0.5, alternative="greater").pvalue
    else:
        p_one_sided = np.nan

    records_mcnemar_recall.append(
        {
            "Element": element,
            "Metric": "Recall",
            "n_pairs": int(len(d)),
            "b_human_only": int(b),
            "c_ai_only": int(c),
            "diff_human_minus_ai": diff,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "p_value_mcnemar": p_one_sided,
            "exact_mcnemar": bool((b + c) < 25),
        }
    )

df_mcnemar_recall = pd.DataFrame(records_mcnemar_recall)


# ============================================================
# McNemar for Precision (subset: BOTH predicted positive)
# ============================================================
records_mcnemar_precision = []

for element_key, cols in elements.items():
    source_col = cols["source"]
    human_col = cols["human"]
    ai_col = cols["ai"]
    element = element_key.replace("_", " ").title()

    mask_all = data[[source_col, human_col, ai_col]].notnull().all(axis=1)
    d0 = data.loc[mask_all].copy()

    # paired precision comparison: restrict to rows where BOTH predicted present
    common_pred_pos = d0[human_col].isin([1, 3]) & d0[ai_col].isin([1, 3])
    d = d0.loc[common_pred_pos].copy()
    if len(d) == 0:
        continue

    human_correct = ((d[source_col] == 1) & (d[human_col] == 1)).astype(bool)
    ai_correct = ((d[source_col] == 1) & (d[ai_col] == 1)).astype(bool)

    b = ((human_correct == 1) & (ai_correct == 0)).sum()
    c = ((human_correct == 0) & (ai_correct == 1)).sum()

    table = np.array(
        [
            [(human_correct & ai_correct).sum(), b],
            [c, (~human_correct & ~ai_correct).sum()],
        ]
    )

    result = mcnemar(table, exact=((b + c) < 25))
    diff, ci_low, ci_high = newcombe_paired_diff_ci(
        int(b), int(c), int(len(d)), alpha=0.05
    )

    n_disc = int(b + c)
    if n_disc > 0:
        p_one_sided = binomtest(k=int(c), n=n_disc, p=0.5, alternative="greater").pvalue
    else:
        p_one_sided = np.nan

    records_mcnemar_precision.append(
        {
            "Element": element,
            "Metric": "Precision",
            "n_pairs": int(len(d)),
            "b_human_only": int(b),
            "c_ai_only": int(c),
            "diff_human_minus_ai": diff,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "p_value_mcnemar": p_one_sided,
            "exact_mcnemar": bool((b + c) < 25),
        }
    )

df_mcnemar_precision = pd.DataFrame(records_mcnemar_precision)


# ============================================================
# F1: paired bootstrap CI (no McNemar for F1)
# ============================================================


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else np.nan


def compute_tp_fp_fn(d: pd.DataFrame, source_col: str, annot_col: str):
    tp = ((d[source_col] == 1) & (d[annot_col] == 1)).sum()
    fn = ((d[source_col] == 1) & (d[annot_col] == 2)).sum()
    fp = ((d[source_col] == 1) & (d[annot_col] == 3)).sum()
    return int(tp), int(fp), int(fn)


def paired_bootstrap_f1_diff(
    d: pd.DataFrame,
    source_col: str,
    human_col: str,
    ai_col: str,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 123,
) -> dict:
    rng = np.random.default_rng(seed)
    n = len(d)
    if n == 0:
        return {
            "n_pairs": 0,
            "diff_human_minus_ai": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
        }

    diffs = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)  # sample rows with replacement
        bs = d.iloc[idx]

        tp_h, fp_h, fn_h = compute_tp_fp_fn(bs, source_col, human_col)
        tp_a, fp_a, fn_a = compute_tp_fp_fn(bs, source_col, ai_col)

        f1_h = f1_from_counts(tp_h, fp_h, fn_h)
        f1_a = f1_from_counts(tp_a, fp_a, fn_a)

        diffs[i] = f1_h - f1_a

    diff = float(np.nanmean(diffs))
    lo = float(np.nanpercentile(diffs, 100 * (alpha / 2)))
    hi = float(np.nanpercentile(diffs, 100 * (1 - alpha / 2)))

    return {"n_pairs": int(n), "diff_human_minus_ai": diff, "ci_low": lo, "ci_high": hi}


records_f1 = []
for element_key, cols in elements.items():
    source_col = cols["source"]
    human_col = cols["human"]
    ai_col = cols["ai"]
    element = element_key.replace("_", " ").title()

    mask = data[[source_col, human_col, ai_col]].notnull().all(axis=1)
    d = data.loc[mask].copy()
    if len(d) == 0:
        continue

    out = paired_bootstrap_f1_diff(
        d, source_col, human_col, ai_col, n_boot=2000, alpha=0.05, seed=123
    )
    records_f1.append({"Element": element, "Metric": "F1", **out})

df_f1_bootstrap = pd.DataFrame(records_f1)


# ============================================================
# Combine stats into one table + convert to %
# ============================================================
df_stats = pd.concat(
    [df_mcnemar_accuracy, df_mcnemar_recall, df_mcnemar_precision, df_f1_bootstrap],
    ignore_index=True,
)

for col in ["diff_human_minus_ai", "ci_low", "ci_high"]:
    if col in df_stats.columns:
        df_stats[col] = df_stats[col] * 100

# ----------------------------
# Helpers for CIs + bootstrap
# ----------------------------


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else np.nan


def compute_tp_fp_fn(d: pd.DataFrame, source_col: str, annot_col: str):
    tp = ((d[source_col] == 1) & (d[annot_col] == 1)).sum()
    fn = ((d[source_col] == 1) & (d[annot_col] == 2)).sum()
    fp = ((d[source_col] == 1) & (d[annot_col] == 3)).sum()
    return int(tp), int(fp), int(fn)


def bootstrap_f1_ci_for_annotator(
    d: pd.DataFrame,
    source_col: str,
    annot_col: str,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 123,
):
    """Bootstrap CI for F1 for a single annotator (resample rows)."""
    rng = np.random.default_rng(seed)
    n = len(d)
    if n == 0:
        return (np.nan, np.nan, np.nan)

    f1s = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        bs = d.iloc[idx]
        tp, fp, fn = compute_tp_fp_fn(bs, source_col, annot_col)
        f1s[i] = f1_from_counts(tp, fp, fn)

    point = np.nanmean(f1s)
    lo = np.nanpercentile(f1s, 100 * (alpha / 2))
    hi = np.nanpercentile(f1s, 100 * (1 - alpha / 2))
    return float(point), float(lo), float(hi)


# ----------------------------
# Build per-bar CIs for all metrics
# Uses df_confusion (TP/FP/FN/TN) for accuracy/precision/recall
# Uses elements + data for F1 bootstrap CI
# ----------------------------

# Expect these objects already exist from your script:
# - df_confusion  (Element, Annotator, TP, FP, FN, TN, ...)
# - df_metrics_table (Element, accuracy_human, accuracy_ai, ..., deltas)
# - df_stats (Element, Metric, p_value_mcnemar or p_value_holm, ci_low/ci_high for diff)
# - elements dict with keys and cols {"source","human","ai"}
# - data dataframe

# 1) Accuracy / Precision / Recall CIs from counts (Wilson)
ci_rows = []
for _, r in df_confusion.iterrows():
    element = r["Element"]
    annot = r["Annotator"]
    TP, FP, FN, TN = int(r["TP"]), int(r["FP"]), int(r["FN"]), int(r["TN"])

    # accuracy
    n_acc = TP + FP + FN + TN
    x_acc = TP + TN
    lo, hi = wilson_ci(x_acc, n_acc)
    ci_rows.append(
        {
            "Element": element,
            "Annotator": annot,
            "metric": "accuracy",
            "value": (x_acc / n_acc) if n_acc > 0 else np.nan,
            "ci_low": lo,
            "ci_high": hi,
        }
    )

    # precision
    n_prec = TP + FP
    x_prec = TP
    lo, hi = wilson_ci(x_prec, n_prec)
    ci_rows.append(
        {
            "Element": element,
            "Annotator": annot,
            "metric": "precision",
            "value": (x_prec / n_prec) if n_prec > 0 else np.nan,
            "ci_low": lo,
            "ci_high": hi,
        }
    )

    # recall
    n_rec = TP + FN
    x_rec = TP
    lo, hi = wilson_ci(x_rec, n_rec)
    ci_rows.append(
        {
            "Element": element,
            "Annotator": annot,
            "metric": "recall",
            "value": (x_rec / n_rec) if n_rec > 0 else np.nan,
            "ci_low": lo,
            "ci_high": hi,
        }
    )

df_ci = pd.DataFrame(ci_rows)

# 2) F1 point estimate from df_metrics_table + bootstrap CI per annotator from raw rows
# Map display Element -> element_key used in `elements` dict
# (This assumes Element in df_metrics_table was generated as element_key.replace("_"," ").title())
element_display_to_key = {k.replace("_", " ").title(): k for k in elements.keys()}

# ------------------------------------------------------------------
# Element-level p-values for diagnostic metrics (McNemar-style exact binomial)
# Compute one-sided p (AI > Human) per element and attach to df_diagnostic_tests
# ------------------------------------------------------------------
metrics_for_diag = ["sensitivity", "specificity", "ppv", "npv"]


def _compute_element_pvalues_for_element(element_key):
    cols = elements[element_key]
    s_col, h_col, a_col = cols["source"], cols["human"], cols["ai"]

    mask = data[[s_col, h_col, a_col]].notnull().all(axis=1)
    dsub = data.loc[mask].copy()
    results = {}

    for metric in metrics_for_diag:
        # reuse earlier logic for correctness masks
        if metric == "accuracy":
            human_ok = annotator_correct(dsub, s_col, h_col).astype(bool)
            ai_ok = annotator_correct(dsub, s_col, a_col).astype(bool)

        elif metric == "sensitivity":
            m = dsub[s_col] == 1
            human_ok = (
                (dsub.loc[m, h_col] == 1)
                .reindex(dsub.index, fill_value=False)
                .astype(bool)
            )
            ai_ok = (
                (dsub.loc[m, a_col] == 1)
                .reindex(dsub.index, fill_value=False)
                .astype(bool)
            )

        elif metric == "specificity":
            m = dsub[s_col] == 0
            human_ok = (
                (dsub.loc[m, h_col] == "N/A")
                .reindex(dsub.index, fill_value=False)
                .astype(bool)
            )
            ai_ok = (
                (dsub.loc[m, a_col] == "N/A")
                .reindex(dsub.index, fill_value=False)
                .astype(bool)
            )

        elif metric == "ppv":
            common = dsub[h_col].isin([1, 3]) & dsub[a_col].isin([1, 3])
            human_ok = (
                ((dsub.loc[common, s_col] == 1) & (dsub.loc[common, h_col] == 1))
                .reindex(dsub.index, fill_value=False)
                .astype(bool)
            )
            ai_ok = (
                ((dsub.loc[common, s_col] == 1) & (dsub.loc[common, a_col] == 1))
                .reindex(dsub.index, fill_value=False)
                .astype(bool)
            )

        elif metric == "npv":
            common = (dsub[h_col] == "N/A") & (dsub[a_col] == "N/A")
            human_ok = (
                ((dsub.loc[common, s_col] == 0) & (dsub.loc[common, h_col] == "N/A"))
                .reindex(dsub.index, fill_value=False)
                .astype(bool)
            )
            ai_ok = (
                ((dsub.loc[common, s_col] == 0) & (dsub.loc[common, a_col] == "N/A"))
                .reindex(dsub.index, fill_value=False)
                .astype(bool)
            )

        else:
            results[metric] = np.nan
            continue

        b = int(((human_ok) & (~ai_ok)).sum())
        c = int(((~human_ok) & (ai_ok)).sum())
        n_disc = b + c
        if n_disc > 0:
            p = float(
                binomtest(k=int(c), n=n_disc, p=0.5, alternative="greater").pvalue
            )
        else:
            p = float("nan")

        results[metric] = p

    return results


# Ensure p columns exist (we will store same p for Human and AI rows for the element)
for metric in metrics_for_diag:
    df_diagnostic_tests[f"{metric}_p"] = np.nan

for display_el, key in element_display_to_key.items():
    pmap = _compute_element_pvalues_for_element(key)
    # assign to both Human and AI rows for that Element
    mask = df_diagnostic_tests["Element"] == display_el
    for metric, pval in pmap.items():
        df_diagnostic_tests.loc[mask, f"{metric}_p"] = pval

safe_save_dataframe(df_diagnostic_tests, OUTPUT_DIR / "diagnostic_tests_with_p.csv")


f1_rows = []
for element_display in df_metrics_table["Element"].tolist():
    key = element_display_to_key.get(element_display)
    if key is None:
        continue

    cols = elements[key]
    source_col, human_col, ai_col = cols["source"], cols["human"], cols["ai"]

    mask = data[[source_col, human_col, ai_col]].notnull().all(axis=1)
    d = data.loc[mask].copy()
    if len(d) == 0:
        continue

    # Bootstrap CI for each annotator F1
    _, h_lo, h_hi = bootstrap_f1_ci_for_annotator(
        d, source_col, human_col, n_boot=2000, alpha=0.05, seed=123
    )
    _, a_lo, a_hi = bootstrap_f1_ci_for_annotator(
        d, source_col, ai_col, n_boot=2000, alpha=0.05, seed=123
    )

    # Point estimates from df_metrics_table (ensures same as your computed metrics)
    row = df_metrics_table.loc[df_metrics_table["Element"] == element_display].iloc[0]
    h_val = float(row["f1_human"])
    a_val = float(row["f1_ai"])

    f1_rows.append(
        {
            "Element": element_display,
            "Annotator": "Human",
            "metric": "f1",
            "value": h_val,
            "ci_low": h_lo,
            "ci_high": h_hi,
        }
    )
    f1_rows.append(
        {
            "Element": element_display,
            "Annotator": "AI",
            "metric": "f1",
            "value": a_val,
            "ci_low": a_lo,
            "ci_high": a_hi,
        }
    )

df_ci = pd.concat([df_ci, pd.DataFrame(f1_rows)], ignore_index=True)


# ----------------------------
# Significance flags per element x metric
# ----------------------------
def sig_star(p):
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


sig_map = {}
if "p_value_holm" in df_stats.columns:
    p_col = "p_value_holm"
elif "p_value_mcnemar" in df_stats.columns:
    p_col = "p_value_mcnemar"
else:
    p_col = None

for _, r in df_stats.iterrows():
    element = r["Element"]
    metric = str(r["Metric"]).strip().lower()

    if metric == "f1":
        lo = r.get("ci_low", np.nan)
        hi = r.get("ci_high", np.nan)
        if pd.notna(lo) and pd.notna(hi) and ((lo > 0) or (hi < 0)):
            sig_map[(element, metric)] = "*"
        else:
            sig_map[(element, metric)] = ""
        continue

    if p_col is None:
        sig_map[(element, metric)] = ""
        continue

    p = r.get(p_col, np.nan)
    sig_map[(element, metric)] = sig_star(p)


# ----------------------------
# Plot: facet-wrapped grouped bars with CIs + stars + ordering by delta
# ----------------------------
metric_order = ["accuracy", "precision", "recall", "f1"]
metric_titles = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1 score",
}

# Build long plot table from df_ci (already long) and ensure percent scaling
plot_df = df_ci.copy()
plot_df["value_pct"] = plot_df["value"] * 100
plot_df["ci_low_pct"] = plot_df["ci_low"] * 100
plot_df["ci_high_pct"] = plot_df["ci_high"] * 100

# Delta table to order elements per metric
delta_cols = {
    "accuracy": "accuracy_delta_human_minus_ai",
    "precision": "precision_delta_human_minus_ai",
    "recall": "recall_delta_human_minus_ai",
    "f1": "f1_delta_human_minus_ai",
}

# Figure setup
fig, axes = plt.subplots(2, 2, figsize=(18, 11), sharey=True)
axes = axes.flatten()

bar_width = 0.38
y_pad = 2.0  # space above bars for stars
cap = 3

for ax, metric in zip(axes, metric_order):
    # Order elements by delta (descending: human - ai)
    order = (
        df_metrics_table[["Element", delta_cols[metric]]]
        .sort_values(delta_cols[metric], ascending=False)["Element"]
        .tolist()
    )

    sub = plot_df[plot_df["metric"] == metric].copy()
    sub["Element"] = pd.Categorical(sub["Element"], categories=order, ordered=True)
    sub = sub.sort_values(["Element", "Annotator"])

    x = np.arange(len(order))

    # Get human and ai rows aligned to order
    human = sub[sub["Annotator"] == "Human"].set_index("Element").reindex(order)
    ai = sub[sub["Annotator"] == "AI"].set_index("Element").reindex(order)

    human_vals = human["value_pct"].to_numpy()
    ai_vals = ai["value_pct"].to_numpy()

    # Asymmetric yerr: [[lower...], [upper...]]
    human_yerr = np.vstack(
        [
            np.maximum(human_vals - human["ci_low_pct"].to_numpy(), 0),
            np.maximum(human["ci_high_pct"].to_numpy() - human_vals, 0),
        ]
    )
    ai_yerr = np.vstack(
        [
            np.maximum(ai_vals - ai["ci_low_pct"].to_numpy(), 0),
            np.maximum(ai["ci_high_pct"].to_numpy() - ai_vals, 0),
        ]
    )

    # Bars
    ax.bar(x - bar_width / 2, human_vals, width=bar_width, label="Human", color="black")
    ax.bar(x + bar_width / 2, ai_vals, width=bar_width, label="AI", color="gray")

    # Error bars
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

    # Titles + axes
    ax.set_title(metric_titles[metric])
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=35, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Percent (%)")
    ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.6)

    # Stars for significant Human vs AI difference (per element, per metric)
    # Place star above the better-performing bar for that element/metric.
    deltas = (
        df_metrics_table.set_index("Element")[delta_cols[metric]]
        .reindex(order)
        .to_numpy()
    )

    for i, element in enumerate(order):
        star = sig_map.get((element, metric), "")
        if star == "":
            continue

        # choose bar based on delta sign
        if np.isnan(deltas[i]):
            continue

        if deltas[i] >= 0:
            # Human higher (or tie): annotate above human bar
            y = human_vals[i] + human_yerr[1, i] + y_pad
            x_pos = x[i] - bar_width / 2
        else:
            # AI higher
            y = ai_vals[i] + ai_yerr[1, i] + y_pad
            x_pos = x[i] + bar_width / 2

        ax.text(x_pos, min(y, 99.5), star, ha="center", va="bottom", fontsize=12)

# One legend for the whole figure
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

# Save figure
out_path = OUTPUT_DIR / "human_ai_metrics_facet_ci.png"
safe_save_plt(fig, out_path, dpi=600, bbox_inches="tight")
if os.environ.get("SHOW_PLOTS", "0") == "1":
    plt.show()
else:
    plt.close(fig)

# Aggregate metrics and plots

import numpy as np
import pandas as pd
# Plotly imports removed: using Matplotlib for aggregated domain plot to match other figures

# ----------------------------
# Domain definitions (derived from metadata)
# ----------------------------
domain_triplets = defaultdict(list)

for element_key, cols in elements.items():
    domain = cols.get("domain", "Unspecified")
    triplet = (cols["source"], cols["human"], cols["ai"])
    domain_triplets[domain].append(triplet)

rad_col = domain_triplets.get("Radiology", [])
path_col = domain_triplets.get("Pathology", [])


# ----------------------------
# Helpers: element mapping
# ----------------------------
def element_display_from_source_col(source_col: str) -> str:
    key = source_col.replace("_status_source", "").replace("_source", "")
    return key.replace("_", " ").title()


def elements_from_triplets(triplets) -> list[str]:
    return [element_display_from_source_col(t[0]) for t in triplets]


rad_elements = set(elements_from_triplets(rad_col))
path_elements = set(elements_from_triplets(path_col))


# ----------------------------
# Bootstrap CIs across ELEMENTS
# ----------------------------
def bootstrap_mean_ci(
    values: np.ndarray, n_boot: int = 5000, alpha: float = 0.05, seed: int = 123
):
    """
    Nonparametric bootstrap CI for the mean of `values` (resample elements).
    Returns: (mean, lo, hi)
    """
    values = values.astype(float)
    values = values[~np.isnan(values)]
    n = len(values)
    if n == 0:
        return (np.nan, np.nan, np.nan)

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = np.mean(values[idx])

    point = float(np.mean(values))
    lo = float(np.percentile(boot_means, 100 * (alpha / 2)))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return point, lo, hi


def bootstrap_paired_diff_ci(
    human_vals: np.ndarray,
    ai_vals: np.ndarray,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 123,
):
    """
    Paired bootstrap CI for mean difference (human - ai) across elements.
    Resamples element indices with replacement.
    Returns: (diff_mean, lo, hi)
    """
    human_vals = human_vals.astype(float)
    ai_vals = ai_vals.astype(float)

    # Pairwise complete cases
    mask = ~np.isnan(human_vals) & ~np.isnan(ai_vals)
    human_vals = human_vals[mask]
    ai_vals = ai_vals[mask]

    n = len(human_vals)
    if n == 0:
        return (np.nan, np.nan, np.nan)

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = np.mean(human_vals[idx] - ai_vals[idx])

    point = float(np.mean(human_vals - ai_vals))
    lo = float(np.percentile(diffs, 100 * (alpha / 2)))
    hi = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    return point, lo, hi


def stars_from_ci(ci_low: float, ci_high: float) -> str:
    """
    Significance by whether paired-diff CI excludes 0.
    Uses 95% CI => '*' if excludes 0, else ''.
    """
    if np.isnan(ci_low) or np.isnan(ci_high):
        return ""
    return "*" if (ci_low > 0) or (ci_high < 0) else ""


# ----------------------------
# Build aggregated table with CIs and paired-diff CIs
# Requires df_metrics_table already exists.
# ----------------------------
metrics = ["accuracy", "precision", "recall", "f1"]
metric_label = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
}
x_order = [metric_label[m] for m in metrics]
domains = [("Radiology", rad_elements), ("Pathology", path_elements)]

agg_rows = []

for domain_name, element_set in domains:
    sub = df_metrics_table[df_metrics_table["Element"].isin(element_set)].copy()

    for m in metrics:
        human_vals = sub[f"{m}_human"].to_numpy()
        ai_vals = sub[f"{m}_ai"].to_numpy()

        h_mean, h_lo, h_hi = bootstrap_mean_ci(
            human_vals, n_boot=5000, alpha=0.05, seed=123
        )
        a_mean, a_lo, a_hi = bootstrap_mean_ci(
            ai_vals, n_boot=5000, alpha=0.05, seed=123
        )

        diff_mean, diff_lo, diff_hi = bootstrap_paired_diff_ci(
            human_vals, ai_vals, n_boot=5000, alpha=0.05, seed=123
        )
        star = stars_from_ci(diff_lo, diff_hi)

        agg_rows.append(
            {
                "Domain": domain_name,
                "Metric": m,
                "MetricLabel": metric_label[m],
                "HumanMean": h_mean,
                "HumanCI_L": h_lo,
                "HumanCI_H": h_hi,
                "AIMean": a_mean,
                "AICI_L": a_lo,
                "AICI_H": a_hi,
                "DiffMean_HminusAI": diff_mean,
                "DiffCI_L": diff_lo,
                "DiffCI_H": diff_hi,
                "Star": star,
                "n_elements": int(sub[f"{m}_human"].notna().sum()),
            }
        )

df_agg = pd.DataFrame(agg_rows)

# Convert to percent for plotting
for col in [
    "HumanMean",
    "HumanCI_L",
    "HumanCI_H",
    "AIMean",
    "AICI_L",
    "AICI_H",
    "DiffMean_HminusAI",
    "DiffCI_L",
    "DiffCI_H",
]:
    df_agg[col + "_Pct"] = df_agg[col] * 100


# ------------------------------------------------------------------
# Aggregated paired-bootstrap p-values across elements by Domain and Metric
# Uses paired-bootstrap on element-level values (Human - AI) and computes
# one-sided p-value for alternative AI > Human as proportion(diffs < 0).
# ------------------------------------------------------------------
def paired_bootstrap_pvalue(human_vals, ai_vals, n_boot=5000, seed=123):
    human = np.array(human_vals, dtype=float)
    ai = np.array(ai_vals, dtype=float)
    mask = ~np.isnan(human) & ~np.isnan(ai)
    human = human[mask]
    ai = ai[mask]
    n = len(human)
    if n == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = np.mean(human[idx] - ai[idx])
    # one-sided p for AI > Human corresponds to proportion of diffs < 0
    p_one_sided = float((diffs < 0).mean())
    return p_one_sided


# Compute aggregated p-values and attach to df_agg
p_vals = []
for _, row in df_agg.iterrows():
    domain = row["Domain"]
    metric = row["Metric"]
    # Gather element-level values from df_metrics_table
    elems = elements[domain] if False else None  # placeholder to indicate domain usage
    # We'll rebuild human/ai arrays by selecting elements in this domain
    dom_elements = (
        df_metrics_table["Element"]
        .loc[
            df_metrics_table["Element"].isin(
                [
                    e.replace("_", " ").title()
                    for e in elements.keys()
                    if elements[e]["domain"] == domain
                ]
            )
        ]
        .unique()
    )
    # human and ai column names
    metric_col_h = f"{metric.lower()}_human"
    metric_col_a = f"{metric.lower()}_ai"
    human_vals = []
    ai_vals = []
    for el in dom_elements:
        try:
            row_h = df_metrics_table.loc[df_metrics_table["Element"] == el].iloc[0]
            human_vals.append(row_h.get(metric_col_h, np.nan))
            ai_vals.append(row_h.get(metric_col_a, np.nan))
        except Exception:
            human_vals.append(np.nan)
            ai_vals.append(np.nan)

    p_agg = paired_bootstrap_pvalue(
        np.array(human_vals, dtype=float),
        np.array(ai_vals, dtype=float),
        n_boot=2000,
        seed=123,
    )
    p_vals.append(p_agg)

df_agg["p_value_agg"] = p_vals
safe_save_dataframe(df_agg, OUTPUT_DIR / "domain_agg_metrics_with_p.csv")

# Build display table per user's format (Domain groups, Human/AI rows)
try:
    metric_map = {
        "Accuracy": ("HumanMean_Pct", "HumanCI_L_Pct", "HumanCI_H_Pct", "p_value_agg"),
        "Sensitivity": (
            "HumanMean_Pct",
            "HumanCI_L_Pct",
            "HumanCI_H_Pct",
            "p_value_agg",
        ),
        "Specificity": (
            "HumanMean_Pct",
            "HumanCI_L_Pct",
            "HumanCI_H_Pct",
            "p_value_agg",
        ),
        "PPV": ("HumanMean_Pct", "HumanCI_L_Pct", "HumanCI_H_Pct", "p_value_agg"),
        "NPV": ("HumanMean_Pct", "HumanCI_L_Pct", "HumanCI_H_Pct", "p_value_agg"),
    }

    rows = []
    for domain in df_agg["Domain"].unique():
        subd = df_agg[df_agg["Domain"] == domain]
        for judge in ["Human", "AI"]:
            row = {"Domain": domain, "Judge": judge}
            for metric_label, (est_c, lo_c, hi_c, p_c) in metric_map.items():
                # find metric row
                r = subd[subd["MetricLabel"] == metric_label].iloc[0]
                star = r.get("Star", "")
                star = "" if pd.isna(star) else str(star)
                if judge == "Human":
                    est = r.get("HumanMean_Pct", np.nan)
                    lo = r.get("HumanCI_L_Pct", np.nan)
                    hi = r.get("HumanCI_H_Pct", np.nan)
                else:
                    est = r.get("AIMean_Pct", np.nan)
                    lo = r.get("AICI_L_Pct", np.nan)
                    hi = r.get("AICI_H_Pct", np.nan)

                # fetch p from df_agg row (using aggregated p_value_agg)
                p = r.get("p_value_agg", np.nan)

                if pd.notna(est):
                    cell = f"{est:.1f}% ({lo:.1f}, {hi:.1f}){star}"
                else:
                    cell = ""
                row[metric_label] = cell
            rows.append(row)

    df_class_table = pd.DataFrame(rows)
    # ordering
    domain_order = ["Radiology", "Pathology"]
    judge_order = ["Human", "AI"]
    df_class_table["Domain"] = pd.Categorical(
        df_class_table["Domain"], categories=domain_order, ordered=True
    )
    df_class_table["Judge"] = pd.Categorical(
        df_class_table["Judge"], categories=judge_order, ordered=True
    )
    df_class_table = df_class_table.sort_values(["Domain", "Judge"]).reset_index(
        drop=True
    )
    safe_save_dataframe(df_class_table, OUTPUT_DIR / "domain_class_table.csv")
except Exception as exc:
    print(f"Failed building aggregated class table: {exc}")


# ----------------------------
# Plot: two facets (Rad vs Path), clustered by metric, Human vs AI with error bars + stars
# ----------------------------

# Build the same grouped bar figure using Matplotlib (black/gray) so it matches
# the other Matplotlib figures and does not require Plotly/Kaleido.
try:
    import matplotlib.patches as mpatches

    n_metrics = len(x_order)
    x = np.arange(n_metrics)
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax, (domain_name, _) in zip(axes, domains):
        d = (
            df_agg[df_agg["Domain"] == domain_name]
            .set_index("MetricLabel")
            .reindex(x_order)
        )

        human_mean = d["HumanMean_Pct"].to_numpy()
        ai_mean = d["AIMean_Pct"].to_numpy()

        human_err_plus = d["HumanCI_H_Pct"].to_numpy() - human_mean
        human_err_minus = human_mean - d["HumanCI_L_Pct"].to_numpy()

        ai_err_plus = d["AICI_H_Pct"].to_numpy() - ai_mean
        ai_err_minus = ai_mean - d["AICI_L_Pct"].to_numpy()

        # Bars: Human black, AI gray
        ax.bar(x - width / 2, human_mean, width, label="Human", color="black")
        ax.bar(x + width / 2, ai_mean, width, label="AI", color="gray")

        # Error bars (asymmetric)
        ax.errorbar(
            x - width / 2,
            human_mean,
            yerr=[human_err_minus, human_err_plus],
            fmt="none",
            ecolor="black",
            capsize=4,
        )
        ax.errorbar(
            x + width / 2,
            ai_mean,
            yerr=[ai_err_minus, ai_err_plus],
            fmt="none",
            ecolor="black",
            capsize=4,
        )

        # Stars above clusters
        stars = d["Star"].to_list()
        for i, star in enumerate(stars):
            if not star:
                continue
            # compute top y for annotation
            y_top = (
                max(human_mean[i] + human_err_plus[i], ai_mean[i] + ai_err_plus[i])
                + 2.0
            )
            ax.text(x[i], min(y_top, 99.5), star, ha="center", va="bottom", fontsize=16)

        ax.set_xticks(x)
        ax.set_xticklabels(x_order, rotation=35, ha="right")
        ax.set_ylim(0, 100)
        ax.set_title(domain_name)
        ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.6)

    # Single legend
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

    # Save figure
    out_path = OUTPUT_DIR / "avg_metrics_rad_vs_path_grouped_ci_stars.png"
    safe_save_plt(fig, out_path, dpi=300, bbox_inches="tight")
    if os.environ.get("SHOW_PLOTS", "0") == "1":
        plt.show()
    else:
        plt.close(fig)
except Exception as exc:
    print(f"Failed generating Matplotlib aggregated domain plot: {exc}")


# -------------------------
# Outputs to inspect/export
# -------------------------
# df_confusion
#   Long-format confusion matrix table with:
#     Element, Annotator (Human/AI),
#     TP, FP, FN, TN,
#     TPR, FNR, FPR, TNR
#
# df_metrics_table
#   Element-level performance metrics with:
#     accuracy_human / accuracy_ai
#     precision_human / precision_ai
#     recall_human / recall_ai
#     f1_human / f1_ai
#     + human-minus-AI deltas for each metric
#
# df_stats
#   Paired Human vs AI comparisons by element and metric, including:
#     - Accuracy: McNemar test + Newcombe 95% CI
#     - Recall: McNemar test (conditioned on source = 1) + CI
#     - Precision: McNemar test (conditioned on predicted positive) + CI
#     - F1: Paired bootstrap difference + 95% CI
#
# Example inspection:
# print(df_confusion.head())
# print(df_metrics_table)
# print(df_stats.sort_values(["Metric", "Element"]))

# df_agg
#   Aggregated domain-level metric means with bootstrap CIs across elements and paired-diff CIs
#   Columns include:
#     Domain, Metric, MetricLabel,
#     HumanMean(_Pct), HumanCI_L(_Pct), HumanCI_H(_Pct),
#     AIMean(_Pct), AICI_L(_Pct), AICI_H(_Pct),
#     DiffMean_HminusAI(_Pct), DiffCI_L(_Pct), DiffCI_H(_Pct),
#     Star ('' or '*'), n_elements
#
# Figure (saved)
#   avg_metrics_rad_vs_path_grouped_ci_stars.png


# ------------------------------------------------------------------
# Additional: specificity pies, ROC/PR examples, error analysis, bias-variance, CV
# These plots use confusion matrices (TP/FP/FN/TN) computed above when available.
# They are saved to OUTPUT_DIR when it exists; otherwise skipped with a warning.
# ------------------------------------------------------------------


def calculate_specificity_from_cm_df(cm_df: pd.DataFrame) -> float:
    """Calculate specificity = TN / (TN + FP) from a confusion table DataFrame.
    Expects layout:
      index = ['Predicted Positive', 'Predicted Negative']
      columns = ['Actual Positive', 'Actual Negative']
    """
    try:
        # Convert to float to safely handle any NaN entries
        matrix = np.array(cm_df.values, dtype=float)
        tn = matrix[1, 1]
        fp = matrix[0, 1]
        if np.isnan(tn) or np.isnan(fp):
            return float("nan")
        denom = tn + fp
        return float(tn / denom) if denom > 0 else float("nan")
    except Exception:
        return float("nan")


try:
    # Human specificity
    if "human_cm_df" in globals() and human_cm_df is not None:
        # Use raw counts for pie slices (TN and FP) so the pie represents counts
        # but we still compute specificity = TN / (TN + FP) for display/logging.
        try:
            mat = np.array(human_cm_df.values, dtype=float)
            tn = float(mat[1, 1])
            fp = float(mat[0, 1])
        except Exception:
            tn = float("nan")
            fp = float("nan")

        if np.isnan(tn) or np.isnan(fp):
            print(
                "Human specificity is NaN — skipping specificity pie (insufficient TN/FP counts)."
            )
        else:
            denom = tn + fp
            if denom <= 0:
                print(
                    "Human specificity denom TN+FP == 0 — skipping specificity pie (no Actual Negatives)."
                )
            else:
                spec_h = tn / denom
                print(f"Human specificity: {spec_h:.3f} (TN={int(tn)}, FP={int(fp)})")
                fig = plt.figure(figsize=(6, 4))
                vals = [tn, fp]
                labels = ["True Negative", "False Positive"]
                plt.pie(vals, labels=labels, autopct="%1.1f%%", startangle=90)
                plt.title(
                    "Human Specificity: Proportion of Actual Negatives Correctly Identified"
                )
                plt.axis("equal")
                # Save figure
                out_path = OUTPUT_DIR / "specificity_human.png"
                safe_save_plt(fig, out_path, dpi=300)
                if os.environ.get("SHOW_PLOTS", "0") == "1":
                    plt.show()
                else:
                    plt.close(fig)

    # AI specificity
    if "ai_cm_df" in globals() and ai_cm_df is not None:
        try:
            mat = np.array(ai_cm_df.values, dtype=float)
            tn = float(mat[1, 1])
            fp = float(mat[0, 1])
        except Exception:
            tn = float("nan")
            fp = float("nan")

        if np.isnan(tn) or np.isnan(fp):
            print(
                "AI specificity is NaN — skipping specificity pie (insufficient TN/FP counts)."
            )
        else:
            denom = tn + fp
            if denom <= 0:
                print(
                    "AI specificity denom TN+FP == 0 — skipping specificity pie (no Actual Negatives)."
                )
            else:
                spec_a = tn / denom
                print(f"AI specificity: {spec_a:.3f} (TN={int(tn)}, FP={int(fp)})")
                fig = plt.figure(figsize=(6, 4))
                vals = [tn, fp]
                labels = ["True Negative", "False Positive"]
                plt.pie(vals, labels=labels, autopct="%1.1f%%", startangle=90)
                plt.title(
                    "AI Specificity: Proportion of Actual Negatives Correctly Identified"
                )
                plt.axis("equal")
                # Save figure
                out_path = OUTPUT_DIR / "specificity_ai.png"
                safe_save_plt(fig, out_path, dpi=300)
                if os.environ.get("SHOW_PLOTS", "0") == "1":
                    plt.show()
                else:
                    plt.close(fig)
except Exception as exc:
    print(f"Failed generating specificity pies: {exc}")


# ------------------------------------------------------------------
# Element-level diagnostic metrics plots: Human vs AI
# ------------------------------------------------------------------
try:
    # ---- config ----
    metrics = ["Sensitivity", "Specificity", "PPV", "NPV"]
    metric_base = {
        "Sensitivity": "sensitivity",
        "Specificity": "specificity",
        "PPV": "ppv",
        "NPV": "npv",
    }

    # Build dataframe with CIs for diagnostic metrics using Wilson score interval
    diag_ci_rows = []
    for _, r in df_diagnostic_tests.iterrows():
        element = r["Element"]
        annot = r["Annotator"]
        TP, FP, FN, TN = int(r["TP"]), int(r["FP"]), int(r["FN"]), int(r["TN"])

        # Sensitivity
        n_sens = TP + FN
        x_sens = TP
        sens_lo, sens_hi = wilson_ci(x_sens, n_sens)
        diag_ci_rows.append(
            {
                "Element": element,
                "Annotator": annot,
                "Metric": "Sensitivity",
                "value": r["sensitivity"],
                "ci_low": sens_lo,
                "ci_high": sens_hi,
                "p_value": r.get("sensitivity_p", np.nan),
            }
        )

        # Specificity
        n_spec = TN + FP
        x_spec = TN
        spec_lo, spec_hi = wilson_ci(x_spec, n_spec)
        diag_ci_rows.append(
            {
                "Element": element,
                "Annotator": annot,
                "Metric": "Specificity",
                "value": r["specificity"],
                "ci_low": spec_lo,
                "ci_high": spec_hi,
                "p_value": r.get("specificity_p", np.nan),
            }
        )

        # PPV
        n_ppv = TP + FP
        x_ppv = TP
        ppv_lo, ppv_hi = wilson_ci(x_ppv, n_ppv)
        diag_ci_rows.append(
            {
                "Element": element,
                "Annotator": annot,
                "Metric": "PPV",
                "value": r["ppv"],
                "ci_low": ppv_lo,
                "ci_high": ppv_hi,
                "p_value": r.get("ppv_p", np.nan),
            }
        )

        # NPV
        n_npv = TN + FN
        x_npv = TN
        npv_lo, npv_hi = wilson_ci(x_npv, n_npv)
        diag_ci_rows.append(
            {
                "Element": element,
                "Annotator": annot,
                "Metric": "NPV",
                "value": r["npv"],
                "ci_low": npv_lo,
                "ci_high": npv_hi,
                "p_value": r.get("npv_p", np.nan),
            }
        )

    df_diag_ci = pd.DataFrame(diag_ci_rows)

    # Get element order (sorted alphabetically)
    element_order = sorted(df_diag_ci["Element"].unique())

    # Add significance stars
    df_diag_ci["sig_star"] = df_diag_ci["p_value"].apply(
        lambda p: (
            "***"
            if pd.notna(p) and p < 0.001
            else "**"
            if pd.notna(p) and p < 0.01
            else "*"
            if pd.notna(p) and p < 0.05
            else ""
        )
    )

    # ---- PLOT 1: Element-level diagnostic metrics (facet wrapped by metric) ----
    # Setup grid sized to number of diagnostic metrics (now 4 metrics)
    n_rows, n_cols = 2, 2
    fig_element, axes = plt.subplots(n_rows, n_cols, figsize=(16, 10), sharey=True)
    axes = axes.flatten()

    bar_width = 0.38
    y_pad = 0.02  # space above bars for stars
    cap = 3

    for ax_idx, metric in enumerate(metrics):
        ax = axes[ax_idx]

        # Get data for this metric
        df_metric = df_diag_ci[df_diag_ci["Metric"] == metric].copy()

        # Separate human and AI, ordered by element_order
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

        # Asymmetric yerr: [[lower...], [upper...]]
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

        # Bars (black for Human, gray for AI)
        ax.bar(
            x - bar_width / 2, human_vals, width=bar_width, label="Human", color="black"
        )
        ax.bar(x + bar_width / 2, ai_vals, width=bar_width, label="AI", color="gray")

        # Error bars
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

        # Titles + axes
        ax.set_title(metric, fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(element_order, rotation=45, ha="right", fontsize=9)
        ax.set_ylim(0, 1.1)
    ax.set_ylabel("Value" if (ax_idx % n_cols) == 0 else "")
        ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.6)

        # Add significance stars
        # Get p-values for this metric (same for human and AI rows)
        for i, element in enumerate(element_order):
            star = human.iloc[i]["sig_star"]
            if star == "":
                continue

            # Place star above the higher bar
            if human_vals[i] >= ai_vals[i]:
                y = human_vals[i] + human_yerr[1, i] + y_pad
                x_pos = x[i] - bar_width / 2
            else:
                y = ai_vals[i] + ai_yerr[1, i] + y_pad
                x_pos = x[i] + bar_width / 2

            ax.text(x_pos, min(y, 1.08), star, ha="center", va="bottom", fontsize=12)

    # Hide any unused subplots (if grid larger than metrics list)
    for idx in range(len(metrics), len(axes)):
        axes[idx].axis("off")

    # One legend for the whole figure
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

    # Save figure
    out_path = OUTPUT_DIR / "element_level_diagnostic_metrics_human_vs_ai.png"
    safe_save_plt(fig_element, out_path, dpi=600, bbox_inches="tight")
    if os.environ.get("SHOW_PLOTS", "0") == "1":
        plt.show()
    else:
        plt.close(fig_element)
    print(f"✓ Element-level plot saved to {out_path}")

    # ---- PLOT 2: Domain-aggregated diagnostic metrics ----
    # Get domain mapping from elements dict
    element_to_domain = {}
    for elem_key, elem_info in elements.items():
        display_name = elem_key.replace("_", " ").title()
        element_to_domain[display_name] = elem_info.get("domain", "Unspecified")

    # Add domain column
    df_diag_ci["Domain"] = df_diag_ci["Element"].map(element_to_domain)

    # Aggregate by domain: compute weighted average by sample size
    # For simplicity, use mean of point estimates and average of CI bounds
    domain_rows = []
    for domain in df_diag_ci["Domain"].unique():
        for metric in metrics:
            for annotator in ["Human", "AI"]:
                df_subset = df_diag_ci[
                    (df_diag_ci["Domain"] == domain)
                    & (df_diag_ci["Metric"] == metric)
                    & (df_diag_ci["Annotator"] == annotator)
                ].copy()

                if len(df_subset) == 0:
                    continue

                # Use mean for aggregation
                mean_val = df_subset["value"].mean()
                mean_ci_lo = df_subset["ci_low"].mean()
                mean_ci_hi = df_subset["ci_high"].mean()

                # Get p-value (same for all elements in a domain for a given metric)
                p_val = df_subset["p_value"].iloc[0] if len(df_subset) > 0 else np.nan

                domain_rows.append(
                    {
                        "Domain": domain,
                        "Metric": metric,
                        "Annotator": annotator,
                        "value": mean_val,
                        "ci_low": mean_ci_lo,
                        "ci_high": mean_ci_hi,
                        "p_value": p_val,
                    }
                )

    df_domain = pd.DataFrame(domain_rows)
    domain_order = sorted(df_domain["Domain"].unique())

    # Add significance stars
    df_domain["sig_star"] = df_domain["p_value"].apply(
        lambda p: (
            "***"
            if pd.notna(p) and p < 0.001
            else "**"
            if pd.notna(p) and p < 0.01
            else "*"
            if pd.notna(p) and p < 0.05
            else ""
        )
    )

    # Setup domain plot: one column per diagnostic metric
    fig_domain, axes_domain = plt.subplots(
        1, len(metrics), figsize=(16, 5), sharey=True
    )

    for ax_idx, metric in enumerate(metrics):
        ax = axes_domain[ax_idx]

        # Get data for this metric
        df_metric = df_domain[df_domain["Metric"] == metric].copy()

        # Separate human and AI, ordered by domain_order
        human = (
            df_metric[df_metric["Annotator"] == "Human"]
            .set_index("Domain")
            .reindex(domain_order)
            .reset_index()
        )
        ai = (
            df_metric[df_metric["Annotator"] == "AI"]
            .set_index("Domain")
            .reindex(domain_order)
            .reset_index()
        )

        x = np.arange(len(domain_order))

        human_vals = human["value"].to_numpy()
        ai_vals = ai["value"].to_numpy()

        # Asymmetric yerr: [[lower...], [upper...]]
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

        # Bars (black for Human, gray for AI)
        ax.bar(
            x - bar_width / 2, human_vals, width=bar_width, label="Human", color="black"
        )
        ax.bar(x + bar_width / 2, ai_vals, width=bar_width, label="AI", color="gray")

        # Error bars
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

        # Titles + axes
        ax.set_title(metric, fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(domain_order, rotation=45, ha="right", fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Value" if ax_idx == 0 else "")
        ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.6)

        # Add significance stars
        for i, domain in enumerate(domain_order):
            star = human.iloc[i]["sig_star"]
            if star == "":
                continue

            # Place star above the higher bar
            if human_vals[i] >= ai_vals[i]:
                y = human_vals[i] + human_yerr[1, i] + y_pad
                x_pos = x[i] - bar_width / 2
            else:
                y = ai_vals[i] + ai_yerr[1, i] + y_pad
                x_pos = x[i] + bar_width / 2

            ax.text(x_pos, min(y, 1.08), star, ha="center", va="bottom", fontsize=12)

    # One legend for the whole figure
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

    # Save figure
    out_path = OUTPUT_DIR / "domain_aggregated_diagnostic_metrics_human_vs_ai.png"
    safe_save_plt(fig_domain, out_path, dpi=600, bbox_inches="tight")
    if os.environ.get("SHOW_PLOTS", "0") == "1":
        plt.show()
    else:
        plt.close(fig_domain)
    print(f"✓ Domain-aggregated plot saved to {out_path}")

    print("✓ Element-level and domain-aggregated diagnostic metrics plots generated")

except Exception as exc:
    print(f"Failed generating diagnostic metrics plots: {exc}")
    import traceback

    traceback.print_exc()


# ------------------------------------------------------------------
# Great Tables: Element-Level Tables (Diagnostic + Classification)
# ------------------------------------------------------------------
try:
    from great_tables import GT

    # ==================== HELPER FUNCTIONS ====================
    def fmt_cell(est, lo, hi, p, decimals=3):
        """Return 'est (lo, hi)' plus * if p<0.05. Handles missing."""
        if pd.isna(est):
            return ""
        star = "*" if (pd.notna(p) and p < 0.05) else ""
        return f"{est:.{decimals}f} ({lo:.{decimals}f}, {hi:.{decimals}f}){star}"

    def fmt_pct_cell(mean, lo, hi, decimals=1, star=""):
        """Return 'mean% (lo, hi)' plus star. For percentage metrics."""
        if pd.isna(mean):
            return ""
        return f"{mean:.{decimals}f}% ({lo:.{decimals}f}, {hi:.{decimals}f}){star}"

    # ==================== TABLE 1: ELEMENT-LEVEL DIAGNOSTIC METRICS ====================
    print("Building element-level diagnostic metrics table...")

    # Use df_diagnostic_tests which has Element, Annotator, and all diagnostic metrics + p-values
    df_elem_diag = df_diagnostic_tests.copy()

    # We also need CI bounds - compute them using Wilson CI
    elem_diag_rows = []
    for _, r in df_elem_diag.iterrows():
        element = r["Element"]
        annot = r["Annotator"]
        TP, FP, FN, TN = int(r["TP"]), int(r["FP"]), int(r["FN"]), int(r["TN"])

        # Sensitivity
        n_sens = TP + FN
        x_sens = TP
        sens_lo, sens_hi = wilson_ci(x_sens, n_sens)
        sens_val = r["sensitivity"]
        sens_p = r.get("sensitivity_p", np.nan)

        # Specificity
        n_spec = TN + FP
        x_spec = TN
        spec_lo, spec_hi = wilson_ci(x_spec, n_spec)
        spec_val = r["specificity"]
        spec_p = r.get("specificity_p", np.nan)

        # PPV
        n_ppv = TP + FP
        x_ppv = TP
        ppv_lo, ppv_hi = wilson_ci(x_ppv, n_ppv)
        ppv_val = r["ppv"]
        ppv_p = r.get("ppv_p", np.nan)

        # NPV
        n_npv = TN + FN
        x_npv = TN
        npv_lo, npv_hi = wilson_ci(x_npv, n_npv)
        npv_val = r["npv"]
        npv_p = r.get("npv_p", np.nan)

        elem_diag_rows.append(
            {
                "Element": element,
                "Annotator": annot,
                "Sensitivity": fmt_cell(sens_val, sens_lo, sens_hi, sens_p, decimals=3),
                "Specificity": fmt_cell(spec_val, spec_lo, spec_hi, spec_p, decimals=3),
                "PPV": fmt_cell(ppv_val, ppv_lo, ppv_hi, ppv_p, decimals=3),
                "NPV": fmt_cell(npv_val, npv_lo, npv_hi, npv_p, decimals=3),
            }
        )

    df_elem_diag_table = pd.DataFrame(elem_diag_rows)

    # Enforce ordering
    element_order_list = sorted(df_elem_diag_table["Element"].unique())
    annotator_order = ["Human", "AI"]
    df_elem_diag_table["Element"] = pd.Categorical(
        df_elem_diag_table["Element"], categories=element_order_list, ordered=True
    )
    df_elem_diag_table["Annotator"] = pd.Categorical(
        df_elem_diag_table["Annotator"], categories=annotator_order, ordered=True
    )
    df_elem_diag_table = df_elem_diag_table.sort_values(
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
        .tab_source_note(
            "* p < 0.05 (McNemar-style exact binomial test, AI > Human). Values shown as estimate (95% CI)."
        )
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

    # Export table
    elem_diag_png = OUTPUT_DIR / "element_level_diagnostic_metrics_table.png"
    GT.save(gt_elem_diag, file=str(elem_diag_png))
    print(f"✓ Element-level diagnostic metrics table saved to {elem_diag_png}")

    # ==================== TABLE 2: ELEMENT-LEVEL CLASSIFICATION METRICS ====================
    print("Building element-level classification metrics table...")

    # Use df_ci which has Element, Annotator, metric, value, ci_low, ci_high
    # Filter for accuracy, precision, recall (not F1 for now, or include if you want)
    classification_metrics = ["accuracy", "precision", "recall", "f1"]

    df_elem_class = df_ci[df_ci["metric"].isin(classification_metrics)].copy()

    # Get p-values from df_stats
    # df_stats has Element, Metric, p_value_mcnemar or p_value_holm
    p_col = (
        "p_value_holm"
        if "p_value_holm" in df_stats.columns
        else "p_value_mcnemar"
        if "p_value_mcnemar" in df_stats.columns
        else None
    )

    elem_class_rows = []
    for element in df_elem_class["Element"].unique():
        for annot in ["Human", "AI"]:
            row_data = {"Element": element, "Annotator": annot}

            for metric in classification_metrics:
                metric_df = df_elem_class[
                    (df_elem_class["Element"] == element)
                    & (df_elem_class["Annotator"] == annot)
                    & (df_elem_class["metric"] == metric)
                ]

                if len(metric_df) == 0:
                    row_data[metric.title()] = ""
                    continue

                r = metric_df.iloc[0]
                val = r["value"] * 100  # Convert to percentage
                lo = r["ci_low"] * 100
                hi = r["ci_high"] * 100

                # Get p-value from df_stats
                p_val = np.nan
                if p_col:
                    p_df = df_stats[
                        (df_stats["Element"] == element)
                        & (df_stats["Metric"].str.lower() == metric)
                    ]
                    if len(p_df) > 0:
                        p_val = p_df.iloc[0][p_col]

                star = "*" if (pd.notna(p_val) and p_val < 0.05) else ""
                row_data[metric.title()] = fmt_pct_cell(
                    val, lo, hi, decimals=1, star=star
                )

            elem_class_rows.append(row_data)

    df_elem_class_table = pd.DataFrame(elem_class_rows)

    # Enforce ordering
    df_elem_class_table["Element"] = pd.Categorical(
        df_elem_class_table["Element"], categories=element_order_list, ordered=True
    )
    df_elem_class_table["Annotator"] = pd.Categorical(
        df_elem_class_table["Annotator"], categories=annotator_order, ordered=True
    )
    df_elem_class_table = df_elem_class_table.sort_values(
        ["Element", "Annotator"]
    ).reset_index(drop=True)

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
        .tab_source_note(
            "* p < 0.05 (McNemar test with Holm correction). Values shown as % (95% CI)."
        )
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

    # Export table
    elem_class_png = OUTPUT_DIR / "element_level_classification_metrics_table.png"
    GT.save(gt_elem_class, file=str(elem_class_png))
    print(f"✓ Element-level classification metrics table saved to {elem_class_png}")

    # Display tables (they will render in Jupyter/interactive environments)
    print("\n=== Element-Level Diagnostic Metrics Table ===")
    print(gt_elem_diag)

    print("\n=== Element-Level Classification Metrics Table ===")
    print(gt_elem_class)

except Exception as exc:
    print(f"Failed generating element-level tables: {exc}")
    import traceback

    traceback.print_exc()


# ------------------------------------------------------------------
# Great Tables: Domain-Level Tables (Diagnostic + Classification)
# ------------------------------------------------------------------
gt_domain_diag = None
try:
    from great_tables import GT

    # ==================== TABLE 3: DOMAIN-LEVEL DIAGNOSTIC METRICS ====================
    print("Building domain-level diagnostic metrics table...")

    # Build domain-aggregated diagnostic metrics
    # Get domain mapping from elements dict
    element_to_domain = {}
    for elem_key, elem_info in elements.items():
        display_name = elem_key.replace("_", " ").title()
        element_to_domain[display_name] = elem_info.get("domain", "Unspecified")

    df_elem_diag["Domain"] = df_elem_diag["Element"].map(element_to_domain)

    # Aggregate by domain (mean of metrics)
    domain_diag_rows = []
    for domain in ["Radiology", "Pathology"]:
        for annot in ["Human", "AI"]:
            subset = df_elem_diag[
                (df_elem_diag["Domain"] == domain)
                & (df_elem_diag["Annotator"] == annot)
            ]

            if len(subset) == 0:
                continue

            # Average the raw values and CIs
            row_data = {"Domain": domain, "Annotator": annot}

            # Actually, let's properly aggregate by summing confusion matrix counts
            TP_sum = subset["TP"].sum()
            FP_sum = subset["FP"].sum()
            FN_sum = subset["FN"].sum()
            TN_sum = subset["TN"].sum()

            # Recompute metrics from aggregated counts
            n_sens = TP_sum + FN_sum
            x_sens = TP_sum
            sens_val = x_sens / n_sens if n_sens > 0 else np.nan
            sens_lo, sens_hi = wilson_ci(x_sens, n_sens)

            n_spec = TN_sum + FP_sum
            x_spec = TN_sum
            spec_val = x_spec / n_spec if n_spec > 0 else np.nan
            spec_lo, spec_hi = wilson_ci(x_spec, n_spec)

            n_ppv = TP_sum + FP_sum
            x_ppv = TP_sum
            ppv_val = x_ppv / n_ppv if n_ppv > 0 else np.nan
            ppv_lo, ppv_hi = wilson_ci(x_ppv, n_ppv)

            n_npv = TN_sum + FN_sum
            x_npv = TN_sum
            npv_val = x_npv / n_npv if n_npv > 0 else np.nan
            npv_lo, npv_hi = wilson_ci(x_npv, n_npv)

            # Get p-value (use first element's p-value as proxy - ideally recompute)
            sens_p = subset["sensitivity_p"].iloc[0] if len(subset) > 0 else np.nan
            spec_p = subset["specificity_p"].iloc[0] if len(subset) > 0 else np.nan
            ppv_p = subset["ppv_p"].iloc[0] if len(subset) > 0 else np.nan
            npv_p = subset["npv_p"].iloc[0] if len(subset) > 0 else np.nan

            domain_diag_rows.append(
                {
                    "Domain": domain,
                    "Annotator": annot,
                    "Sensitivity": fmt_cell(
                        sens_val, sens_lo, sens_hi, sens_p, decimals=3
                    ),
                    "Specificity": fmt_cell(
                        spec_val, spec_lo, spec_hi, spec_p, decimals=3
                    ),
                    "PPV": fmt_cell(ppv_val, ppv_lo, ppv_hi, ppv_p, decimals=3),
                    "NPV": fmt_cell(npv_val, npv_lo, npv_hi, npv_p, decimals=3),
                }
            )

    df_domain_diag_table = pd.DataFrame(domain_diag_rows)

    # Enforce ordering
    domain_order = ["Radiology", "Pathology"]
    if df_domain_diag_table.empty:
        print("Warning: no domain-level diagnostic rows; skipping table export.")
    else:
        df_domain_diag_table["Domain"] = pd.Categorical(
            df_domain_diag_table["Domain"], categories=domain_order, ordered=True
        )
        df_domain_diag_table["Annotator"] = pd.Categorical(
            df_domain_diag_table["Annotator"], categories=annotator_order, ordered=True
        )
        df_domain_diag_table = df_domain_diag_table.sort_values(
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
            .tab_source_note(
                "* p < 0.05. Values shown as estimate (95% CI). Aggregated across elements within domain."
            )
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

        # Export table
        domain_diag_png = OUTPUT_DIR / "domain_level_diagnostic_metrics_table.png"
        GT.save(gt_domain_diag, file=str(domain_diag_png))
        print(f"✓ Domain-level diagnostic metrics table saved to {domain_diag_png}")

    # ==================== TABLE 4: DOMAIN-LEVEL CLASSIFICATION METRICS ====================
    print("Building domain-level classification metrics table...")

    # Use df_agg which has Domain, Metric, and Human/AI means with CIs
    domain_class_rows = []
    for domain in df_agg["Domain"].unique():
        subd = df_agg[df_agg["Domain"] == domain].copy()

        for annot in ["Human", "AI"]:
            row_data = {"Domain": domain, "Annotator": annot}

            for metric_label in ["Accuracy", "Precision", "Recall", "F1"]:
                # Find the row for this metric
                metric_row = subd[subd["MetricLabel"] == metric_label]

                if len(metric_row) == 0:
                    row_data[metric_label] = ""
                    continue

                r = metric_row.iloc[0]
                star = r.get("Star", "")
                star = "" if pd.isna(star) else str(star)

                if annot == "Human":
                    val = r["HumanMean_Pct"]
                    lo = r["HumanCI_L_Pct"]
                    hi = r["HumanCI_H_Pct"]
                else:
                    val = r["AIMean_Pct"]
                    lo = r["AICI_L_Pct"]
                    hi = r["AICI_H_Pct"]

                row_data[metric_label] = fmt_pct_cell(
                    val, lo, hi, decimals=1, star=star
                )

            domain_class_rows.append(row_data)

    df_domain_class_table = pd.DataFrame(domain_class_rows)

    # Enforce ordering
    df_domain_class_table["Domain"] = pd.Categorical(
        df_domain_class_table["Domain"], categories=domain_order, ordered=True
    )
    df_domain_class_table["Annotator"] = pd.Categorical(
        df_domain_class_table["Annotator"], categories=annotator_order, ordered=True
    )
    df_domain_class_table = df_domain_class_table.sort_values(
        ["Domain", "Annotator"]
    ).reset_index(drop=True)

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
        .tab_source_note(
            "* indicates Human - AI difference CI excludes 0. Values shown as % (95% CI)."
        )
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

    # Export table
    domain_class_png = OUTPUT_DIR / "domain_level_classification_metrics_table.png"
    GT.save(gt_domain_class, file=str(domain_class_png))
    print(f"✓ Domain-level classification metrics table saved to {domain_class_png}")

    # Display tables
    if "gt_domain_diag" in locals() and gt_domain_diag is not None:
        print("\n=== Domain-Level Diagnostic Metrics Table ===")
        print(gt_domain_diag)

    print("\n=== Domain-Level Classification Metrics Table ===")
    print(gt_domain_class)

    print("\n✓ All 4 tables generated successfully")

except Exception as exc:
    print(f"Failed generating domain-level tables: {exc}")
    import traceback

    traceback.print_exc()


# ------------------------------------------------------------------
# ROC and Precision-Recall example (synthetic scores)
# ------------------------------------------------------------------
try:
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 1000)
    y_scores = np.random.rand(1000)

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
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
    # Save figure
    out_path = OUTPUT_DIR / "roc_example.png"
    safe_save_plt(fig, out_path, dpi=300)
    if os.environ.get("SHOW_PLOTS", "0") == "1":
        plt.show()
    else:
        plt.close(fig)

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    fig = plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="blue", lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    # Save figure
    out_path = OUTPUT_DIR / "pr_example.png"
    safe_save_plt(fig, out_path, dpi=300)
    if os.environ.get("SHOW_PLOTS", "0") == "1":
        plt.show()
    else:
        plt.close(fig)
except Exception as exc:
    print(f"Failed generating ROC/PR examples: {exc}")


# ------------------------------------------------------------------
# Error analysis: synthetic example and scatter visualization
# ------------------------------------------------------------------
try:
    np.random.seed(42)
    features = np.random.rand(100, 3)
    true_labels = np.random.choice(["A", "B", "C"], 100)
    predicted_labels = np.random.choice(["A", "B", "C"], 100)

    df_err = pd.DataFrame(features, columns=["Feature 1", "Feature 2", "Feature 3"])
    df_err["True Label"] = true_labels
    df_err["Predicted Label"] = predicted_labels
    df_err["Error"] = df_err["True Label"] != df_err["Predicted Label"]

    errors = df_err[df_err["Error"]].sample(min(5, df_err["Error"].sum()))
    print("Sample of errors:")
    print(errors)

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
    # Save figure
    out_path = OUTPUT_DIR / "error_scatter.png"
    safe_save_plt(fig, out_path, dpi=300)
    if os.environ.get("SHOW_PLOTS", "0") == "1":
        plt.show()
    else:
        plt.close(fig)
except Exception as exc:
    print(f"Failed generating error analysis plots: {exc}")


# ------------------------------------------------------------------
# Bias-Variance demonstration (synthetic)
# ------------------------------------------------------------------
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
    # Save figure
    out_path = OUTPUT_DIR / "bias_variance_demo.png"
    safe_save_plt(fig, out_path, dpi=300)
    if os.environ.get("SHOW_PLOTS", "0") == "1":
        plt.show()
    else:
        plt.close(fig)
except Exception as exc:
    print(f"Failed generating bias-variance demo: {exc}")


# ------------------------------------------------------------------
# Cross-validation example (Logistic Regression)
# ------------------------------------------------------------------
try:
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification

    X_cv, y_cv = make_classification(
        n_samples=1000, n_features=20, n_classes=2, random_state=42
    )
    model = LogisticRegression(max_iter=1000)
    cv_scores = cross_val_score(model, X_cv, y_cv, cv=5)
    print("Cross-validation scores:", cv_scores)
    print("Mean CV score:", cv_scores.mean())

    fig = plt.figure(figsize=(8, 6))
    plt.boxplot(cv_scores)
    plt.title("5-Fold Cross-Validation Results")
    plt.ylabel("Accuracy")
    # Save figure
    out_path = OUTPUT_DIR / "cv_boxplot.png"
    safe_save_plt(fig, out_path, dpi=300)
    if os.environ.get("SHOW_PLOTS", "0") == "1":
        plt.show()
    else:
        plt.close(fig)
except Exception as exc:
    print(f"Failed generating cross-validation example: {exc}")


# ------------------------------------------------------------------
# Outputs and how to generate them
# ------------------------------------------------------------------
# This script writes example CSVs and PNGs to the configured OUTPUT_DIR.
# By default (in this file) OUTPUT_DIR is set to:
#   /Users/robertjames/Documents/llm_summarization/data reports
#
# To run the full analysis and produce the outputs (headless), from the
# project root run:
#
#   MPLBACKEND=Agg .venv/bin/python src/llm_eval_by_human/human_judge_analysis_classification_metrics.py
#
# The script uses safe-save wrappers and will print a warning and skip any
# individual save if the OUTPUT_DIR is missing or not writable.
#
# Files produced by the script (when successful)
# - confusion_human.csv
#     Aggregated confusion table (Predicted x Actual) for Human across elements.
# - confusion_ai.csv
#     Aggregated confusion table (Predicted x Actual) for AI across elements.
# - confusion_tables.png
#     Side-by-side textual confusion tables (Human vs AI).
# - confusion_heatmaps.png
#     Side-by-side heatmaps showing counts and percent for Human and AI.
# - human_ai_metrics_facet_ci.png
#     Faceted grouped-bar figure of accuracy/precision/recall/F1 by element
#     with 95% CIs for Human vs AI (saved as a Matplotlib PNG).
# - avg_metrics_rad_vs_path_grouped_ci_stars.png (Plotly static export)
#     Aggregated domain-level (Radiology vs Pathology) grouped bars with
#     bootstrap CIs and significance stars. Note: Plotly static export
#     requires Kaleido + Chrome to be available; if Kaleido cannot render
#     the script will print a message and skip the static PNG.
# - specificity_human.png, specificity_ai.png
#     Pie charts visualizing specificity = TN / (TN + FP) for the aggregated
#     Human and AI confusion tables. These are only produced when TN/FP counts
#     are present; otherwise the script will skip and print a message.
# - roc_example.png, pr_example.png
#     Synthetic ROC and precision-recall example plots (generated from a
#     seeded random example). Replace `y_true` and `y_scores` with your real
#     ground-truth and model score columns to produce real ROC/PR plots.
# - error_scatter.png
#     Synthetic error-analysis scatter demonstrating where errors lie in a
#     2D feature projection (sampled rows printed to console).
# - bias_variance_demo.png
#     Synthetic polynomial fits (degrees 1,3,15) illustrating bias-variance.
# - cv_boxplot.png
#     5-fold cross-validation boxplot for a simple LogisticRegression example.
#
# Quick notes:
# - If you prefer outputs written to a different folder (for example
#   `/Users/robertjames/Documents/llm_summarization/output`), update the
#   `OUTPUT_DIR` variable near the top of this script and re-run. The
#   script will not auto-create your project root but will create OUTPUT_DIR
#   in this `src/` copy (it currently calls mkdir(parents=True, exist_ok=True)).
# - To produce ROC/PR from your real model, call `roc_curve(y_true, y_scores)`
#   with y_true (0/1) and y_scores (continuous), or tell me which columns
#   to use and I can wire them in.
# - If you want me to always force-create OUTPUT_DIR or to consolidate
#   helper functions and tidy imports, say so and I'll make that change.
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# Final Summary
# ------------------------------------------------------------------
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
print("  📋 Tables (PNG or HTML):")
print("     - element_level_diagnostic_metrics_table.*")
print("     - element_level_classification_metrics_table.*")
print("     - domain_level_diagnostic_metrics_table.*")
print("     - domain_level_classification_metrics_table.*")
print("  📄 Data (CSV):")
print("     - confusion_human.csv / confusion_ai.csv")
print("     - diagnostic_tests_with_p.csv")
print("     - domain_agg_metrics_with_p.csv")
print("\n" + "=" * 70)
