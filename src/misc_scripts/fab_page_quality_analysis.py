"""
Fabrication × Page-Level Document Quality Analysis
====================================================
For each fabricated feature (AI label = 3), identifies the relevant document
type (radiology vs pathology), pulls page-level quality metrics from
fab_page_level_doc_quality.csv, and tests whether quality metrics differ
between pages in cases where that feature was fabricated vs correctly extracted.

Outputs:
  reports/fab_quality_results.csv         — long-format merged table
  reports/fab_quality_stats.csv           — Mann-Whitney U + correlation per metric
  reports/fab_page_quality_analysis.png   — plots
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

sns.set_style("whitegrid")

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FAB_DIR      = PROJECT_ROOT / "data" / "fab_source"
QUAL_CSV     = PROJECT_ROOT / "data" / "features" / "fab_page_level_doc_quality.csv"
REPORTS_DIR  = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Feature → doc_type + domain mapping ───────────────────────────────────────
FEATURE_META = {
    "lesion_size":                       {"ai_col": "lesion_size_status_ai",                        "domain": "radiology"},
    "laterality":                        {"ai_col": "laterality_status_ai",                         "domain": "radiology"},
    "lesion_location":                   {"ai_col": "lesion_location_status_ai",                    "domain": "radiology"},
    "calcifications_asymmetry":          {"ai_col": "calcifications_asymmetry_status_ai",           "domain": "radiology"},
    "additional_enhancement_mri":        {"ai_col": "additional_enhancement_mri_status_ai",         "domain": "radiology"},
    "extent":                            {"ai_col": "extent_status_ai",                             "domain": "radiology"},
    "accurate_clip_placement":           {"ai_col": "accurate_clip_placement_status_ai",            "domain": "radiology"},
    "workup_recommendation":             {"ai_col": "workup_recommendation_status_ai",              "domain": "radiology"},
    "lymph_node":                        {"ai_col": "Lymph node_status_ai",                         "domain": "radiology"},
    "chronology_preserved":              {"ai_col": "chronology_preserved_status_ai",               "domain": "radiology"},
    "biopsy_method":                     {"ai_col": "biopsy_method_status_ai",                      "domain": "pathology"},
    "invasive_component_size":           {"ai_col": "invasive_component_size_pathology_status_ai",  "domain": "pathology"},
    "histologic_diagnosis":              {"ai_col": "histologic_diagnosis_status_ai",               "domain": "pathology"},
    "receptor_status":                   {"ai_col": "receptor_status_ai",                           "domain": "pathology"},
}

QUALITY_METRICS = [
    "laplacian_var",
    "tenengrad",
    "rms_contrast",
    "intensity_spread_p95_p5",
    "mean_brightness",
    "skew_angle_deg",
]

METRIC_LABELS = {
    "laplacian_var":           "Sharpness\n(Laplacian Var)",
    "tenengrad":               "Sharpness\n(Tenengrad)",
    "rms_contrast":            "RMS Contrast",
    "intensity_spread_p95_p5": "Intensity Spread\n(P95−P5)",
    "mean_brightness":         "Mean Brightness",
    "skew_angle_deg":          "Skew Angle (°)",
}

# Surgeon last-name → case_folder prefix map
SURGEON_PREFIX = {
    "Barrio":    "AB",
    "Capko":     "DC",
    "El-Tamer":  "MET",
    "Heerdt":    "AH",
    "Kirstein":  "LK",
    "Lee":       "ML",
    "Montag":    "GM",
    "Moo":       "TM",
    "Pawloski":  "KP",
    "Plitas":    "GP",
    "Tadros":    "AT",
}


# ==============================================================================
# Step 1 — Load data
# ==============================================================================
print("Loading data...")
val_df = pd.read_excel(FAB_DIR / "llm_validation_failure_analysis.xlsx", sheet_name="AI_Has_3 ")
dq     = pd.read_csv(QUAL_CSV)

print(f"  Fabrication cases   : {len(val_df)}")
print(f"  Quality CSV rows    : {len(dq)}  ({dq['case_folder'].nunique()} case folders)")


# ==============================================================================
# Step 2 — Map fabrication cases → case_folder
# ==============================================================================
def infer_case_folder(surgeon_full: str, patient_init: str, tumor_val) -> str:
    """Derive case_folder key from surgeon name + patient initials."""
    last = str(surgeon_full).split(",")[0].strip()
    prefix = SURGEON_PREFIX.get(last, last[:2].upper())
    suffix = "DCIS" if str(tumor_val).strip() in ("0", "DCIS") else "INV"
    return f"{prefix}_{patient_init}_{suffix}"

# Build case_folder column — try exact match against known quality folders
known_folders = set(dq["case_folder"].unique())

def find_folder(surgeon, pi, tumor):
    candidates = [
        infer_case_folder(surgeon, pi, tumor),
        f"{SURGEON_PREFIX.get(str(surgeon).split(',')[0].strip(), '')}_{ pi}_invasive",
        f"DC_{pi}_invasive",  # Capko edge case
    ]
    for c in candidates:
        if c in known_folders:
            return c
    # Prefix + initials partial match
    prefix = SURGEON_PREFIX.get(str(surgeon).split(",")[0].strip(), "")
    for f in known_folders:
        if f.startswith(f"{prefix}_{pi}"):
            return f
    return None

val_df["case_folder"] = val_df.apply(
    lambda r: find_folder(r["surgeon"], r["patient_initials"], r.get("tumor_invasive_dcis", 1)),
    axis=1
)

matched = val_df["case_folder"].notna().sum()
print(f"  Matched to quality CSV: {matched} / {len(val_df)} fabrication cases")
unmatched = val_df[val_df["case_folder"].isna()][["surgeon", "patient_initials"]].values.tolist()
if unmatched:
    print(f"  Unmatched: {unmatched}")


# ==============================================================================
# Step 3 — Melt AI labels → long format: (case_folder, feature, is_fabricated)
# ==============================================================================
feature_rows = []

for _, row in val_df.iterrows():
    cf = row["case_folder"]
    if cf is None:
        continue
    for feat, meta in FEATURE_META.items():
        ai_col = meta["ai_col"]
        if ai_col not in val_df.columns:
            continue
        ai_val = row[ai_col]
        if pd.isna(ai_val):
            continue
        try:
            ai_int = int(float(ai_val))
        except (ValueError, TypeError):
            continue
        feature_rows.append({
            "case_folder":    cf,
            "feature":        feat,
            "domain":         meta["domain"],
            "ai_label":       ai_int,
            "is_fabricated":  ai_int == 3,
        })

df_labels = pd.DataFrame(feature_rows)
print(f"\nLong-format labels: {len(df_labels)} (case × feature) pairs")
print(f"  Fabricated (ai=3): {df_labels['is_fabricated'].sum()}")
print(f"  Fabrications by feature:")
fab_by_feat = df_labels[df_labels["is_fabricated"]].groupby("feature").size().sort_values(ascending=False)
for feat, n in fab_by_feat.items():
    print(f"    {feat:<40} : {n}")


# ==============================================================================
# Step 4 — Aggregate page quality per (case_folder, domain)
# ==============================================================================
# Per page metrics for each (case_folder, doc_type) — doc_type matches domain
page_agg = (
    dq.groupby(["case_folder", "doc_type"])[QUALITY_METRICS + ["is_blurry", "is_low_contrast"]]
    .agg({
        **{m: ["mean", "min", "max", "std"] for m in QUALITY_METRICS},
        "is_blurry": "mean",
        "is_low_contrast": "mean",
    })
)
page_agg.columns = ["_".join(c).strip("_") for c in page_agg.columns]
page_agg = page_agg.reset_index()
page_agg.rename(columns={"doc_type": "domain"}, inplace=True)

print(f"\nPage quality aggregation: {len(page_agg)} (case_folder × domain) pairs")


# ==============================================================================
# Step 5 — Join labels with quality metrics
# ==============================================================================
df_merged = df_labels.merge(page_agg, on=["case_folder", "domain"], how="left")
print(f"After join: {len(df_merged)} rows, {df_merged[QUALITY_METRICS[0]+'_mean'].notna().sum()} with quality data")

# Save merged table
out_path = REPORTS_DIR / "fab_quality_results.csv"
df_merged.to_csv(out_path, index=False)
print(f"Saved → {out_path}")


# ==============================================================================
# Step 6 — Statistical tests: per quality metric, fabricated vs not
# ==============================================================================
print("\n" + "=" * 72)
print("STATISTICAL TESTS: fabricated vs non-fabricated feature instances")
print("=" * 72)

stat_rows = []
for metric in QUALITY_METRICS:
    mean_col = f"{metric}_mean"
    if mean_col not in df_merged.columns:
        continue
    fab_vals   = df_merged.loc[df_merged["is_fabricated"] == True,  mean_col].dropna()
    nofab_vals = df_merged.loc[df_merged["is_fabricated"] == False, mean_col].dropna()

    if len(fab_vals) < 3 or len(nofab_vals) < 3:
        continue

    # Mann-Whitney U
    u_stat, p_val = stats.mannwhitneyu(fab_vals, nofab_vals, alternative="two-sided")

    # Point-biserial correlation (is_fabricated binary vs metric)
    combined = df_merged[[mean_col, "is_fabricated"]].dropna()
    rho, p_rho = stats.pointbiserialr(
        combined["is_fabricated"].astype(int),
        combined[mean_col]
    )

    sig = "**" if p_val < 0.01 else ("*" if p_val < 0.05 else ("." if p_val < 0.10 else ""))

    print(f"\n{METRIC_LABELS.get(metric, metric)}")
    print(f"  Fabricated    : n={len(fab_vals):3d}  mean={fab_vals.mean():8.2f}  median={fab_vals.median():8.2f}")
    print(f"  Not fabricated: n={len(nofab_vals):3d}  mean={nofab_vals.mean():8.2f}  median={nofab_vals.median():8.2f}")
    print(f"  Mann-Whitney U: U={u_stat:.1f}  p={p_val:.4f} {sig}")
    print(f"  Point-biserial: rho={rho:+.3f}  p={p_rho:.4f}")

    stat_rows.append({
        "metric":         metric,
        "n_fabricated":   len(fab_vals),
        "n_not_fab":      len(nofab_vals),
        "mean_fab":       round(fab_vals.mean(), 4),
        "mean_nofab":     round(nofab_vals.mean(), 4),
        "median_fab":     round(fab_vals.median(), 4),
        "median_nofab":   round(nofab_vals.median(), 4),
        "mw_u":           round(u_stat, 2),
        "mw_p":           round(p_val, 4),
        "sig":            sig,
        "rho":            round(rho, 4),
        "rho_p":          round(p_rho, 4),
    })

df_stats = pd.DataFrame(stat_rows).sort_values("mw_p")
df_stats.to_csv(REPORTS_DIR / "fab_quality_stats.csv", index=False)
print(f"\nSaved stats → {REPORTS_DIR / 'fab_quality_stats.csv'}")

print("\n\nSUMMARY (sorted by p-value):")
print(df_stats[["metric", "mean_fab", "mean_nofab", "mw_p", "sig", "rho"]].to_string(index=False))


# ==============================================================================
# Step 7 — Domain-stratified analysis (radiology vs pathology features)
# ==============================================================================
print("\n" + "=" * 72)
print("DOMAIN-STRATIFIED: radiology vs pathology features")
print("=" * 72)

for domain in ["radiology", "pathology"]:
    sub = df_merged[df_merged["domain"] == domain]
    fab  = sub[sub["is_fabricated"] == True]
    nfab = sub[sub["is_fabricated"] == False]
    print(f"\n  {domain.upper()} — fab={len(fab)}  not_fab={len(nfab)}")
    for metric in QUALITY_METRICS[:3]:
        mc = f"{metric}_mean"
        if mc not in sub.columns:
            continue
        fv = fab[mc].dropna()
        nv = nfab[mc].dropna()
        if len(fv) < 2 or len(nv) < 2:
            continue
        u, p = stats.mannwhitneyu(fv, nv, alternative="two-sided")
        sig = "*" if p < 0.05 else ""
        print(f"    {metric:<35} fab={fv.mean():.2f}  no_fab={nv.mean():.2f}  p={p:.3f}{sig}")


# ==============================================================================
# Step 8 — Per-feature breakdown
# ==============================================================================
print("\n" + "=" * 72)
print("PER-FEATURE: mean quality for fabricated instances")
print("=" * 72)
primary_metric = "laplacian_var_mean"
if primary_metric in df_merged.columns:
    feat_summary = (
        df_merged.groupby(["feature", "is_fabricated"])[primary_metric]
        .agg(["mean", "count"])
        .reset_index()
        .pivot(index="feature", columns="is_fabricated", values=["mean", "count"])
    )
    feat_summary.columns = ["mean_not_fab", "mean_fab", "n_not_fab", "n_fab"]
    feat_summary["diff"] = feat_summary["mean_fab"] - feat_summary["mean_not_fab"]
    print(feat_summary.sort_values("diff").to_string())


# ==============================================================================
# Step 9 — Visualisation
# ==============================================================================
print("\nGenerating plots...")
fig, axes = plt.subplots(3, 3, figsize=(18, 16))
fig.suptitle("Fabrication × Page-Level Document Quality\n(All 14 Fabrication Cases, Relevant Doc Type per Feature)",
             fontsize=14, fontweight="bold", y=1.01)

# Plot 1–6: boxplots per quality metric
for idx, metric in enumerate(QUALITY_METRICS):
    ax = axes[idx // 3][idx % 3]
    mean_col = f"{metric}_mean"
    if mean_col not in df_merged.columns:
        ax.set_visible(False)
        continue
    plot_data = df_merged[[mean_col, "is_fabricated"]].dropna()
    plot_data["group"] = plot_data["is_fabricated"].map({True: "Fabricated\n(AI=3)", False: "Correct\n(AI≠3)"})

    # Boxplot
    groups = ["Correct\n(AI≠3)", "Fabricated\n(AI=3)"]
    group_data = [
        plot_data.loc[plot_data["group"] == g, mean_col].values
        for g in groups
    ]
    bp = ax.boxplot(group_data, labels=groups, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5))
    colors = ["#3498db", "#e74c3c"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Jitter overlay
    for i, (g, c) in enumerate(zip(groups, colors), 1):
        y = plot_data.loc[plot_data["group"] == g, mean_col].values
        x = np.random.normal(i, 0.06, size=len(y))
        ax.scatter(x, y, alpha=0.5, color=c, s=20, zorder=3)

    # Annotate stats
    row = df_stats[df_stats["metric"] == metric]
    if len(row) > 0:
        p = row.iloc[0]["mw_p"]
        rho = row.iloc[0]["rho"]
        sig = row.iloc[0]["sig"]
        ax.set_title(f"{METRIC_LABELS.get(metric, metric)}\np={p:.3f}{sig}  ρ={rho:+.3f}",
                     fontsize=9, fontweight="bold")
    else:
        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=9, fontweight="bold")
    ax.tick_params(axis="x", labelsize=8)

# Plot 7: correlation heatmap (metric × is_fabricated, stratified by domain)
ax7 = axes[2][0]
corr_data = {}
for metric in QUALITY_METRICS:
    mc = f"{metric}_mean"
    if mc not in df_merged.columns:
        continue
    sub = df_merged[["is_fabricated", "domain", mc]].dropna()
    for domain in ["radiology", "pathology"]:
        d = sub[sub["domain"] == domain]
        if len(d) > 5:
            rho, _ = stats.pointbiserialr(d["is_fabricated"].astype(int), d[mc])
            corr_data[f"{metric[:8]}\n({domain[:3]})"] = rho

if corr_data:
    corr_series = pd.Series(corr_data).sort_values()
    colors_bar = ["#e74c3c" if v < 0 else "#3498db" for v in corr_series.values]
    ax7.barh(corr_series.index, corr_series.values, color=colors_bar, edgecolor="black", alpha=0.8)
    ax7.axvline(0, color="black", linewidth=1)
    ax7.set_xlabel("Point-biserial ρ\n(positive = higher metric in fab cases)")
    ax7.set_title("Correlation: Quality Metric\nvs Fabrication (by domain)", fontweight="bold", fontsize=9)
    ax7.tick_params(axis="y", labelsize=7)
else:
    ax7.set_visible(False)

# Plot 8: fabrication rate per feature, sorted
ax8 = axes[2][1]
feat_fab_rate = (
    df_merged.groupby("feature")["is_fabricated"]
    .agg(lambda x: x.sum() / len(x) * 100)
    .sort_values()
)
dom_map = {f: FEATURE_META[f]["domain"] for f in feat_fab_rate.index}
bar_colors = ["#3498db" if dom_map.get(f) == "radiology" else "#e67e22" for f in feat_fab_rate.index]
ax8.barh(feat_fab_rate.index, feat_fab_rate.values, color=bar_colors, edgecolor="black", alpha=0.8)
blue_patch = mpatches.Patch(color="#3498db", alpha=0.8, label="Radiology")
orange_patch = mpatches.Patch(color="#e67e22", alpha=0.8, label="Pathology")
ax8.legend(handles=[blue_patch, orange_patch], fontsize=8)
ax8.set_xlabel("Fabrication Rate (%)")
ax8.set_title("Fabrication Rate per Feature\n(across 14 cases with quality data)", fontweight="bold", fontsize=9)
ax8.tick_params(axis="y", labelsize=7)

# Plot 9: blurry/low-contrast page rate per case, colored by total fabrications
ax9 = axes[2][2]
blurry_by_case = (
    dq.groupby("case_folder")["is_blurry"].mean() * 100
).reset_index()
blurry_by_case.columns = ["case_folder", "pct_blurry"]

fab_count_by_case = (
    df_merged[df_merged["is_fabricated"]].groupby("case_folder").size()
    .reset_index(name="n_fab_features")
)

blurry_joined = blurry_by_case.merge(fab_count_by_case, on="case_folder", how="left")
blurry_joined["n_fab_features"] = blurry_joined["n_fab_features"].fillna(0)
blurry_joined = blurry_joined.sort_values("pct_blurry", ascending=True)

sc = ax9.scatter(
    blurry_joined["pct_blurry"],
    blurry_joined["n_fab_features"],
    s=80, edgecolors="black", linewidths=0.5,
    c=blurry_joined["pct_blurry"], cmap="RdYlGn_r", vmin=0, vmax=40
)
for _, row in blurry_joined.iterrows():
    ax9.annotate(row["case_folder"], (row["pct_blurry"], row["n_fab_features"]),
                 fontsize=6, xytext=(3, 3), textcoords="offset points")
plt.colorbar(sc, ax=ax9, label="% Blurry Pages")
ax9.set_xlabel("% Blurry Pages in Case")
ax9.set_ylabel("N Fabricated Features")
rho9, p9 = stats.spearmanr(blurry_joined["pct_blurry"], blurry_joined["n_fab_features"])
ax9.set_title(f"Blurry Page Rate vs N Fabricated Features\nSpearman ρ={rho9:+.3f}  p={p9:.3f}",
              fontweight="bold", fontsize=9)

plt.tight_layout()
out_fig = REPORTS_DIR / "fab_page_quality_analysis.png"
plt.savefig(out_fig, dpi=300, bbox_inches="tight")
plt.show()
print(f"\nSaved → {out_fig}")
print("\nDone.")
