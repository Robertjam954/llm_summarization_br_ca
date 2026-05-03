"""
V1 Prompt Validation — Exploratory Data Analysis
=================================================
Validation data: 200 patients, human-authored vs AI (V1 prompt) summaries,
each assessed against source documents per clinical feature.

Encoding:
  source col : 0 = feature absent from source, 1 = feature present
  human/ai col: 'na' / NaN = not applicable (source absent)
               1 = correctly identified
               2 = omitted
               3 = fabricated

Outputs (saved to reports/eda_v1_validation/):
  1. feature_presence_table.csv  — counts of source==1 per feature
  2. fig1_feature_presence.png   — bar chart of source presence counts
  3. fig2_pct_correct.png        — side-by-side human vs AI % correct
  4. fig3_pct_omitted.png        — side-by-side human vs AI % omitted
  5. fig4_pct_fabricated.png     — side-by-side human vs AI % fabricated
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.patches import Patch

sys.stdout.reconfigure(encoding="utf-8")

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH    = (PROJECT_ROOT / "data" / "raw" /
                "merged_llm_summary_validation_datasheet_deidentified copy.xlsx")
OUT_DIR      = PROJECT_ROOT / "reports" / "eda_v1_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Feature map (display name → column roots) ─────────────────────────────────
ELEMENTS = {
    "Lesion Size":                       "lesion_size",
    "Lesion Laterality":                 "laterality",
    "Lesion Location":                   "lesion_location",
    "Calcifications / Asymmetry":        "calcifications_asymmetry",
    "Additional Enhancement (MRI)":      "additional_enhancement_mri",
    "Extent":                            "extent",
    "Accurate Clip Placement":           "accurate_clip_placement",
    "Workup Recommendation":             "workup_recommendation",
    "Lymph Node":                        "Lymph node",
    "Chronology Preserved":              "chronology_preserved",
    "Biopsy Method":                     "biopsy_method",
    "Invasive Component Size (Path.)":   "invasive_component_size_pathology",
    "Histologic Diagnosis":              "histologic_diagnosis",
    "Receptor Status":                   "receptor",
}

def src_col(root):  return f"{root}_status_source"
def hum_col(root):  return f"{root}_status_human"
def ai_col(root):   return f"{root}_status_ai"


# ── Style constants ────────────────────────────────────────────────────────────
HUM_COLOR   = "#0072B2"   # Okabe-Ito blue   — human
AI_COLOR    = "#CC79A7"   # Okabe-Ito reddish purple — AI
BAR_WIDTH   = 0.35
LABEL_FS    = 8.5
TITLE_FS    = 13
AXIS_FS     = 10
TICK_FS     = 8.5
FIG_W       = 14
FIG_H       = 6.5
FIG_H_SBS   = 7.2

plt.rcParams.update({
    "font.family":     "sans-serif",
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.grid":        True,
    "axes.grid.axis":   "y",
    "grid.alpha":       0.4,
    "grid.linestyle":   "--",
})


# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_excel(DATA_PATH, sheet_name="Sheet1")
print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

# Coerce all status columns to numeric (handles 'na', NaN, strings)
for display, root in ELEMENTS.items():
    for col in [src_col(root), hum_col(root), ai_col(root)]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            print(f"  WARN: column not found → {col}")

N = len(df)


# ── Helper: compute per-feature metric ─────────────────────────────────────────
def pct_metric(root, actor_col_fn, code):
    """
    For rows where source==1, what fraction have actor==code?
    Returns (count_numerator, count_denominator, pct).
    """
    src   = df[src_col(root)]
    actor = df[actor_col_fn(root)]
    denom = (src == 1).sum()
    numer = ((src == 1) & (actor == code)).sum()
    pct   = (numer / denom * 100) if denom > 0 else np.nan
    return int(numer), int(denom), pct


# ── Published 95% CI and p-values (Picture1.jpg) ─────────────────────────────
# CI format: (lower_pct, upper_pct). (0, 0) = zero count, no CI to show.
# p_acc: accuracy p-value (Human vs AI).
PUBLISHED = {
    "Lesion Size": {
        "human_acc_ci":  (92, 98),     "ai_acc_ci":  (91, 98),    "p_acc": 0.61,
        "human_omit_ci": (0.94, 6.2),  "ai_omit_ci": (0.94, 6.2),
        "human_fab_ci":  (0.18, 4.0),  "ai_fab_ci":  (0.66, 5.5),
    },
    "Lesion Laterality": {
        "human_acc_ci":  (97, 100),    "ai_acc_ci":  (97, 100),   "p_acc": 0.99,
        "human_omit_ci": (0.03, 3.2),  "ai_omit_ci": (0.03, 3.2),
        "human_fab_ci":  (0, 0),       "ai_fab_ci":  (0, 0),
    },
    "Lesion Location": {
        "human_acc_ci":  (95, 100),    "ai_acc_ci":  (95, 99),    "p_acc": 0.99,
        "human_omit_ci": (0.39, 4.7),  "ai_omit_ci": (0.65, 5.4),
        "human_fab_ci":  (0, 0),       "ai_fab_ci":  (0, 0),
    },
    "Calcifications / Asymmetry": {
        "human_acc_ci":  (87, 97),     "ai_acc_ci":  (93, 99),    "p_acc": 0.23,
        "human_omit_ci": (3, 12),      "ai_omit_ci": (0.61, 7.3),
        "human_fab_ci":  (0, 0),       "ai_fab_ci":  (0, 0),
    },
    "Additional Enhancement (MRI)": {
        "human_acc_ci":  (74, 93),     "ai_acc_ci":  (87, 99),    "p_acc": 0.07,
        "human_omit_ci": (6.7, 26),    "ai_omit_ci": (0.61, 13),
        "human_fab_ci":  (0, 0),       "ai_fab_ci":  (0, 0),
    },
    "Extent": {
        "human_acc_ci":  (77, 93),     "ai_acc_ci":  (88, 99),    "p_acc": 0.04,
        "human_omit_ci": (6.6, 21),    "ai_omit_ci": (0.87, 10),
        "human_fab_ci":  (0.06, 7.0),  "ai_fab_ci":  (0.06, 7.0),
    },
    "Accurate Clip Placement": {
        "human_acc_ci":  (80, 91),     "ai_acc_ci":  (93, 99),    "p_acc": 0.0001,
        "human_omit_ci": (9.3, 20),    "ai_omit_ci": (1.0, 6.7),
        "human_fab_ci":  (0, 0),       "ai_fab_ci":  (0.03, 3.5),
    },
    "Workup Recommendation": {
        "human_acc_ci":  (76, 88),     "ai_acc_ci":  (86, 95),    "p_acc": 0.02,
        "human_omit_ci": (12, 24),     "ai_omit_ci": (3.6, 12),
        "human_fab_ci":  (0, 0),       "ai_fab_ci":  (0.22, 4.9),
    },
    "Lymph Node": {
        "human_acc_ci":  (64, 78),     "ai_acc_ci":  (90, 98),    "p_acc": 0.0001,
        "human_omit_ci": (22, 36),     "ai_omit_ci": (2.3, 9.8),
        "human_fab_ci":  (0, 0),       "ai_fab_ci":  (0, 0),
    },
    "Chronology Preserved": {
        "human_acc_ci":  (94, 99),     "ai_acc_ci":  (96, 100),   "p_acc": 0.21,
        "human_omit_ci": (0.65, 5.4),  "ai_omit_ci": (0.03, 3.2),
        "human_fab_ci":  (0.03, 3.2),  "ai_fab_ci":  (0.03, 3.2),
    },
    "Biopsy Method": {
        "human_acc_ci":  (88, 96),     "ai_acc_ci":  (97, 100),   "p_acc": 0.01,
        "human_omit_ci": (3.7, 11),    "ai_omit_ci": (0, 0),
        "human_fab_ci":  (0.03, 3.2),  "ai_fab_ci":  (0.03, 3.2),
    },
    "Invasive Component Size (Path.)": {
        "human_acc_ci":  (23, 38),     "ai_acc_ci":  (88, 97),    "p_acc": 0.0001,
        "human_omit_ci": (62, 77),     "ai_omit_ci": (2.1, 10),
        "human_fab_ci":  (0, 0),       "ai_fab_ci":  (0.24, 5.4),
    },
    "Histologic Diagnosis": {
        "human_acc_ci":  (95, 99),     "ai_acc_ci":  (94, 99),    "p_acc": 0.74,
        "human_omit_ci": (0.64, 5.4),  "ai_omit_ci": (0.64, 5.4),
        "human_fab_ci":  (0, 0),       "ai_fab_ci":  (0.03, 3.2),
    },
    "Receptor Status": {
        "human_acc_ci":  (92, 98),     "ai_acc_ci":  (80, 90),    "p_acc": 0.002,
        "human_omit_ci": (1.3, 7.3),   "ai_omit_ci": (8.7, 19),
        "human_fab_ci":  (0.03, 3.4),  "ai_fab_ci":  (0.19, 4.3),
    },
}


def _sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""


# ── Table 1: Feature presence counts ──────────────────────────────────────────
presence_rows = []
for display, root in ELEMENTS.items():
    col = src_col(root)
    if col not in df.columns:
        continue
    count = int((df[col] == 1).sum())
    presence_rows.append({"Feature": display, "N present in source": count,
                           "N absent / N/A": N - count,
                           "Total cases": N,
                           "% present": f"{count / N * 100:.1f}%"})

presence_df = pd.DataFrame(presence_rows)
presence_df.to_csv(OUT_DIR / "feature_presence_table.csv", index=False)
print(f"\nFeature presence table saved ({len(presence_df)} features)")
print(presence_df.to_string(index=False))


# ── Figure 1: Feature presence bar chart ──────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(FIG_W, FIG_H))
fig1.subplots_adjust(bottom=0.22)

feat_names = presence_df["Feature"].tolist()
counts     = presence_df["N present in source"].tolist()
x          = np.arange(len(feat_names))

bars = ax1.bar(x, counts, color=HUM_COLOR, edgecolor="white",
               linewidth=0.8, zorder=3, width=0.6)

for bar, cnt in zip(bars, counts):
    ax1.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 1.5,
             str(cnt), ha="center", va="bottom",
             fontsize=LABEL_FS, fontweight="bold", color="#333333")

ax1.set_xticks(x)
ax1.set_xticklabels(feat_names, rotation=35, ha="right", fontsize=TICK_FS)
ax1.set_ylabel("Number of cases (out of 200)", fontsize=AXIS_FS)
ax1.set_title("Features Present in Source Documents",
              fontsize=TITLE_FS, fontweight="bold", pad=12)
ax1.set_ylim(0, max(counts) * 1.15)
ax1.axhline(N, color="#999999", linestyle="--", linewidth=0.8, zorder=2)
ax1.text(len(feat_names) - 0.4, N + 1.5, "n=200",
         ha="right", fontsize=8, color="#666666")

fig1.savefig(OUT_DIR / "fig1_feature_presence.png", dpi=150, bbox_inches="tight")
plt.close(fig1)
print("\nFigure 1 saved: fig1_feature_presence.png")


# ── Generic side-by-side bar plot helper ──────────────────────────────────────
def plot_side_by_side(code, title, ylabel, filename, y_max=None,
                      legend_below=False, hum_label="Human", ai_label="AI",
                      show_ci=False, show_stars=False):
    """
    For each feature where source==1, compute:
        human pct = (source==1 & human==code) / (source==1)
        ai    pct = (source==1 & ai==code)    / (source==1)
    Optionally overlays published 95% CI error bars and significance stars.
    """
    ci_key = {1: "acc", 2: "omit", 3: "fab"}[code]

    rows = []
    for display, root in ELEMENTS.items():
        _, denom_h, pct_h = pct_metric(root, hum_col, code)
        _, denom_a, pct_a = pct_metric(root, ai_col,  code)
        rows.append({"feature": display,
                     "human_pct": pct_h,
                     "ai_pct":    pct_a,
                     "n":         denom_h})
    plot_df = pd.DataFrame(rows)
    n_feat  = len(plot_df)

    # ── CI arrays from published table ───────────────────────────────────────
    human_ci_lo = np.zeros(n_feat)
    human_ci_hi = np.zeros(n_feat)
    ai_ci_lo    = np.zeros(n_feat)
    ai_ci_hi    = np.zeros(n_feat)

    if show_ci:
        for i, feat in enumerate(plot_df["feature"]):
            pub = PUBLISHED.get(feat, {})
            human_ci_lo[i], human_ci_hi[i] = pub.get(f"human_{ci_key}_ci", (0, 0))
            ai_ci_lo[i],    ai_ci_hi[i]    = pub.get(f"ai_{ci_key}_ci",    (0, 0))

    h_vals = plot_df["human_pct"].values
    a_vals = plot_df["ai_pct"].values

    def _yerr(vals, ci_lo, ci_hi):
        no_ci  = (ci_lo == 0) & (ci_hi == 0)
        lo_err = np.where(no_ci | np.isnan(vals), 0.0, np.maximum(0, vals - ci_lo))
        hi_err = np.where(no_ci | np.isnan(vals), 0.0, np.maximum(0, ci_hi - vals))
        return np.array([lo_err, hi_err])

    yerr_h = _yerr(h_vals, human_ci_lo, human_ci_hi) if show_ci else None
    yerr_a = _yerr(a_vals, ai_ci_lo,    ai_ci_hi)    if show_ci else None
    err_kw = dict(ecolor="#333333", elinewidth=1.2, capsize=4, zorder=4)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H_SBS))
    fig.subplots_adjust(bottom=0.40 if legend_below else 0.25,
                        top=0.90)
    x = np.arange(n_feat)

    bars_h = ax.bar(x - BAR_WIDTH / 2, h_vals,
                    width=BAR_WIDTH, color=HUM_COLOR,
                    edgecolor="white", linewidth=0.7, label=hum_label,
                    zorder=3, yerr=yerr_h,
                    error_kw=err_kw if show_ci else {})
    bars_a = ax.bar(x + BAR_WIDTH / 2, a_vals,
                    width=BAR_WIDTH, color=AI_COLOR,
                    edgecolor="white", linewidth=0.7, label=ai_label,
                    zorder=3, yerr=yerr_a,
                    error_kw=err_kw if show_ci else {})

    # ── Value labels (positioned above CI upper bound when CIs shown) ─────────
    for i, bar in enumerate(bars_h):
        h = bar.get_height()
        if not np.isnan(h) and h > 0:
            top = max(h, human_ci_hi[i]) if (show_ci and human_ci_hi[i] > 0) else h
            ax.text(bar.get_x() + bar.get_width() / 2, top + 0.8,
                    f"{h:.0f}%", ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold", color="#333333")
    for i, bar in enumerate(bars_a):
        h = bar.get_height()
        if not np.isnan(h) and h > 0:
            top = max(h, ai_ci_hi[i]) if (show_ci and ai_ci_hi[i] > 0) else h
            ax.text(bar.get_x() + bar.get_width() / 2, top + 0.8,
                    f"{h:.0f}%", ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold", color="#333333")

    # ── Significance stars (figure 2 only) ────────────────────────────────────
    if show_stars:
        for i, feat in enumerate(plot_df["feature"]):
            p     = PUBLISHED.get(feat, {}).get("p_acc", 1.0)
            stars = _sig_stars(p)
            if stars:
                bracket_y = max(human_ci_hi[i], ai_ci_hi[i]) + 2.5
                x_h = i - BAR_WIDTH / 2
                x_a = i + BAR_WIDTH / 2
                ax.plot([x_h, x_h, x_a, x_a],
                        [bracket_y - 1.0, bracket_y, bracket_y, bracket_y - 1.0],
                        color="#333333", lw=0.8, zorder=5)
                ax.text(i, bracket_y + 0.4, stars, ha="center", va="bottom",
                        fontsize=9, fontweight="bold", color="#333333")

    # ── n= annotation below feature names ────────────────────────────────────
    for i, row in plot_df.iterrows():
        ax.text(i, -8, f"n={row['n']}", ha="center", fontsize=7,
                color="#777777", transform=ax.get_xaxis_transform())

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["feature"], rotation=35, ha="right",
                       fontsize=TICK_FS)
    ax.set_ylabel(ylabel, fontsize=AXIS_FS)
    ax.set_title(title, fontsize=TITLE_FS, fontweight="bold", pad=12)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.set_ylim(0, (y_max or 105))

    if legend_below:
        ax.legend(
            handles=[Patch(facecolor=HUM_COLOR, label=hum_label),
                     Patch(facecolor=AI_COLOR,  label=ai_label)],
            fontsize=10, frameon=False, ncol=2,
            loc="upper center", bbox_to_anchor=(0.5, -0.36),
            borderaxespad=0,
        )
    else:
        ax.legend(
            handles=[Patch(facecolor=HUM_COLOR, label=hum_label),
                     Patch(facecolor=AI_COLOR,  label=ai_label)],
            fontsize=9.5, frameon=True, framealpha=0.9,
            loc="upper right",
        )

    fig.savefig(OUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filename}")
    return plot_df


# ── Figure 2: % Correct (code 1) ──────────────────────────────────────────────
correct_df = plot_side_by_side(
    code         = 1,
    title        = "Accurate (%) Annotations in Human and AI Generated Summaries"
                   " with Initial Prompt",
    ylabel       = "% Accurate",
    filename     = "fig2_pct_correct.png",
    y_max        = 115,
    legend_below = True,
    hum_label    = "Human",
    ai_label     = "AI",
    show_ci      = True,
    show_stars   = True,
)

# ── Figure 3: % Omitted (code 2) ──────────────────────────────────────────────
omitted_df = plot_side_by_side(
    code         = 2,
    title        = "Omitted (%) Annotations in Human and AI Generated Summaries"
                   " with Initial Prompt",
    ylabel       = "% Omitted",
    filename     = "fig3_pct_omitted.png",
    y_max        = 90,
    legend_below = True,
    hum_label    = "Human",
    ai_label     = "AI",
    show_ci      = True,
)

# ── Figure 4: % Fabricated (code 3) ───────────────────────────────────────────
fab_df = plot_side_by_side(
    code         = 3,
    title        = "Fabricated (%) Annotations in Human and AI Generated Summaries"
                   " with Initial Prompt",
    ylabel       = "% Fabricated",
    filename     = "fig4_pct_fabricated.png",
    y_max        = 12,
    legend_below = True,
    hum_label    = "Human",
    ai_label     = "AI",
    show_ci      = True,
)

# ── Print summary table ────────────────────────────────────────────────────────
summary = correct_df[["feature", "n"]].copy()
summary = summary.rename(columns={"n": "N (source=1)"})
summary["Human % correct"]    = correct_df["human_pct"].map(lambda x: f"{x:.1f}%")
summary["AI % correct"]       = correct_df["ai_pct"].map(lambda x: f"{x:.1f}%")
summary["Human % omitted"]    = omitted_df["human_pct"].map(lambda x: f"{x:.1f}%")
summary["AI % omitted"]       = omitted_df["ai_pct"].map(lambda x: f"{x:.1f}%")
summary["Human % fabricated"] = fab_df["human_pct"].map(lambda x: f"{x:.1f}%")
summary["AI % fabricated"]    = fab_df["ai_pct"].map(lambda x: f"{x:.1f}%")

summary.to_csv(OUT_DIR / "feature_metrics_summary.csv", index=False)
print("\n" + "=" * 70)
print("FEATURE-LEVEL METRICS SUMMARY")
print("=" * 70)
print(summary.to_string(index=False))
print(f"\nAll outputs saved to: {OUT_DIR}")


# ── Figure 5: Per-feature AI fabrication rate — V1 vs V2 ──────────────────────
# V2 data is a placeholder (NaN) until V2 validation data is available.
v1_fab_vals = fab_df["ai_pct"].values
v2_fab_vals = np.full(len(fab_df), np.nan)
feat_names5  = fab_df["feature"].tolist()

fig5, ax5 = plt.subplots(figsize=(FIG_W, FIG_H_SBS))
fig5.subplots_adjust(bottom=0.40, top=0.90)
x5 = np.arange(len(feat_names5))

bars_v1 = ax5.bar(x5 - BAR_WIDTH / 2, v1_fab_vals,
                  width=BAR_WIDTH, color=HUM_COLOR,
                  edgecolor="white", linewidth=0.7, zorder=3)

for bar in bars_v1:
    h = bar.get_height()
    if not np.isnan(h) and h > 0:
        ax5.text(bar.get_x() + bar.get_width() / 2, h + 0.05,
                 f"{h:.1f}%", ha="center", va="bottom",
                 fontsize=7.5, fontweight="bold", color="#333333")

ax5.set_xticks(x5)
ax5.set_xticklabels(feat_names5, rotation=35, ha="right", fontsize=TICK_FS)
ax5.set_ylabel("% Fabricated", fontsize=AXIS_FS)
ax5.set_title("AI Feature Fabrication Rate for Unstructured (V1) and Structured (V2) Prompts",
              fontsize=TITLE_FS, fontweight="bold", pad=12)
ax5.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
ax5.set_ylim(0, max(float(np.nanmax(v1_fab_vals)) * 2, 4))
ax5.grid(axis="y", color="#eeeeee", zorder=0)
ax5.set_axisbelow(True)

for i, row in fab_df.iterrows():
    ax5.text(i, -8, f"n={row['n']}", ha="center", fontsize=7,
             color="#777777", transform=ax5.get_xaxis_transform())

ax5.legend(
    handles=[Patch(facecolor=HUM_COLOR, label="V1 (Unstructured)"),
             Patch(facecolor=AI_COLOR,  label="V2 (Structured)")],
    fontsize=10, frameon=False, ncol=2,
    loc="upper center", bbox_to_anchor=(0.5, -0.36),
    borderaxespad=0,
)

fig5.savefig(OUT_DIR / "fig5_fab_v1_v2_per_feature.png", dpi=150, bbox_inches="tight")
plt.close(fig5)
print("Saved: fig5_fab_v1_v2_per_feature.png")


# ── Figure 6: Overall AI fabrication rate — V1 vs V2 ──────────────────────────
# Overall = unweighted mean of per-feature AI fabrication rates.
# V2 is a placeholder (NaN) until V2 validation data is available.
v1_overall_fab = float(fab_df["ai_pct"].mean())
v2_overall_fab = np.nan

fig6, ax6 = plt.subplots(figsize=(5, 5))
fig6.subplots_adjust(bottom=0.18, top=0.82)

x6      = [0, 1]
vals6   = [v1_overall_fab, v2_overall_fab]
colors6 = [HUM_COLOR, AI_COLOR]

bars6 = ax6.bar(x6, vals6, color=colors6,
                edgecolor="white", linewidth=0.7, width=0.5, zorder=3)

h = bars6[0].get_height()
ax6.text(bars6[0].get_x() + bars6[0].get_width() / 2, h + 0.03,
         f"{h:.2f}%", ha="center", va="bottom",
         fontsize=12, fontweight="bold", color="#333333")

ax6.set_xticks(x6)
ax6.set_xticklabels(["V1", "V2"], fontsize=AXIS_FS + 1)
ax6.set_ylabel("Fabrication rate (%)", fontsize=AXIS_FS)
ax6.set_title("AI Overall Fabrication Rate for\nUnstructured (V1) and Structured (V2) Prompts",
              fontsize=TITLE_FS, fontweight="bold", pad=12)
ax6.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
ax6.set_ylim(0, max(v1_overall_fab * 2.5, 3))
ax6.grid(axis="y", color="#eeeeee", zorder=0)
ax6.set_axisbelow(True)

fig6.savefig(OUT_DIR / "fig6_overall_fab_v1_v2.png", dpi=150, bbox_inches="tight")
plt.close(fig6)
print("Saved: fig6_overall_fab_v1_v2.png")
