"""
run_fab_cases_batch.py
Run the 14 confirmed AI fabrication cases through the full LangGraph pipeline.

Data flow per case:
  OCR cache files (concat) → chunk_text → FAISS → LangGraph (13 features)
  → experiments/runs/{run_id}/{case_id}.json
  → experiments/runs/{run_id}/feature_outputs.parquet
  → HCAT safety report  →  reports/*.png
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(
    r"C:\Users\jamesr4\OneDrive - Memorial Sloan Kettering Cancer Center"
    r"\Documents\GitHub\llm_summarization_br_ca"
)
DATA = Path(r"C:\Users\jamesr4\loc\data_private")

sys.path.insert(0, str(PROJECT_ROOT))

OCR_CACHE    = DATA / "ocr_cache"
MAPPING_CSV  = DATA / "breast_bot_deidentified" / "case_id_mapping.csv"
FAB_XLSX     = DATA / "raw" / "ai_fabrications_dataset.xlsx"
REPORTS_DIR  = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

PROMPT_ID  = "rag_verify_v1"
MODEL_ID   = "gpt-4o"

_SURGEON_MAP = {
    "el tamer": "el tamer", "el-tamer": "el tamer",
    "sacchini": "sacchini", "giannakou": "giannakou",
    "montagna": "montag", "montag": "montag", "lisa allen": "allen",
}


def _norm(s: str) -> str:
    return _SURGEON_MAP.get(str(s).strip().lower(), str(s).strip().lower())


def load_fab_cases() -> list[dict]:
    """Return list of {case_id, ocr_text, mrn, fab_features} for 14 fab cases."""
    mapping = pd.read_csv(MAPPING_CSV)
    fab     = pd.read_excel(FAB_XLSX)

    ai_cols = [c for c in fab.columns if c.endswith("_status_ai")]
    for c in ai_cols:
        fab[c] = pd.to_numeric(fab[c], errors="coerce")
    fab["surgeon_last"]  = fab["surgeon"].str.split(",").str[0].str.strip()
    fab["surgeon_norm"]  = fab["surgeon_last"].apply(_norm)
    fab["fab_features"]  = fab.apply(
        lambda r: [c.replace("_status_ai", "") for c in ai_cols if r[c] == 3],
        axis=1,
    )

    mapping["surgeon_norm"]    = mapping["surgeon"].apply(_norm)
    mapping["folder_initials"] = mapping["case_folder"].str.split("_").str[1].str.upper()

    rep = (
        mapping.dropna(subset=["original_path"])
        .drop_duplicates(subset=["surgeon_norm", "folder_initials"])
        [["surgeon_norm", "folder_initials", "case_folder"]]
    )
    df = fab.merge(
        rep,
        left_on=["surgeon_norm", "patient_initials"],
        right_on=["surgeon_norm", "folder_initials"],
        how="left",
    )

    cases = []
    for _, row in df.iterrows():
        cf      = str(row.get("case_folder", ""))
        doc_ids = mapping[mapping["case_folder"] == cf]["case_id"].tolist()
        texts   = []
        for did in doc_ids:
            cache_file = OCR_CACHE / f"{did}.txt"
            if cache_file.exists():
                doc_txt = cache_file.read_text(encoding="utf-8", errors="ignore")
                fn      = mapping.loc[mapping["case_id"] == did,
                                      "original_filename"].values
                header  = f"\n\n[DOCUMENT: {fn[0] if len(fn) else did}]\n"
                texts.append(header + doc_txt)

        if not texts:
            print(f"  WARNING: no OCR cache for {cf} — skipping")
            continue

        ocr_text = "\n".join(texts)
        cases.append({
            "case_id":      cf,
            "ocr_text":     ocr_text,
            "mrn":          int(row["mrn"]),
            "fab_features": row["fab_features"],
        })
        print(
            f"  Loaded {cf:<22} {len(texts)} docs "
            f"| {len(ocr_text):>7} chars "
            f"| fab={row['fab_features']}"
        )

    return cases


def run_pipeline(cases: list[dict]) -> list[dict]:
    from src.utils.io_utils import generate_run_id
    from src.workflows.orchestration import run_batch

    run_id = generate_run_id()
    print(f"\n{'='*60}")
    print(f"RUN ID   : {run_id}")
    print(f"Cases    : {len(cases)}")
    print(f"Prompt   : {PROMPT_ID}")
    print(f"Model    : {MODEL_ID}")
    print(f"Started  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    # Strip metadata fields before passing to run_batch
    batch_input = [
        {"case_id": c["case_id"], "ocr_text": c["ocr_text"]}
        for c in cases
    ]

    results = run_batch(
        cases=batch_input,
        prompt_id=PROMPT_ID,
        model_id=MODEL_ID,
        run_id=run_id,
        save_results=True,
    )
    return results, run_id


def compute_and_plot_hcat(results: list[dict], cases_meta: list[dict],
                          run_id: str) -> None:
    """Compute HCAT metrics and generate figures comparing pipeline vs ground truth."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    from eval.metrics.hcat_metrics import compute_batch_hcat, hcat_report_to_df
    from src.utils.io_utils import flatten_case_results_to_df

    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

    # Flatten to DataFrame
    df = flatten_case_results_to_df(results)
    if df.empty:
        print("WARNING: no feature rows to analyze")
        return

    report   = compute_batch_hcat(df, run_id=run_id,
                                  prompt_id=PROMPT_ID, model_id=MODEL_ID)
    hcat_df  = hcat_report_to_df(report)

    print("\n" + "="*60)
    print("HCAT BATCH REPORT")
    print("="*60)
    print(f"  Mean fabrication rate : {report.mean_fabrication_rate:.1%}")
    print(f"  Mean accuracy         : {report.mean_accuracy:.1%}")
    print(f"  Mean omission rate    : {report.mean_omission_rate:.1%}")
    print(f"  Mean safety score     : {report.mean_safety_score:.1%}")

    # ── Ground truth overlay ──────────────────────────────────────────────────
    gt_by_case = {c["case_id"]: c["fab_features"] for c in cases_meta}

    gt_rows = []
    for r in results:
        cid  = r.get("case_id", "")
        feats = r.get("features", {})
        gt_fab = gt_by_case.get(cid, [])
        for feat_name, feat_data in feats.items():
            short = feat_name.replace("feature_", "").split("_", 1)[-1]
            gt_rows.append({
                "case_id":      cid,
                "feature":      feat_name,
                "feature_short": short,
                "verdict":      feat_data.get("verdict"),
                "confidence":   feat_data.get("confidence"),
                "gt_fabrication": any(
                    short in f or f in short for f in gt_fab
                ),
            })

    gt_df = pd.DataFrame(gt_rows)

    # ── Figure 1: Verdict distribution per case ───────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = {
        "CORRECT": "#2ecc71", "FABRICATION": "#e74c3c",
        "OMISSION": "#f39c12", "UNCERTAIN": "#95a5a6",
    }

    # A: Verdict stacked bar per case
    ax = axes[0, 0]
    pivot = (
        gt_df.groupby(["case_id", "verdict"])
        .size()
        .unstack(fill_value=0)
    )
    pivot.plot(kind="bar", stacked=True, ax=ax,
               color=[colors.get(c, "#bdc3c7") for c in pivot.columns],
               edgecolor="white", linewidth=0.8)
    ax.set_title("Verdict Distribution per Case", fontweight="bold")
    ax.set_xlabel("Case")
    ax.set_ylabel("Feature Count")
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.legend(title="Verdict", fontsize=8)

    # B: Fabrication detection vs ground truth
    ax = axes[0, 1]
    gt_fab = gt_df[gt_df["gt_fabrication"]]
    pred_fab = (
        gt_fab.groupby("case_id")
        .apply(lambda g: (g["verdict"] == "FABRICATION").mean())
        .reset_index(name="detection_rate")
    )
    bar_colors = ["#e74c3c" if r >= 0.5 else "#f39c12"
                  for r in pred_fab["detection_rate"]]
    ax.bar(pred_fab["case_id"], pred_fab["detection_rate"] * 100,
           color=bar_colors, edgecolor="white")
    ax.axhline(50, color="black", linestyle="--", alpha=0.5,
               label="50% threshold")
    ax.set_title("GT Fabrication Detection Rate per Case", fontweight="bold")
    ax.set_ylabel("Detection Rate (%)")
    ax.tick_params(axis="x", rotation=50, labelsize=7)
    ax.legend(fontsize=8)

    # C: Confidence distribution by verdict
    ax = axes[1, 0]
    for verdict, group in gt_df.dropna(subset=["confidence", "verdict"]).groupby("verdict"):
        ax.hist(group["confidence"], bins=15, alpha=0.6,
                color=colors.get(verdict, "#bdc3c7"), label=verdict)
    ax.set_title("Confidence Distribution by Verdict", fontweight="bold")
    ax.set_xlabel("Extraction Confidence")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)

    # D: HCAT metrics summary
    ax = axes[1, 1]
    metrics = ["fabrication_rate", "accuracy", "omission_rate", "safety_score"]
    vals    = [
        report.mean_fabrication_rate, report.mean_accuracy,
        report.mean_omission_rate, report.mean_safety_score,
    ]
    bar_c = ["#e74c3c", "#2ecc71", "#f39c12", "#3498db"]
    bars  = ax.bar(metrics, [v * 100 for v in vals],
                   color=bar_c, edgecolor="white")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1%}", ha="center", fontsize=9, fontweight="bold")
    ax.set_title("HCAT Mean Metrics (14 Fab Cases)", fontweight="bold")
    ax.set_ylabel("%")
    ax.set_ylim(0, 110)
    ax.tick_params(axis="x", rotation=20)

    plt.suptitle(
        f"LangGraph Fabrication Analysis — 14 Confirmed Cases\n"
        f"run_id={run_id}  model={MODEL_ID}",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    out1 = REPORTS_DIR / f"fab_cases_hcat_{run_id[:8]}.png"
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: {out1}")

    # ── Figure 2: Feature-level fabrication heatmap ───────────────────────────
    if not gt_df.empty:
        hm = gt_df.pivot_table(
            index="case_id", columns="feature_short",
            values="verdict",
            aggfunc=lambda x: (x == "FABRICATION").mean(),
        ).fillna(0)

        fig2, ax2 = plt.subplots(figsize=(max(14, len(hm.columns) * 1.2), 6))
        import seaborn as sns
        sns.heatmap(
            hm, annot=True, fmt=".0%", cmap="RdYlGn_r",
            vmin=0, vmax=1, ax=ax2, linewidths=0.5,
            cbar_kws={"label": "Fabrication Rate"},
        )
        ax2.set_title(
            "Feature-Level Fabrication Rate — 14 GT Cases",
            fontsize=13, fontweight="bold",
        )
        ax2.set_xlabel("Clinical Feature")
        ax2.set_ylabel("Case")
        plt.tight_layout()
        out2 = REPORTS_DIR / f"fab_cases_feature_heatmap_{run_id[:8]}.png"
        plt.savefig(out2, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Figure saved: {out2}")

    # ── Save ground-truth overlay CSV ─────────────────────────────────────────
    out_dir = (PROJECT_ROOT / "experiments" / "runs" / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_df.to_csv(out_dir / "gt_overlay.csv", index=False)
    hcat_df.to_csv(out_dir / "hcat_report.csv", index=False)

    summary = {
        "run_id":   run_id,
        "timestamp": datetime.utcnow().isoformat(),
        "n_cases":  len(results),
        "prompt_id": PROMPT_ID,
        "model_id":  MODEL_ID,
        "mean_fabrication_rate": report.mean_fabrication_rate,
        "mean_accuracy":         report.mean_accuracy,
        "mean_omission_rate":    report.mean_omission_rate,
        "mean_safety_score":     report.mean_safety_score,
    }
    with open(out_dir / "batch_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll outputs in: {out_dir}")


if __name__ == "__main__":
    print("Loading 14 fabrication cases from OCR cache...")
    cases = load_fab_cases()
    print(f"\nLoaded {len(cases)} cases.\n")

    results, run_id = run_pipeline(cases)

    print(f"\nPipeline complete. Processed {len(results)} cases.")
    errors = [r for r in results if "error" in r]
    if errors:
        print(f"Errors ({len(errors)}):")
        for e in errors:
            print(f"  {e['case_id']}: {e['error']}")

    compute_and_plot_hcat(results, cases, run_id)
    print("\nDone.")
