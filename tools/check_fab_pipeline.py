"""
Sanity check: validates the fabrication/omission pipeline mapping end-to-end.
Run with: python tools/check_fab_pipeline.py
"""
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_PRIVATE = Path(r"C:\Users\jamesr4\loc\data_private")
VALIDATION_XLS = DATA_PRIVATE / "raw" / "merged_llm_summary_validation_datasheet_identified.xlsx"
MAPPING_CSV    = DATA_PRIVATE / "breast_bot_deidentified" / "case_id_mapping.csv"
OCR_CACHE      = DATA_PRIVATE / "ocr_cache"
OCR_CACHE.mkdir(exist_ok=True)

# ── Load and identify errors ──────────────────────────────────────────────────
df_val  = pd.read_excel(VALIDATION_XLS)
mapping = pd.read_csv(MAPPING_CSV)

AI_COLS = [c for c in df_val.columns if c.endswith("_status_ai")]
for c in AI_COLS:
    df_val[c] = pd.to_numeric(df_val[c], errors="coerce")

bad_mask = ((df_val[AI_COLS] == 2) | (df_val[AI_COLS] == 3)).any(axis=1)
df_bad = df_val[bad_mask].copy()

print(f"Total cases in sheet : {len(df_val)}")
print(f"Cases with errors    : {len(df_bad)}")
print(f"  Fabrication (3)    : {(df_val[AI_COLS] == 3).any(axis=1).sum()}")
print(f"  Omission    (2)    : {(df_val[AI_COLS] == 2).any(axis=1).sum()}")

# ── Build mapping ─────────────────────────────────────────────────────────────
mapping["patient_initials_folder"] = (
    mapping["case_folder"].str.split("_").str[1].str.upper()
)

SURGEON_MAP = {
    "el tamer":  "el tamer", "el-tamer": "el tamer",
    "sacchini":  "sacchini",
    "giannakou": "giankou", "giankou": "giankou",
    "montagna":  "montag",  "montag":   "montag",
    "lisa allen": "allen",
}

def norm_surgeon(s: str) -> str:
    sl = str(s).strip().lower()
    return SURGEON_MAP.get(sl, sl)

df_bad = df_bad.copy()
df_bad["surgeon_last"] = df_bad["surgeon"].str.split(",").str[0].str.strip()
df_bad["surgeon_norm"] = df_bad["surgeon_last"].apply(norm_surgeon)
mapping["surgeon_norm"] = mapping["surgeon"].apply(norm_surgeon)

df_merged = df_bad.merge(
    mapping,
    left_on=["surgeon_norm", "patient_initials"],
    right_on=["surgeon_norm", "patient_initials_folder"],
    how="left",
)

# ── Check PDF existence ───────────────────────────────────────────────────────
df_merged["pdf_exists"] = df_merged["deidentified_path"].apply(
    lambda p: Path(p).exists() if pd.notna(p) else False
)

print(f"\nMapped PDF rows      : {len(df_merged)}")
print(f"PDFs on disk         : {df_merged['pdf_exists'].sum()} / {len(df_merged)}")
print(f"Missing PDFs         : {(~df_merged['pdf_exists']).sum()}")

# ── Per-patient summary ───────────────────────────────────────────────────────
pat_sum = (
    df_merged.groupby("mrn")
    .agg(total=("case_id", "count"), found=("pdf_exists", "sum"))
    .reset_index()
)
print(f"\nPatients             : {len(pat_sum)}")
all_found = (pat_sum["total"] == pat_sum["found"]).sum()
print(f"All PDFs found       : {all_found} / {len(pat_sum)} patients")
print(f"PDFs per patient     : min={int(pat_sum['found'].min())}  max={int(pat_sum['found'].max())}  "
      f"median={int(pat_sum['found'].median())}")

# ── OCR cache status ──────────────────────────────────────────────────────────
all_pdfs = [
    Path(p) for p in df_merged["deidentified_path"].dropna().unique()
    if Path(p).exists()
]
cached = sum(1 for p in all_pdfs if (OCR_CACHE / (p.stem + ".txt")).exists())
to_ocr  = len(all_pdfs) - cached

print(f"\nUnique PDFs          : {len(all_pdfs)}")
print(f"Already OCR-cached   : {cached}")
print(f"Still to OCR         : {to_ocr}")

# 4-thread parallel estimate: 5 pages/PDF × 4.6 s/page ÷ 4 workers
est_min = to_ocr * 5 * 4.6 / 60 / 4
print(f"Est. OCR time (4 th) : ~{est_min:.0f} min")

# ── Error breakdown by feature ────────────────────────────────────────────────
feat_errors = pd.DataFrame({
    "feature": [c.replace("_status_ai", "") for c in AI_COLS],
    "fabrications": [(df_val[c] == 3).sum() for c in AI_COLS],
    "omissions":    [(df_val[c] == 2).sum() for c in AI_COLS],
}).assign(total=lambda x: x["fabrications"] + x["omissions"])
feat_errors = feat_errors[feat_errors["total"] > 0].sort_values("total", ascending=False)

print("\nFeature-level AI errors:")
print(feat_errors.to_string(index=False))

print("\nSanity check complete.")


# =============================================================================
# MODULE SUMMARY
# =============================================================================
# File   : tools/check_fab_pipeline.py
# Purpose: Standalone sanity-check script that validates the entire
#          MRN-to-PDF mapping pipeline for the 63 AI fabrication/omission
#          cases before running the LangGraph pipeline.
#
# Run with: python tools/check_fab_pipeline.py
#
# What it checks:
#   1. Loads validation sheet + identifies all 63 error cases (2=omission,
#      3=fabrication in _status_ai columns).
#   2. Maps each patient (MRN x surgeon_last) to their case_folder(s) in
#      case_id_mapping.csv using surgeon normalisation + patient initials.
#   3. Verifies all deidentified PDF paths exist on disk.
#   4. Reports OCR cache status (how many PDFs already cached in ocr_cache/).
#   5. Prints per-feature error counts from the validation sheet.
#
# Outputs (all to stdout):
#   - Total error cases, fabrication/omission split
#   - Mapped PDF row count, PDFs on disk vs missing
#   - Per-patient min/max/median PDF count
#   - OCR cache status + estimated remaining OCR time
#   - Feature-level error breakdown table
#
# Consumed by:
#   Developer verification before running
#   fabrication_analysis/03_fabrication_omission_pipeline.ipynb
# =============================================================================
