"""
Build fab_page_level_doc_quality.csv
=====================================
Iterates every PDF under breast_bot_deidentified, renders each page at
RENDER_DPI, computes page-level image quality metrics, and saves one row
per page to data/features/fab_page_level_doc_quality.csv.

doc_type is inferred from the original_filename in case_id_mapping.csv:
  imaging_*  →  radiology
  path_*     →  pathology

laplacian_var_pct and rms_contrast_pct are the within-dataset percentile rank
(0-100) of each page's laplacian_var / rms_contrast across the full page set.
Lower percentile = poorer quality relative to the rest of the corpus.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import fitz   # PyMuPDF
from scipy.stats import percentileofscore

RENDER_DPI   = 150
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT  = Path(r"C:\Users\jamesr4\loc\data_private\breast_bot_deidentified")
MAPPING_CSV  = SOURCE_ROOT / "case_id_mapping.csv"
OUT_CSV      = PROJECT_ROOT / "data" / "features" / "fab_page_level_doc_quality.csv"
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)


# ── Doc-type classification from original filename ────────────────────────────
def _doc_type(original_filename: str) -> str:
    fn = str(original_filename).lower()
    if fn.startswith("path"):
        return "pathology"
    return "radiology"   # imaging_*, mri_*, us_*, mammo_*, etc.


# ── Page rendering ─────────────────────────────────────────────────────────────
def _render_gray(page) -> np.ndarray:
    mat = fitz.Matrix(RENDER_DPI / 72, RENDER_DPI / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)


def _estimate_skew(gray: np.ndarray) -> float:
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                            threshold=80, minLineLength=80, maxLineGap=10)
    if lines is None:
        return 0.0
    angles = []
    for ln in lines:
        x1, y1, x2, y2 = ln[0]
        if x2 != x1:
            a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(a) < 45:
                angles.append(a)
    return float(np.median(angles)) if angles else 0.0


def _page_metrics(page) -> dict:
    gray = _render_gray(page)
    lap  = cv2.Laplacian(gray, cv2.CV_64F)
    sx   = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy   = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return {
        "laplacian_var":           float(lap.var()),
        "tenengrad":               float(np.sum(sx ** 2 + sy ** 2)),
        "rms_contrast":            float(np.std(gray.astype(np.float32))),
        "intensity_spread_p95_p5": float(np.percentile(gray, 95) - np.percentile(gray, 5)),
        "mean_brightness":         float(np.mean(gray)),
        "skew_angle_deg":          _estimate_skew(gray),
    }


# ── Load mapping ───────────────────────────────────────────────────────────────
print("Loading case_id_mapping.csv …")
mapping = pd.read_csv(MAPPING_CSV)
# case_id (stem) → case_folder, doc_type
id_map = {}
for _, row in mapping.iterrows():
    stem = str(row["case_id"]).strip()
    id_map[stem] = {
        "case_folder": str(row["case_folder"]).strip(),
        "doc_type":    _doc_type(str(row.get("original_filename", "imaging_"))),
    }
print(f"  Mapping entries: {len(id_map)}")

# ── Walk all PDFs ──────────────────────────────────────────────────────────────
pdf_paths = sorted(SOURCE_ROOT.rglob("*.pdf"))
print(f"Found {len(pdf_paths)} PDFs under {SOURCE_ROOT}")

rows = []
errors = []

for i, pdf_path in enumerate(pdf_paths, 1):
    stem = pdf_path.stem   # e.g. CASE_0A6EE726BD10
    meta = id_map.get(stem)
    if meta is None:
        errors.append(f"No mapping for {stem}")
        continue

    case_folder = meta["case_folder"]
    doc_type    = meta["doc_type"]

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        errors.append(f"Cannot open {pdf_path.name}: {exc}")
        continue

    for pg_idx, page in enumerate(doc):
        try:
            m = _page_metrics(page)
        except Exception as exc:
            errors.append(f"Page error {pdf_path.name} p{pg_idx}: {exc}")
            continue
        rows.append({
            "case_folder": case_folder,
            "doc_type":    doc_type,
            "pdf_stem":    stem,
            "page_num":    pg_idx + 1,
            **m,
        })

    doc.close()

    if i % 100 == 0 or i == len(pdf_paths):
        print(f"  [{i}/{len(pdf_paths)}] pages so far: {len(rows)}")

print(f"\nTotal pages extracted: {len(rows)}")
if errors:
    print(f"Errors ({len(errors)}):")
    for e in errors[:20]:
        print(f"  {e}")

# ── Compute percentile ranks for blur and contrast ────────────────────────────
df = pd.DataFrame(rows)

lap_series = df["laplacian_var"].dropna().values
con_series = df["rms_contrast"].dropna().values

df["laplacian_var_pct"] = df["laplacian_var"].apply(
    lambda v: float(percentileofscore(lap_series, v, kind="rank")) if pd.notna(v) else np.nan
)
df["rms_contrast_pct"] = df["rms_contrast"].apply(
    lambda v: float(percentileofscore(con_series, v, kind="rank")) if pd.notna(v) else np.nan
)
print(f"\nPercentile ranks computed for {len(df)} pages")
print(f"  laplacian_var_pct : median={df['laplacian_var_pct'].median():.1f}  min={df['laplacian_var_pct'].min():.1f}  max={df['laplacian_var_pct'].max():.1f}")
print(f"  rms_contrast_pct  : median={df['rms_contrast_pct'].median():.1f}  min={df['rms_contrast_pct'].min():.1f}  max={df['rms_contrast_pct'].max():.1f}")

# ── Save ───────────────────────────────────────────────────────────────────────
df.to_csv(OUT_CSV, index=False)
print(f"\nSaved {len(df)} rows → {OUT_CSV}")
print(f"  Cases    : {df['case_folder'].nunique()}")
print(f"  Doc types: {df['doc_type'].value_counts().to_dict()}")
print(f"  Columns  : {list(df.columns)}")
