"""
OCR Image Quality Scoring — Breast Bot Project
Extracted from notebooks/05_feature_extraction_ocr_bert.ipynb Part 1
Runs on all source PDFs: Laplacian variance, Tenengrad, RMS contrast,
intensity spread, mean brightness, skew angle.
"""

import hashlib
from pathlib import Path

import os

import fitz
import pytesseract
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ── Tesseract configuration ───────────────────────────────────────────────────
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Users\jamesr4\AppData\Local\miniforge3\Library\bin\tesseract.exe"
)
os.environ["TESSDATA_PREFIX"] = (
    r"C:\Users\jamesr4\AppData\Local\miniforge3\share\tessdata"
)

sns.set_style("whitegrid")

# ── Paths ─────────────────────────────────────────────────────────────────────
_ONEDRIVE = Path(r"C:\Users\jamesr4\OneDrive - Memorial Sloan Kettering Cancer Center")
_matches = [
    d for d in _ONEDRIVE.iterdir()
    if d.is_dir() and "Moo" in d.name and "Breast Bot" in d.name
]
if not _matches:
    raise FileNotFoundError("Breast Bot Project folder not found.")
SOURCE_ROOT = _matches[0]

FEATURE_DIR = Path(r"C:\Users\jamesr4\OneDrive - Memorial Sloan Kettering Cancer Center\Documents\GitHub\llm_summarization_br_ca\data\features")
REPORTS_DIR = Path(r"C:\Users\jamesr4\OneDrive - Memorial Sloan Kettering Cancer Center\Documents\GitHub\llm_summarization_br_ca\reports")
OUTPUT_CSV_PAGE = FEATURE_DIR / "page_level_ocr_quality.csv"
OUTPUT_CSV_CASE = FEATURE_DIR / "case_level_ocr_quality.csv"
OUTPUT_PLOT = REPORTS_DIR / "ocr_quality_distributions.png"

FEATURE_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300


# ── Quality metric functions ──────────────────────────────────────────────────
def render_page_to_gray(doc, page_index: int, dpi: int = 300):
    """Render a PDF page to grayscale numpy array."""
    page = doc.load_page(page_index)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, 3
    )
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return gray, pix.width, pix.height


def laplacian_variance(gray: np.ndarray) -> float:
    """Blur detection — higher = sharper."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def tenengrad(gray: np.ndarray) -> float:
    """Gradient energy / sharpness — higher = sharper."""
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.sqrt(gx ** 2 + gy ** 2).mean())


def rms_contrast(gray: np.ndarray) -> float:
    """RMS contrast — higher = better contrast."""
    return float(gray.astype(float).std())


def intensity_spread(gray: np.ndarray) -> float:
    """Intensity spread p95 - p5."""
    return float(np.percentile(gray, 95) - np.percentile(gray, 5))


def mean_brightness(gray: np.ndarray) -> float:
    """Mean pixel brightness."""
    return float(gray.mean())


def estimate_skew_angle(gray: np.ndarray) -> float:
    """Estimate skew angle via Hough line transform."""
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=200)
    if lines is None or len(lines) == 0:
        return 0.0
    angles = []
    for rho, theta in lines[:, 0]:
        angle = (theta * 180 / np.pi) - 90
        if -45 < angle < 45:
            angles.append(angle)
    return float(np.median(angles)) if angles else 0.0


def compute_page_quality(gray, width, height, dpi=300) -> dict:
    return {
        "laplacian_var": laplacian_variance(gray),
        "tenengrad": tenengrad(gray),
        "rms_contrast": rms_contrast(gray),
        "intensity_spread_p95_p5": intensity_spread(gray),
        "mean_brightness": mean_brightness(gray),
        "skew_angle_deg": estimate_skew_angle(gray),
        "dpi": dpi,
        "width_px": width,
        "height_px": height,
    }


def generate_case_id(surgeon: str, case_folder: str, filename: str) -> str:
    combined = f"{surgeon}/{case_folder}/{filename}"
    h = hashlib.sha256(combined.encode()).hexdigest()[:12]
    return f"CASE_{h.upper()}"


# ── Main scoring loop ─────────────────────────────────────────────────────────
def run_quality_scoring():
    print("=" * 70)
    print("OCR IMAGE QUALITY SCORING — BREAST BOT PROJECT")
    print("=" * 70)
    print(f"Source: {SOURCE_ROOT}")
    print(f"Features output: {FEATURE_DIR}")

    # Collect all PDFs preserving surgeon/case structure
    pdf_paths = []
    for surgeon_dir in sorted(SOURCE_ROOT.iterdir()):
        if not surgeon_dir.is_dir():
            continue
        for case_dir in sorted(surgeon_dir.iterdir()):
            if not case_dir.is_dir():
                continue
            for pdf in sorted(case_dir.glob("*.pdf")):
                pdf_paths.append({
                    "surgeon": surgeon_dir.name,
                    "case": case_dir.name,
                    "filename": pdf.name,
                    "path": pdf,
                })

    print(f"Found {len(pdf_paths)} PDFs across all cases\n")

    quality_rows = []

    for info in tqdm(pdf_paths, desc="Scoring image quality", unit="pdf"):
        surgeon = info["surgeon"]
        case = info["case"]
        filename = info["filename"]
        pdf_path = info["path"]
        case_id = generate_case_id(surgeon, case, filename)

        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            quality_rows.append({
                "case_id": case_id,
                "surgeon": surgeon,
                "case": case,
                "file": filename,
                "page": None,
                "status": "ERROR_OPEN",
                "error": str(e),
            })
            continue

        for p in range(doc.page_count):
            try:
                gray, w, h = render_page_to_gray(doc, p, DPI)
                metrics = compute_page_quality(gray, w, h, DPI)
                metrics.update({
                    "case_id": case_id,
                    "surgeon": surgeon,
                    "case": case,
                    "file": filename,
                    "page": p + 1,
                    "total_pages": doc.page_count,
                    "status": "OK",
                })
                quality_rows.append(metrics)
            except Exception as e:
                quality_rows.append({
                    "case_id": case_id,
                    "surgeon": surgeon,
                    "case": case,
                    "file": filename,
                    "page": p + 1,
                    "status": f"ERROR_PAGE",
                    "error": str(e),
                })

        doc.close()

    df = pd.DataFrame(quality_rows)
    df.to_csv(OUTPUT_CSV_PAGE, index=False)
    print(f"\n✅ Page-level quality saved: {OUTPUT_CSV_PAGE}")
    print(f"   Rows: {len(df)}")

    # ── Flag quality issues ───────────────────────────────────────────────────
    df_ok = df[df["status"] == "OK"].copy()
    if len(df_ok) > 0:
        blur_thresh = df_ok["laplacian_var"].quantile(0.10)
        contrast_thresh = df_ok["rms_contrast"].quantile(0.10)
        df_ok["flag_blurry"] = df_ok["laplacian_var"] < blur_thresh
        df_ok["flag_low_contrast"] = df_ok["rms_contrast"] < contrast_thresh

        print(f"\nBlur threshold (10th pctile laplacian_var): {blur_thresh:.2f}")
        print(f"Contrast threshold (10th pctile rms_contrast): {contrast_thresh:.2f}")
        print(f"Pages flagged blurry:       {df_ok['flag_blurry'].sum():,}")
        print(f"Pages flagged low-contrast: {df_ok['flag_low_contrast'].sum():,}")

        # ── Case-level summary ────────────────────────────────────────────────
        case_quality = (
            df_ok.groupby(["case_id", "surgeon", "case"])
            .agg(
                num_pages=("page", "count"),
                avg_laplacian_var=("laplacian_var", "mean"),
                worst_laplacian_var=("laplacian_var", "min"),
                avg_tenengrad=("tenengrad", "mean"),
                avg_rms_contrast=("rms_contrast", "mean"),
                avg_brightness=("mean_brightness", "mean"),
                pct_blurry=("flag_blurry", "mean"),
                pct_low_contrast=("flag_low_contrast", "mean"),
            )
            .reset_index()
        )
        case_quality.to_csv(OUTPUT_CSV_CASE, index=False)
        print(f"\n✅ Case-level quality saved: {OUTPUT_CSV_CASE}")
        print(f"   Cases: {len(case_quality)}")

        # Worst-quality cases
        print("\n── Top 10 worst-quality cases (lowest avg Laplacian) ──")
        worst = case_quality.nsmallest(10, "avg_laplacian_var")[
            ["surgeon", "case", "avg_laplacian_var", "avg_rms_contrast",
             "num_pages", "pct_blurry"]
        ]
        print(worst.to_string(index=False))

        # ── Quality plots ─────────────────────────────────────────────────────
        metrics_to_plot = [
            ("laplacian_var", "Laplacian Variance (blur)"),
            ("tenengrad", "Tenengrad (sharpness)"),
            ("rms_contrast", "RMS Contrast"),
            ("intensity_spread_p95_p5", "Intensity Spread (p95-p5)"),
            ("mean_brightness", "Mean Brightness"),
            ("skew_angle_deg", "Skew Angle (deg)"),
        ]

        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        for ax, (metric, title) in zip(axes.flatten(), metrics_to_plot):
            vals = df_ok[metric].dropna()
            if len(vals) > 0:
                ax.hist(vals, bins=40, color="#3498db", edgecolor="black", alpha=0.8)
            ax.set_title(title, fontweight="bold", fontsize=11)
            ax.set_xlabel(metric, fontsize=9)
            ax.set_ylabel("Page count")

        plt.suptitle(
            f"Per-Page OCR Image Quality Distributions\n"
            f"Breast Bot Project — {len(df_ok):,} pages from {len(pdf_paths)} PDFs",
            fontsize=13, fontweight="bold"
        )
        plt.tight_layout()
        plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n✅ Quality plot saved: {OUTPUT_PLOT}")

    else:
        print("No successfully processed pages — check errors above.")

    # ── Final summary ─────────────────────────────────────────────────────────
    error_rows = df[df["status"] != "OK"]
    print("\n" + "=" * 70)
    print("QUALITY SCORING COMPLETE")
    print("=" * 70)
    print(f"Total PDFs:         {len(pdf_paths):,}")
    print(f"Total pages scored: {len(df_ok):,}")
    print(f"Errors:             {len(error_rows):,}")
    print(f"\nOutputs:")
    print(f"  {OUTPUT_CSV_PAGE}")
    print(f"  {OUTPUT_CSV_CASE}")
    print(f"  {OUTPUT_PLOT}")


if __name__ == "__main__":
    run_quality_scoring()
