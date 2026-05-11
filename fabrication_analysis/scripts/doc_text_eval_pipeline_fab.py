"""
doc_text_eval_pipeline_fab.py

Doc and Text Eval Pipeline - 14 Confirmed AI Fabrication Cases

Stage 0: DOCX -> PDF conversion (handles any .docx source docs in case folders)
Stage 1: PDF validation - confirm all source docs are valid PDFs
Stage 2: Load case manifest from case_id_mapping.csv
Stage 3: Image quality scoring (Laplacian, Tenengrad, RMS contrast, brightness, skew)
Stage 4: Text extraction via Claude Vision sub-agents
           - All pages per document sent in ONE API call (using_sub_agents pattern)
           - ThreadPoolExecutor for concurrent document processing
           - Document-level caching to avoid re-calling on reruns
Stage 5: Assemble per-case text with [DOCUMENT: ...] headers
Stage 6: Knowledge graph construction (PatientNode, SourceDocumentNode, EvidenceChunkNode)
Stage 7: Feature extraction via Claude API with v2 system prompt (13 features)
Stage 8: Outputs - quality CSVs, per-case JSONs, combined JSON, GraphML KG, plots

Reference pattern: notebooks/using_sub_agents.ipynb (concurrent PDF->Vision sub-agents)

Usage:
    pip install PyMuPDF anthropic opencv-python networkx langchain-text-splitters python-dotenv tqdm
    cd llm_summarization_br_ca
    python fabrication_analysis/scripts/doc_text_eval_pipeline_fab.py
"""

from __future__ import annotations

import sys

# ── Dependency check ──────────────────────────────────────────────────────────
_MISSING = []
try:
    import fitz  # PyMuPDF
except ImportError:
    _MISSING.append("PyMuPDF")
try:
    import cv2
except ImportError:
    _MISSING.append("opencv-python")
try:
    import anthropic
except ImportError:
    _MISSING.append("anthropic")
try:
    import networkx  # noqa: F401
except ImportError:
    _MISSING.append("networkx")

if _MISSING:
    print(f"ERROR: Missing required packages: {', '.join(_MISSING)}")
    print("Run: pip install PyMuPDF anthropic opencv-python networkx "
          "langchain-text-splitters python-dotenv tqdm")
    sys.exit(1)

import base64
import json
import re
import shutil
import subprocess
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import io
import cv2
import fitz
import networkx as nx
import numpy as np
import pandas as pd
import anthropic
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from PIL import Image, ImageOps
from tqdm import tqdm

load_dotenv()

# ── Paths and config ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

FAB_SOURCE  = PROJECT_ROOT / "data" / "fab_source"
MAPPING_CSV = FAB_SOURCE / "case_id_mapping.csv"
PROMPT_FILE = PROJECT_ROOT / "prompts" / "updated_developer_prompt_v2.txt"
OUTPUT_DIR  = PROJECT_ROOT / "data" / "processed" / "doc_text_eval_fab_cases"
FEAT_DIR    = PROJECT_ROOT / "data" / "features"
KG_DIR      = PROJECT_ROOT / "data" / "knowledge_graph"
REPORTS_DIR = PROJECT_ROOT / "reports"

for _d in [OUTPUT_DIR, FEAT_DIR, KG_DIR, REPORTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

FAB_CASE_FOLDERS = {
    "AB_NP_DCIS", "AH_UD_INV", "AT_RW_INV", "DC_BF_invasive",
    "GM_DR_INV", "GM_XM_INV", "GP_MC_INV", "KP_MM_INV",
    "LK_SW_INV", "MET_KS", "MET_MM", "MET_MW", "ML_WY_INV", "TM_FS_DCIS",
}

VISION_MODEL     = "claude-sonnet-4-20250514"
EXTRACTION_MODEL = "claude-sonnet-4-20250514"
QUALITY_DPI      = 300
VISION_DPI       = 150          # 150 DPI → ~1195×1568 px letter page, ~21px x-height for 10pt text
MAX_PAGES_PER_CALL = 20         # ≤100 direct API; keep at 20 to manage per-request payload

# Sonnet 4 Vision API hard constraints
_MAX_LONG_EDGE   = 1568         # px — anything larger resized internally, wasting bandwidth
_MIN_EDGE        = 200          # px — below this Claude may hallucinate
_TARGET_BYTES    = int(3.75 * 1024 * 1024)   # decoded bytes ≤ 3.75 MB → safe after base64 (+33%)
_MAX_REQUEST_MB  = 30           # MB — hard limit is 32 MB; use 30 as safe ceiling

# System prompt for Vision calls — detailed enough to reach 1024-token cache minimum
VISION_SYSTEM_PROMPT = """You are an expert medical document transcriptionist specializing in oncology and breast imaging records. Your sole task is to extract all text from medical document page images with perfect fidelity.

EXTRACTION RULES:
1. Extract ALL visible text exactly as written — do not correct spelling, grammar, or abbreviations.
2. Preserve document structure: section headers, bullet points, numbered lists, and paragraph spacing.
3. Preserve all medical measurements exactly: sizes (cm, mm), dosages, percentages, ratios, BI-RADS scores.
4. Preserve all dates in their original format.
5. For tables: reproduce the tabular layout using spacing or pipe characters to preserve column alignment.
6. For stamps, watermarks, or overlaid text: include verbatim inside [STAMP: ...] notation.
7. For handwritten annotations: transcribe exactly and note [handwritten] if portions are illegible.
8. For redacted or obscured text: write [REDACTED] exactly as it appears, or [ILLEGIBLE] if obscured by image quality.
9. For multi-column layouts: transcribe left column fully before right column.
10. Include page headers, footers, and page numbers if visible.
11. Do not skip any text regardless of apparent clinical relevance — administrative text, facility names, provider signatures, and order numbers must all be transcribed.
12. Do not add interpretation, commentary, or summaries — output transcribed text only.
13. Begin transcription immediately with no preamble such as "Here is the text:" or "The document contains:".

DOCUMENT TYPES YOU WILL ENCOUNTER:
- Screening and diagnostic mammography reports with BI-RADS classifications (1-6)
- Breast ultrasound reports with lesion measurements and clock-face locations
- Breast MRI reports with enhancement patterns and background parenchymal enhancement
- Core needle biopsy pathology reports with histologic diagnoses
- Surgical pathology reports with receptor status (ER, PR, HER2, Ki-67)
- Outside institution radiology and pathology reports
- Internal institutional review and second-opinion reports
- Laboratory reports with reference ranges
- Radiology order forms and requisitions

SPECIAL HANDLING FOR MEDICAL CONTENT:
- Receptor status: preserve exact percentages and Allred scores (e.g., "ER 100% Allred 8/8")
- HER2 results: preserve IHC scores (0, 1+, 2+, 3+) and FISH ratios exactly
- Lesion sizes: preserve all dimensions (e.g., "1.2 x 0.8 x 0.9 cm") and measurement methods
- Clock-face locations: preserve exactly (e.g., "6:00 4 cm FN")
- BI-RADS: preserve category and any sub-classifications
- Clip/marker placement: preserve clip type and post-procedure confirmation status
- Lymph node status: preserve size, number, and morphologic descriptors

OUTPUT: Plain transcribed text only, preserving all line breaks and structure from the source document."""


# ── Inline quality metric functions (from ocr_quality_scoring.py) ─────────────
# Inlined to avoid Windows-specific module-level paths in that file.

def render_page_to_gray(doc, page_index: int, dpi: int = 300):
    page = doc.load_page(page_index)
    mat  = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pix  = page.get_pixmap(matrix=mat, alpha=False)
    img  = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return gray, pix.width, pix.height


def laplacian_variance(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def tenengrad(gray: np.ndarray) -> float:
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.sqrt(gx ** 2 + gy ** 2).mean())


def rms_contrast(gray: np.ndarray) -> float:
    return float(gray.astype(float).std())


def intensity_spread(gray: np.ndarray) -> float:
    return float(np.percentile(gray, 95) - np.percentile(gray, 5))


def mean_brightness(gray: np.ndarray) -> float:
    return float(gray.mean())


def estimate_skew_angle(gray: np.ndarray) -> float:
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
        "laplacian_var":           laplacian_variance(gray),
        "tenengrad":               tenengrad(gray),
        "rms_contrast":            rms_contrast(gray),
        "intensity_spread_p95_p5": intensity_spread(gray),
        "mean_brightness":         mean_brightness(gray),
        "skew_angle_deg":          estimate_skew_angle(gray),
        "dpi":                     dpi,
        "width_px":                width,
        "height_px":               height,
    }


# ── Helper functions ──────────────────────────────────────────────────────────

def classify_doc_type(filename: str) -> str:
    fn = filename.lower()
    # Progress notes / H&P / clinic notes — secondary documents, excluded from extraction
    if any(k in fn for k in ["h&p", "progress", "clinic", "consult", "office", "visit",
                               "note", "hpi", "soap", "encounter"]):
        return "progress_note"
    if any(k in fn for k in ["path", "pathol", "biopsy", "specimen", "surgical", "stc"]):
        return "pathology"
    if any(k in fn for k in ["mammo", "mmg", "mri", "us ", "us.", "imaging", "ultrasound",
                               "radiol", "dx ", "review", "scan"]):
        return "radiology"
    return "other"


EXCLUDED_DOC_TYPES = {"progress_note"}


def enhance_page_image(img_rgb: np.ndarray) -> np.ndarray:
    """
    Full preprocessing pipeline for scanned medical document pages:
    1. Denoise (fast non-local means)
    2. Deskew (correct rotation via Hough lines)
    3. Deblur / sharpen (unsharp mask)
    4. Contrast enhance (CLAHE)
    Returns enhanced RGB image.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # 1. Denoise
    gray = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7,
                                     searchWindowSize=21)

    # 2. Deskew if skew > 0.5 degrees
    angle = estimate_skew_angle(gray)
    if abs(angle) > 0.5:
        h, w = gray.shape
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        gray = cv2.warpAffine(gray, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)

    # 3. Unsharp mask (deblur / sharpen)
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=3)
    gray = cv2.addWeighted(gray, 1.6, blurred, -0.6, 0)

    # 4. CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def _encode_page_for_api(img_rgb: np.ndarray) -> tuple[bytes, str]:
    """
    Encode a rendered page image following Sonnet 4 hard constraints.

    - Long edge capped at 1568 px; short edge >= 200 px
    - Decoded bytes <= 3.75 MB (safe after base64 +33% overhead)
    - sRGB 8-bit, alpha flattened
    - Line art / documents  -> PNG (no ringing on text edges)
    - Complex scans         -> JPEG q=85 progressive, quality ladder to q=70 floor
    - If still over: shrink long edge 10% and retry at q=70

    Returns (raw_bytes, media_type). Caller uploads to Files API or base64-encodes.
    """
    img = Image.fromarray(img_rgb.astype(np.uint8), "RGB")
    w, h = img.size

    if min(w, h) < _MIN_EDGE:
        scale = _MIN_EDGE / min(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        w, h = img.size

    if max(w, h) > _MAX_LONG_EDGE:
        img.thumbnail((_MAX_LONG_EDGE, _MAX_LONG_EDGE), Image.LANCZOS)
        w, h = img.size

    # Format: few unique colors = line art -> PNG; else JPEG
    colors = img.getcolors(maxcolors=4096)
    use_png = colors is not None and len(colors) < 512

    if use_png:
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        data = buf.getvalue()
        if len(data) <= _TARGET_BYTES:
            return data, "image/png"
        # PNG too large - fall through to JPEG

    for quality in [85, 80, 75, 70]:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality,
                 progressive=True, subsampling=2, optimize=True)
        data = buf.getvalue()
        if len(data) <= _TARGET_BYTES:
            return data, "image/jpeg"

    # Final fallback: shrink long edge 10%
    img.thumbnail((int(max(w, h) * 0.9), int(max(w, h) * 0.9)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70, optimize=True)
    data = buf.getvalue()
    if len(data) > _TARGET_BYTES:
        raise ValueError(
            f"Cannot compress image to ≤3.75 MB: {len(data)/1024/1024:.1f} MB after all attempts"
        )
    return data, "image/jpeg"


def _upload_page_file(
    client: anthropic.Anthropic,
    page_bytes: bytes,
    media_type: str,
    filename: str,
) -> str:
    """Upload one page image to the Files API and return its file_id."""
    ext = "png" if media_type == "image/png" else "jpg"
    response = client.beta.files.upload(
        file=(f"{filename}.{ext}", page_bytes, media_type),
    )
    return response.id


# ── Stage 0: DOCX -> PDF conversion ──────────────────────────────────────────

def _find_converter() -> str | None:
    """Return path to LibreOffice or soffice, or None if not available."""
    for cmd in ["soffice", "libreoffice"]:
        path = shutil.which(cmd)
        if path:
            return path
    # Common Mac install path
    for candidate in [
        "/Applications/LibreOffice.app/Contents/MacOS/soffice",
        "/usr/local/bin/soffice",
    ]:
        if Path(candidate).exists():
            return candidate
    return None


def convert_docx_to_pdf(case_folders: list[str]) -> list[Path]:
    """
    Scan each fab case folder for .docx files and convert to PDF.
    Returns list of newly created PDF paths.
    Uses LibreOffice headless (most reliable cross-platform converter).
    """
    print("\n[Stage 0] DOCX -> PDF conversion")
    print("-" * 50)

    docx_files: list[Path] = []
    for folder in case_folders:
        folder_path = FAB_SOURCE / folder
        if folder_path.exists():
            docx_files.extend(folder_path.glob("*.docx"))
            docx_files.extend(folder_path.glob("*.DOCX"))

    if not docx_files:
        print("  No .docx files found in fab case folders. Skipping.")
        return []

    converter = _find_converter()
    if not converter:
        print(f"  WARNING: Found {len(docx_files)} .docx file(s) but LibreOffice is not installed.")
        print("  Install with: brew install --cask libreoffice")
        print("  Skipping DOCX conversion — .docx files will not be processed.")
        return []

    print(f"  Converter: {converter}")
    print(f"  Converting {len(docx_files)} .docx file(s)...")

    converted: list[Path] = []
    for docx_path in docx_files:
        out_dir  = docx_path.parent
        pdf_path = out_dir / (docx_path.stem + ".pdf")
        if pdf_path.exists():
            print(f"  Already converted: {pdf_path.name}")
            converted.append(pdf_path)
            continue
        try:
            result = subprocess.run(
                [converter, "--headless", "--convert-to", "pdf",
                 "--outdir", str(out_dir), str(docx_path)],
                capture_output=True, text=True, timeout=60,
            )
            if pdf_path.exists():
                print(f"  Converted: {docx_path.name} -> {pdf_path.name}")
                converted.append(pdf_path)
            else:
                print(f"  FAILED: {docx_path.name} — {result.stderr.strip()}")
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT: {docx_path.name}")
        except Exception as e:
            print(f"  ERROR: {docx_path.name} — {e}")

    print(f"  Converted {len(converted)} / {len(docx_files)} DOCX files")
    return converted


# ── Stage 1: PDF validation ───────────────────────────────────────────────────

def validate_pdfs(manifest_rows: pd.DataFrame) -> pd.DataFrame:
    """
    Confirm all source docs exist as valid PDFs.
    Returns manifest with validation columns added.
    """
    print("\n[Stage 1] PDF validation")
    print("-" * 50)

    results = []
    for _, row in manifest_rows.iterrows():
        pdf_path = FAB_SOURCE / row["case_folder"] / f"{row['case_id']}.pdf"
        rec = {
            "case_id":           row["case_id"],
            "case_folder":       row["case_folder"],
            "original_filename": row["original_filename"],
            "pdf_path":          str(pdf_path),
            "pdf_exists":        pdf_path.exists(),
            "pdf_valid":         False,
            "n_pages_detected":  0,
            "validation_error":  "",
        }
        if rec["pdf_exists"]:
            try:
                doc = fitz.open(pdf_path)
                rec["pdf_valid"]        = True
                rec["n_pages_detected"] = doc.page_count
                doc.close()
            except Exception as e:
                rec["validation_error"] = str(e)
        results.append(rec)

    val_df   = pd.DataFrame(results)
    n_total  = len(val_df)
    n_valid  = val_df["pdf_valid"].sum()
    n_miss   = (~val_df["pdf_exists"]).sum()
    n_broken = (val_df["pdf_exists"] & ~val_df["pdf_valid"]).sum()

    print(f"  Total mapped PDFs : {n_total}")
    print(f"  Valid PDFs        : {n_valid}")
    print(f"  Missing           : {n_miss}")
    print(f"  Broken/unreadable : {n_broken}")

    if n_miss or n_broken:
        for _, r in val_df[~val_df["pdf_valid"]].iterrows():
            status = "MISSING" if not r["pdf_exists"] else "BROKEN"
            msg    = f" — {r['validation_error']}" if r["validation_error"] else ""
            print(f"  [{status}] {r['case_folder']}/{r['case_id']}.pdf{msg}")

    return val_df


# ── Stage 2: Load case manifest ───────────────────────────────────────────────

def load_manifest(val_df: pd.DataFrame) -> dict:
    manifest = defaultdict(list)
    for _, row in val_df[val_df["pdf_valid"]].iterrows():
        manifest[row["case_folder"]].append({
            "case_id":           row["case_id"],
            "original_filename": row["original_filename"],
            "pdf_path":          Path(row["pdf_path"]),
        })
    print(f"\n[Stage 2] Manifest: {len(manifest)} case folders, "
          f"{sum(len(v) for v in manifest.values())} valid PDFs")
    return dict(manifest)


# ── Stage 3: Image quality scoring ───────────────────────────────────────────

def run_quality_stage(manifest: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    page_csv = FEAT_DIR / "fab_page_level_doc_quality.csv"
    case_csv = FEAT_DIR / "fab_case_level_doc_quality.csv"
    if page_csv.exists() and case_csv.exists():
        print("\n[Stage 3] Image quality scoring — loading from cache...")
        page_df = pd.read_csv(page_csv)
        case_df = pd.read_csv(case_csv)
        print(f"  Page-level CSV: {page_csv}  ({len(page_df)} rows, cached)")
        print(f"  Case-level CSV: {case_csv}  ({len(case_df)} rows, cached)")
        return page_df, case_df

    print("\n[Stage 3] Image quality scoring...")
    quality_rows = []

    for case_folder, docs in tqdm(manifest.items(), desc="  Quality"):
        for doc_info in docs:
            case_id           = doc_info["case_id"]
            original_filename = doc_info["original_filename"]
            pdf_path          = doc_info["pdf_path"]

            try:
                doc = fitz.open(pdf_path)
            except Exception as e:
                print(f"  ERROR opening {pdf_path.name}: {e}")
                continue

            native_chars = sum(
                len(doc.load_page(p).get_text().strip())
                for p in range(doc.page_count)
            )
            pdf_type = "native" if native_chars / max(doc.page_count, 1) > 50 else "scanned"

            for p in range(doc.page_count):
                try:
                    gray, w, h = render_page_to_gray(doc, p, QUALITY_DPI)
                    metrics    = compute_page_quality(gray, w, h, QUALITY_DPI)
                    metrics.update({
                        "case_id":           case_id,
                        "case_folder":       case_folder,
                        "original_filename": original_filename,
                        "page":              p + 1,
                        "total_pages":       doc.page_count,
                        "pdf_type":          pdf_type,
                        "doc_type":          classify_doc_type(original_filename),
                        "status":            "OK",
                    })
                except Exception as e:
                    metrics = {
                        "case_id": case_id, "case_folder": case_folder,
                        "original_filename": original_filename,
                        "page": p + 1, "total_pages": doc.page_count,
                        "pdf_type": pdf_type,
                        "doc_type": classify_doc_type(original_filename),
                        "status": "ERROR", "error": str(e),
                    }
                quality_rows.append(metrics)
            doc.close()

    page_df = pd.DataFrame(quality_rows)
    ok_df   = page_df[page_df["status"] == "OK"].copy()

    if len(ok_df) > 0:
        blur_thresh     = ok_df["laplacian_var"].quantile(0.10)
        contrast_thresh = ok_df["rms_contrast"].quantile(0.10)
        ok_mask = page_df["status"] == "OK"
        page_df.loc[ok_mask, "is_blurry"]       = page_df.loc[ok_mask, "laplacian_var"] < blur_thresh
        page_df.loc[ok_mask, "is_low_contrast"]  = page_df.loc[ok_mask, "rms_contrast"] < contrast_thresh
        print(f"  Blur threshold (p10 laplacian_var)  : {blur_thresh:.2f}")
        print(f"  Contrast threshold (p10 rms_contrast): {contrast_thresh:.2f}")

    case_df = (
        ok_df.groupby("case_folder")
        .agg(
            num_pdfs=("case_id", "nunique"),
            num_pages=("page", "count"),
            avg_laplacian_var=("laplacian_var", "mean"),
            worst_laplacian_var=("laplacian_var", "min"),
            avg_tenengrad=("tenengrad", "mean"),
            avg_rms_contrast=("rms_contrast", "mean"),
            avg_brightness=("mean_brightness", "mean"),
            pct_native=("pdf_type", lambda x: (x == "native").mean()),
        )
        .reset_index()
    )

    page_df.to_csv(FEAT_DIR / "fab_page_level_doc_quality.csv", index=False)
    case_df.to_csv(FEAT_DIR / "fab_case_level_doc_quality.csv", index=False)
    print(f"  Page-level CSV: {FEAT_DIR / 'fab_page_level_doc_quality.csv'}  ({len(page_df)} rows)")
    print(f"  Case-level CSV: {FEAT_DIR / 'fab_case_level_doc_quality.csv'}  ({len(case_df)} rows)")
    return page_df, case_df


# ── Stage 4: Claude Vision text extraction — dual mode (raw vs preprocessed) ──

def _call_vision_api(
    client: anthropic.Anthropic,
    file_ids: list[str],  # Files API file_ids (already uploaded)
) -> str:
    """
    Send all page images via streaming using Files API references.
    - Images uploaded once via Files API; referenced by file_id (no base64 overhead)
    - System prompt cached (>=1024 tokens) to reduce time-to-first-token
    - Streaming for faster perceived latency
    - Chunks at MAX_PAGES_PER_CALL
    """
    page_groups = [
        file_ids[i: i + MAX_PAGES_PER_CALL]
        for i in range(0, len(file_ids), MAX_PAGES_PER_CALL)
    ]
    text_parts = []
    for group in page_groups:
        content = [
            {"type": "image", "source": {"type": "file", "file_id": fid}}
            for fid in group
        ]
        content.append({
            "type": "text",
            "text": "Transcribe all text from the medical document page(s) above.",
        })
        with client.messages.stream(
            model=VISION_MODEL,
            max_tokens=4096,
            system=[{
                "type": "text",
                "text": VISION_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": content}],
            extra_headers={"anthropic-beta": "files-api-2025-04-14"},
        ) as stream:
            text_parts.append(stream.get_final_text())
    return "\n".join(text_parts)


def _extract_doc_one_mode(
    client: anthropic.Anthropic,
    case_id: str,
    original_filename: str,
    pdf_path: Path,
    case_dir: Path,
    preprocess: bool,
) -> dict:
    """
    Extract text for one mode (raw or preprocessed).
    Cache suffix: _full_text_raw.txt or _full_text_preprocessed.txt.
    """
    mode  = "preprocessed" if preprocess else "raw"
    cache = case_dir / f"{case_id}_full_text_{mode}.txt"

    if cache.exists():
        doc_text   = cache.read_text(encoding="utf-8")
        word_count = len(doc_text.split())
        return {
            "case_id": case_id, "original_filename": original_filename,
            "doc_type": classify_doc_type(original_filename),
            "mode": mode, "doc_text": doc_text,
            "n_pages": 0, "word_count": word_count,
            "char_count": len(doc_text),
            "words_per_page": 0, "chars_per_page": 0, "cached": True,
        }

    file_ids: list[str] = []
    try:
        doc     = fitz.open(pdf_path)
        n_pages = doc.page_count

        for p in range(n_pages):
            mat = fitz.Matrix(VISION_DPI / 72.0, VISION_DPI / 72.0)
            pix = doc.load_page(p).get_pixmap(matrix=mat, alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, 3)
            if preprocess:
                img = enhance_page_image(img)
            page_bytes, media_type = _encode_page_for_api(img)
            fid = _upload_page_file(
                client, page_bytes, media_type,
                f"{case_id}_p{p:03d}_{mode}",
            )
            file_ids.append(fid)
        doc.close()

        doc_text = _call_vision_api(client, file_ids)
        cache.write_text(doc_text, encoding="utf-8")

        word_count = len(doc_text.split())
        return {
            "case_id": case_id, "original_filename": original_filename,
            "doc_type": classify_doc_type(original_filename),
            "mode": mode, "doc_text": doc_text,
            "n_pages": n_pages, "word_count": word_count,
            "char_count": len(doc_text),
            "words_per_page": word_count / max(n_pages, 1),
            "chars_per_page": len(doc_text) / max(n_pages, 1),
            "cached": False,
        }

    except Exception as e:
        print(f"  ERROR vision [{mode}] {pdf_path.name}: {e}")
        return {
            "case_id": case_id, "original_filename": original_filename,
            "doc_type": classify_doc_type(original_filename),
            "mode": mode, "doc_text": "", "n_pages": 0,
            "word_count": 0, "char_count": 0,
            "words_per_page": 0, "chars_per_page": 0, "cached": False,
        }

    finally:
        # Always delete uploaded page files to avoid storage accumulation
        for fid in file_ids:
            try:
                client.beta.files.delete(fid)
            except Exception:
                pass


def _save_extraction_comparison(case_folder: str, raw: dict, pre: dict) -> None:
    """Save side-by-side comparison metrics for one document."""
    out = OUTPUT_DIR / case_folder / "extraction_comparison.json"
    existing = []
    if out.exists():
        try:
            with open(out) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, ValueError):
            existing = []

    entry = {
        "case_id":            raw["case_id"],
        "original_filename":  raw["original_filename"],
        "raw_word_count":     raw["word_count"],
        "pre_word_count":     pre["word_count"],
        "word_count_diff":    pre["word_count"] - raw["word_count"],
        "raw_char_count":     raw["char_count"],
        "pre_char_count":     pre["char_count"],
        "raw_snippet":        raw["doc_text"][:300].replace("\n", " "),
        "pre_snippet":        pre["doc_text"][:300].replace("\n", " "),
    }
    existing.append(entry)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)


def extract_text_stage(manifest: dict, client: anthropic.Anthropic) -> dict:
    """
    Concurrent Vision extraction — both raw and preprocessed modes per document.
    - Raw:          original image, just size-constrained JPEG encoding
    - Preprocessed: denoise -> deskew -> unsharp mask -> CLAHE -> size-constrained JPEG
    Cache files: {case_id}_full_text_raw.txt and {case_id}_full_text_preprocessed.txt
    Comparison metrics saved to {case_folder}/extraction_comparison.json
    Downstream (KG + feature extraction) uses the preprocessed version.
    """
    print("\n[Stage 4] Claude Vision extraction — raw vs preprocessed (concurrent)...")

    tasks: list[tuple] = []
    for case_folder, docs in manifest.items():
        case_dir = OUTPUT_DIR / case_folder
        case_dir.mkdir(parents=True, exist_ok=True)
        for doc_info in docs:
            tasks.append((case_folder, doc_info, case_dir))

    # Each task runs both modes; we track (case_folder, raw_result, pre_result)
    all_doc_texts: dict[str, list[dict]] = defaultdict(list)
    n_cached_raw = n_new_raw = n_cached_pre = n_new_pre = 0

    def _process(args):
        case_folder, doc_info, case_dir = args
        raw = _extract_doc_one_mode(client, doc_info["case_id"],
                                     doc_info["original_filename"],
                                     doc_info["pdf_path"], case_dir,
                                     preprocess=False)
        pre = _extract_doc_one_mode(client, doc_info["case_id"],
                                     doc_info["original_filename"],
                                     doc_info["pdf_path"], case_dir,
                                     preprocess=True)
        _save_extraction_comparison(case_folder, raw, pre)
        return case_folder, raw, pre

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(_process, t): t for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(tasks), desc="  Vision"):
            case_folder, raw, pre = fut.result()
            # Downstream uses preprocessed; attach both for comparison
            pre["raw_doc_text"] = raw["doc_text"]
            all_doc_texts[case_folder].append(pre)
            n_cached_raw += raw["cached"]; n_new_raw += not raw["cached"]
            n_cached_pre += pre["cached"]; n_new_pre += not pre["cached"]

    print(f"  Raw        : {n_new_raw} new, {n_cached_raw} cached")
    print(f"  Preprocessed: {n_new_pre} new, {n_cached_pre} cached")
    return dict(all_doc_texts)


# ── Stage 5: Assemble per-case text ──────────────────────────────────────────

def assemble_case_texts(doc_texts: dict) -> dict:
    """
    Returns {case_folder: {"raw": raw_text, "preprocessed": pre_text}}.
    Progress notes (H&P, clinic notes) are excluded - they are secondary
    documents that summarize primary sources and must not be used for extraction.
    doc["doc_text"]     = preprocessed-mode extracted text
    doc["raw_doc_text"] = raw-mode extracted text
    """
    case_texts = {}
    total_excluded = 0
    for case_folder, docs in doc_texts.items():
        sorted_docs = sorted(docs, key=lambda d: d["original_filename"])
        raw_parts = []
        pre_parts = []
        for d in sorted_docs:
            doc_type = classify_doc_type(d["original_filename"])
            if doc_type in EXCLUDED_DOC_TYPES:
                print(f"  [EXCLUDED progress_note] {case_folder} / {d['original_filename']}")
                total_excluded += 1
                continue
            header = f"\n\n[DOCUMENT: {d['original_filename']}]\n"
            raw_parts.append(header + d.get("raw_doc_text", d["doc_text"]))
            pre_parts.append(header + d["doc_text"])
        case_texts[case_folder] = {
            "raw":          "\n".join(raw_parts),
            "preprocessed": "\n".join(pre_parts),
        }
    print(f"\n[Stage 5] Assembled {len(case_texts)} case texts "
          f"(raw + preprocessed, {total_excluded} progress notes excluded)")
    return case_texts


# ── Stage 6: Knowledge graph construction ────────────────────────────────────

def _chunk_text(text: str, case_id: str, chunk_size: int = 1000, overlap: int = 200) -> list[dict]:
    """Inline text chunker - avoids langchain_core dependency conflict."""
    words = text.split()
    chunks = []
    step = max(chunk_size - overlap, 1)
    for i, start in enumerate(range(0, max(len(words), 1), step)):
        chunk_words = words[start: start + chunk_size]
        if not chunk_words:
            break
        chunk_text = " ".join(chunk_words)
        total = max(len(words) // chunk_size, 1)
        chunks.append({
            "chunk_id":    f"{case_id}_{i:04d}_{str(uuid.uuid4())[:6]}",
            "case_id":     case_id,
            "document_id": case_id,
            "page_num":    max(1, round((i / total) * 10)),
            "text":        chunk_text,
        })
    return chunks

def _flatten_v2_features(features: dict) -> dict[str, str]:
    flat = {}
    for k, v in features.items():
        if k == "lesions":
            continue
        flat[k] = str(v)
    for lesion in features.get("lesions", []):
        lid = lesion.get("lesion_id", "L?")
        for k, v in lesion.items():
            if k == "lesion_id":
                continue
            flat[f"{lid}_{k}"] = str(v.get("value", v) if isinstance(v, dict) else v)
    return flat


def build_kg_for_case(
    case_folder: str,
    docs: list[dict],
    case_ocr_text: str,
    features: dict | None,
    run_id: str,
) -> "KnowledgeGraph":
    from src.graph.graph_schema import (
        KnowledgeGraph, PatientNode, SourceDocumentNode,
        EvidenceChunkNode, ClinicalFeatureNode, ExtractionClaimNode,
    )
    kg = KnowledgeGraph()
    kg.patients.append(PatientNode(patient_id=case_folder, case_id=case_folder))

    for d in docs:
        kg.documents.append(SourceDocumentNode(
            document_id=d["case_id"],
            case_id=case_folder,
            modality=d["doc_type"],
        ))

    for chunk in _chunk_text(case_ocr_text, case_id=case_folder):
        kg.evidence_chunks.append(EvidenceChunkNode(
            chunk_id=chunk["chunk_id"],
            case_id=case_folder,
            document_id=case_folder,
            page_num=chunk["page_num"],
            text=chunk["text"],
        ))

    if features and "error" not in features:
        for feat_name, value_str in _flatten_v2_features(features).items():
            feat_id = f"{case_folder}_{feat_name}"
            kg.features.append(ClinicalFeatureNode(
                feature_id=feat_id,
                case_id=case_folder,
                feature_name=feat_name,
                value=value_str,
                confidence=1.0,
            ))
            kg.claims.append(ExtractionClaimNode(
                claim_id=str(uuid.uuid4())[:8],
                feature_id=feat_id,
                value=value_str,
                model_id=EXTRACTION_MODEL,
                prompt_id="v2",
                run_id=run_id,
            ))

    return kg


# ── Stage 7: Feature extraction via v2 prompt (Batch API) ────────────────────

def load_v2_prompt() -> str:
    return PROMPT_FILE.read_text(encoding="utf-8")


def _load_feature_cache(case_folder: str, mode: str) -> dict | None:
    cache = OUTPUT_DIR / case_folder / f"feature_extraction_{mode}.json"
    if cache.exists():
        with open(cache) as f:
            return json.load(f)
    return None


def _save_feature_cache(case_folder: str, mode: str, features: dict) -> None:
    cache = OUTPUT_DIR / case_folder / f"feature_extraction_{mode}.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "w") as f:
        json.dump(features, f, indent=2, ensure_ascii=False)


def _parse_features(raw_text: str) -> dict:
    m = re.search(r'\{.*\}', raw_text, re.DOTALL)
    if not m:
        return {"raw_response": raw_text}
    try:
        return json.loads(m.group())
    except (json.JSONDecodeError, ValueError):
        return {"raw_response": raw_text, "parse_error": "invalid_json"}


def _count_tokens(
    client: anthropic.Anthropic,
    v2_prompt: str,
    case_ocr_text: str,
) -> int:
    try:
        resp = client.messages.count_tokens(
            model=EXTRACTION_MODEL,
            system=[{"type": "text", "text": v2_prompt,
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": case_ocr_text}],
        )
        return resp.input_tokens
    except Exception:
        return -1


def run_feature_extraction_batch(
    case_texts: dict,
    client: anthropic.Anthropic,
    v2_prompt: str,
) -> dict[str, dict]:
    """
    Submit all feature extraction requests as a single Batch API job.
    - 50% cost reduction vs individual calls
    - Prompt caching on v2 system prompt across all requests
    - Token counts logged before submission
    Returns {case_folder: {"raw": features, "preprocessed": features}}
    """
    import time

    print("\n[Stage 7] Feature extraction — Batch API (raw + preprocessed)...")

    # Identify which requests still need extraction (skip cached)
    pending: list[tuple[str, str, str]] = []  # (case_folder, mode, text)
    results: dict[str, dict] = {}

    for case_folder, texts in case_texts.items():
        results[case_folder] = {}
        for mode in ("raw", "preprocessed"):
            cached = _load_feature_cache(case_folder, mode)
            if cached is not None:
                results[case_folder][mode] = cached
                print(f"  [cached] {case_folder} / {mode}")
            else:
                pending.append((case_folder, mode, texts[mode]))

    if not pending:
        print("  All feature extractions loaded from cache.")
        return results

    # Token count each request before submitting
    print(f"\n  Counting tokens for {len(pending)} requests...")
    total_tokens = 0
    for case_folder, mode, text in pending:
        n = _count_tokens(client, v2_prompt, text)
        total_tokens += max(n, 0)
        print(f"    {case_folder:<22} [{mode:<12}] {n:>6} tokens")
    print(f"  Total input tokens: {total_tokens:,}")

    # Build batch request list
    system_block = [{
        "type": "text",
        "text": v2_prompt,
        "cache_control": {"type": "ephemeral"},
    }]
    batch_requests = [
        {
            "custom_id": f"{case_folder}__{mode}",
            "params": {
                "model":      EXTRACTION_MODEL,
                "max_tokens": 4096,
                "system":     system_block,
                "messages":   [{"role": "user", "content": text}],
            },
        }
        for case_folder, mode, text in pending
    ]

    # Submit batch
    print(f"\n  Submitting batch of {len(batch_requests)} requests...")
    batch = client.messages.batches.create(requests=batch_requests)
    print(f"  Batch ID: {batch.id}  |  Status: {batch.processing_status}")

    # Poll until complete
    poll_interval = 30
    while batch.processing_status != "ended":
        time.sleep(poll_interval)
        batch = client.messages.batches.retrieve(batch.id)
        counts = batch.request_counts
        print(f"  [{batch.processing_status}] "
              f"processing={counts.processing}  "
              f"succeeded={counts.succeeded}  "
              f"errored={counts.errored}")

    print(f"  Batch complete — "
          f"succeeded={batch.request_counts.succeeded}  "
          f"errored={batch.request_counts.errored}")

    # Retrieve and save results
    for result in client.messages.batches.results(batch.id):
        case_folder, mode = result.custom_id.split("__", 1)
        if result.result.type == "succeeded":
            raw_text = result.result.message.content[0].text
            features = _parse_features(raw_text)
        else:
            print(f"  ERROR [{case_folder} / {mode}]: {result.result}")
            features = {"error": str(result.result)}
        _save_feature_cache(case_folder, mode, features)
        results[case_folder][mode] = features
        print(f"  [saved] {case_folder} / {mode}")

    return results


# ── Stage 8: Outputs ──────────────────────────────────────────────────────────

def save_text_extraction_json(case_folder: str, docs: list[dict]) -> None:
    out_path = OUTPUT_DIR / case_folder / "text_extraction.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = [{k: v for k, v in d.items() if k != "doc_text"} for d in docs]
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)


def generate_quality_plots(page_df: pd.DataFrame, doc_texts: dict) -> None:
    ok_df = page_df[page_df.get("status", pd.Series("OK", index=page_df.index)) == "OK"] \
        if "status" in page_df else page_df

    doc_rows = [
        {
            "doc_type":       d["doc_type"],
            "word_count":     d["word_count"],
            "chars_per_page": d["chars_per_page"],
            "words_per_page": d["words_per_page"],
        }
        for docs in doc_texts.values()
        for d in docs
    ]
    doc_df = pd.DataFrame(doc_rows)

    sns.set_theme(style="whitegrid", font_scale=0.95)
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))

    for ax, metric, title in [
        (axes[0, 0], "word_count",     "Word Count by Doc Type"),
        (axes[0, 1], "chars_per_page", "Chars per Page by Doc Type"),
        (axes[0, 2], "words_per_page", "Words per Page by Doc Type"),
    ]:
        if not doc_df.empty and metric in doc_df:
            sns.boxplot(data=doc_df, x="doc_type", y=metric, ax=ax,
                        palette="Set2", width=0.5)
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.set_xlabel("Document Type")
        ax.tick_params(axis="x", rotation=15)

    quality_metrics = [
        ("laplacian_var",            "Laplacian Variance (blur)"),
        ("tenengrad",                "Tenengrad (sharpness)"),
        ("rms_contrast",             "RMS Contrast"),
        ("intensity_spread_p95_p5",  "Intensity Spread p95-p5"),
        ("mean_brightness",          "Mean Brightness"),
        ("skew_angle_deg",           "Skew Angle (deg)"),
    ]
    for ax, (metric, title) in zip(
        [axes[1,0], axes[1,1], axes[1,2], axes[2,0], axes[2,1], axes[2,2]],
        quality_metrics,
    ):
        if metric in ok_df.columns:
            vals = ok_df[metric].dropna()
            if len(vals) > 0:
                ax.hist(vals, bins=30, color="#3498db", edgecolor="white", alpha=0.85)
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.set_ylabel("Page count")

    plt.suptitle(
        f"Document & Text Quality - 14 Confirmed AI Fabrication Cases\n"
        f"{len(ok_df):,} pages across 84 PDFs",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    out = REPORTS_DIR / "doc_text_eval_quality_plots.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Quality plots: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S") + "_" + str(uuid.uuid4())[:8]

    print("=" * 60)
    print("DOC & TEXT EVAL PIPELINE - 14 FAB CASES")
    print("=" * 60)
    print(f"Run ID  : {run_id}")
    print(f"Project : {PROJECT_ROOT}")
    print(f"Source  : {FAB_SOURCE}")

    client    = anthropic.Anthropic()
    v2_prompt = load_v2_prompt()

    # ── Stage 0: DOCX -> PDF conversion ──────────────────────────────────────
    convert_docx_to_pdf(sorted(FAB_CASE_FOLDERS))

    # ── Stage 1: PDF validation ───────────────────────────────────────────────
    mapping  = pd.read_csv(MAPPING_CSV)
    fab_rows = mapping[mapping["case_folder"].isin(FAB_CASE_FOLDERS)].copy()
    val_df   = validate_pdfs(fab_rows)
    val_df.to_csv(FEAT_DIR / "fab_pdf_validation.csv", index=False)
    print(f"  Validation CSV: {FEAT_DIR / 'fab_pdf_validation.csv'}")

    # ── Stage 2: Load manifest ────────────────────────────────────────────────
    manifest = load_manifest(val_df)
    if not manifest:
        print("ERROR: No valid PDFs found. Aborting.")
        sys.exit(1)

    # ── Stage 3: Image quality scoring ───────────────────────────────────────
    page_df, case_df = run_quality_stage(manifest)

    # ── Stage 4: Claude Vision text extraction ────────────────────────────────
    doc_texts = extract_text_stage(manifest, client)

    # ── Stage 5: Assemble per-case texts ─────────────────────────────────────
    case_texts = assemble_case_texts(doc_texts)

    # ── Stage 7: Feature extraction via Batch API ────────────────────────────
    all_features = run_feature_extraction_batch(case_texts, client, v2_prompt)

    # ── Stage 6+: KG construction + assemble results ─────────────────────────
    print("\n[Stage 6] Knowledge graph construction...")
    from src.graph.build_graph import build_networkx_graph, save_graphml

    global_G         = None
    all_case_results = []

    for case_folder in tqdm(sorted(manifest.keys()), desc="  Cases"):
        docs  = doc_texts.get(case_folder, [])
        texts = case_texts.get(case_folder, {"raw": "", "preprocessed": ""})
        case_ocr_text_pre = texts["preprocessed"]

        # Docs excluding progress notes (for text metrics)
        source_docs = [d for d in docs
                       if classify_doc_type(d["original_filename"]) not in EXCLUDED_DOC_TYPES]

        save_text_extraction_json(case_folder, docs)

        features_raw = all_features.get(case_folder, {}).get("raw", {})
        features_pre = all_features.get(case_folder, {}).get("preprocessed", {})

        # KG uses preprocessed text and features
        kg = build_kg_for_case(case_folder, source_docs, case_ocr_text_pre, features_pre, run_id)
        G  = build_networkx_graph(kg)
        global_G = G if global_G is None else nx.compose(global_G, G)

        text_metrics = {
            "n_docs":         len(source_docs),
            "total_pages":    sum(d["n_pages"] for d in source_docs),
            "total_words":    sum(d["word_count"] for d in source_docs),
            "total_chars":    sum(d["char_count"] for d in source_docs),
            "words_per_page": sum(d["words_per_page"] for d in source_docs) / max(len(source_docs), 1),
        }

        all_case_results.append({
            "case_folder":           case_folder,
            "run_id":                run_id,
            "text_metrics":          text_metrics,
            "features_raw":          features_raw,
            "features_preprocessed": features_pre,
            "kg_nodes":              G.number_of_nodes(),
            "kg_edges":              G.number_of_edges(),
        })
        print(
            f"  {case_folder:<22} src_docs={len(source_docs)}  "
            f"words={text_metrics['total_words']:,}  "
            f"kg_nodes={G.number_of_nodes()}"
        )

    # ── Stage 8: Save all outputs ─────────────────────────────────────────────
    print("\n[Stage 8] Saving outputs...")

    combined_out = OUTPUT_DIR / "all_cases_doc_text_eval.json"
    with open(combined_out, "w", encoding="utf-8") as f:
        json.dump(all_case_results, f, indent=2, default=str)
    print(f"  Combined JSON : {combined_out}")

    if global_G is not None:
        # GraphML does not support None values — replace with empty string
        for _, data in global_G.nodes(data=True):
            for k, v in list(data.items()):
                if v is None:
                    data[k] = ""
        for _, _, data in global_G.edges(data=True):
            for k, v in list(data.items()):
                if v is None:
                    data[k] = ""
        graphml_out = KG_DIR / f"fab_cases_kg_{run_id[:15]}.graphml"
        save_graphml(global_G, graphml_out)
        print(f"  GraphML KG    : {graphml_out}")

    generate_quality_plots(page_df, doc_texts)

    summary = {
        "run_id":           run_id,
        "timestamp":        datetime.utcnow().isoformat(),
        "n_cases":          len(manifest),
        "n_pdfs":           sum(len(v) for v in manifest.values()),
        "n_pages_total":    int(page_df[page_df["status"] == "OK"]["page"].count()),
        "model_vision":     VISION_MODEL,
        "model_extraction": EXTRACTION_MODEL,
        "outputs": {
            "pdf_validation_csv": str(FEAT_DIR / "fab_pdf_validation.csv"),
            "page_quality_csv":   str(FEAT_DIR / "fab_page_level_doc_quality.csv"),
            "case_quality_csv":   str(FEAT_DIR / "fab_case_level_doc_quality.csv"),
            "combined_json":      str(combined_out),
            "quality_plots":      str(REPORTS_DIR / "doc_text_eval_quality_plots.png"),
        },
    }
    with open(REPORTS_DIR / "doc_text_eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary JSON  : {REPORTS_DIR / 'doc_text_eval_summary.json'}")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Cases : {len(all_case_results)}")
    print(f"Run ID: {run_id}")


if __name__ == "__main__":
    main()
