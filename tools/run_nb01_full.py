"""
NB01 Full Runner — Deidentification of all PDFs + Validation Excel
Runs all NB01 logic with detailed timing logs.
Output: data_private/deidentified/ + reports/nb01_timing_log.csv
"""
import os
import re
import time
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple
from datetime import datetime

import fitz  # PyMuPDF
import pytesseract
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault(
    "TESSDATA_PREFIX",
    r"C:\Users\jamesr4\AppData\Local\miniforge3\share\tessdata",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(os.getenv(
    "PROJECT_ROOT",
    r"C:\Users\jamesr4\OneDrive - Memorial Sloan Kettering Cancer Center"
    r"\Documents\GitHub\llm_summarization_br_ca",
))
DATA_PRIVATE_DIR = Path(os.getenv(
    "DATA_PRIVATE_DIR", r"C:\Users\jamesr4\loc\data_private"
))

RAW_DIR    = DATA_PRIVATE_DIR / "raw"
DEID_DIR   = DATA_PRIVATE_DIR / "deidentified"
OUTPUT_DIR = PROJECT_ROOT / "reports"

DEID_DIR.mkdir(parents=True, exist_ok=True)
(DEID_DIR / "pdfs").mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GLOBAL_START = time.time()
timing_rows = []

def log_time(step, detail="", start=None):
    elapsed = time.time() - (start or GLOBAL_START)
    row = {
        "timestamp": datetime.now().isoformat(),
        "step": step,
        "detail": detail,
        "elapsed_s": round(elapsed, 2),
        "wall_clock_s": round(time.time() - GLOBAL_START, 2),
    }
    timing_rows.append(row)
    print(f"[{row['wall_clock_s']:>8.1f}s] {step}: {detail}")

# ── Redaction rules (from NB01) ──────────────────────────────────────────────
@dataclass
class RedactionRule:
    name: str
    pattern: str
    flags: int = re.IGNORECASE

DEFAULT_RULES: List[RedactionRule] = [
    RedactionRule("EMAIL", r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"),
    RedactionRule("URL", r"\bhttps?://\S+\b|\bwww\.\S+\b"),
    RedactionRule("PHONE", r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    RedactionRule("SSN", r"\b\d{3}-\d{2}-\d{4}\b"),
    RedactionRule("MRN_LABEL", r"\b(MRN|Medical\s*Record\s*#|Med\s*Rec|Patient\s*ID)\b"),
    RedactionRule("MRN_NUMBER", r"\b\d{6,10}\b"),
    RedactionRule("DATE_MDY", r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
    RedactionRule("DATE_TEXT", r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+\d{4}\b"),
    RedactionRule("ADDRESS", r"\b\d{1,6}\s+[A-Z0-9][A-Z0-9\s.-]{2,}\s+(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr)\b"),
    RedactionRule("ZIP", r"\b\d{5}(?:-\d{4})?\b"),
]

CONTEXT_LABELS = {"name", "patient", "dob", "dateofbirth", "birth", "mrn", "acct", "account", "accession"}

@dataclass
class DeidConfig:
    dpi: int = 300
    ocr_lang: str = "eng"
    pad_px: int = 4
    contextual_numeric_redaction: bool = True
    redact_after_label_tokens: int = 8
    enable_broad_numeric_redaction: bool = False

# ── Helper functions (from NB01) ─────────────────────────────────────────────
def pil_to_cv(img):
    arr = np.array(img)
    return arr if arr.ndim == 2 else cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def cv_to_pil(arr):
    return Image.fromarray(arr) if arr.ndim == 2 else Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))

def compile_rules(rules):
    return [(r.name, re.compile(r.pattern, r.flags)) for r in rules]

def normalize_token(t):
    return re.sub(r"[^a-z0-9]+", "", t.lower())

def ocr_tokens_with_boxes(pil_img, lang):
    data = pytesseract.image_to_data(pil_img, lang=lang, output_type=pytesseract.Output.DATAFRAME)
    data = data.dropna(subset=["text"])
    data["text"] = data["text"].astype(str)
    data = data[data["text"].str.strip().ne("")]
    data["conf"] = pd.to_numeric(data["conf"], errors="coerce").fillna(-1)
    return data

def token_matches_any_rule(token, compiled_rules):
    for name, pat in compiled_rules:
        if pat.search(token):
            return name
    return None

def get_redaction_boxes(ocr_df, compiled_rules, config):
    redactions = []
    ocr_df = ocr_df.copy()
    ocr_df["norm"] = ocr_df["text"].map(normalize_token)
    for idx, row in ocr_df.iterrows():
        token = row["text"]
        rule = token_matches_any_rule(token, compiled_rules)
        if rule == "MRN_NUMBER" and not config.enable_broad_numeric_redaction:
            rule = None
        if rule:
            redactions.append({"rule": rule, "text": token, "left": int(row["left"]),
                               "top": int(row["top"]), "width": int(row["width"]), "height": int(row["height"])})
    if config.contextual_numeric_redaction:
        for _, line_df in ocr_df.groupby(["block_num", "par_num", "line_num"]):
            line_df = line_df.sort_values("word_num")
            norms = line_df["norm"].tolist()
            rows_list = line_df.to_dict("records")
            for i, n in enumerate(norms):
                if n in CONTEXT_LABELS:
                    for j in range(i + 1, min(i + 1 + config.redact_after_label_tokens, len(rows_list))):
                        rr = rows_list[j]
                        if normalize_token(rr["text"]) == "":
                            continue
                        redactions.append({"rule": "CONTEXT_AFTER_LABEL", "text": rr["text"],
                                           "left": int(rr["left"]), "top": int(rr["top"]),
                                           "width": int(rr["width"]), "height": int(rr["height"])})
    seen = set()
    uniq = []
    for r in redactions:
        k = (r["left"], r["top"], r["width"], r["height"], r["text"], r["rule"])
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    return uniq

def apply_redactions_to_image(cv_img, redactions, pad_px):
    out = cv_img.copy()
    h, w = out.shape[:2]
    for r in redactions:
        x1, y1 = max(0, r["left"] - pad_px), max(0, r["top"] - pad_px)
        x2, y2 = min(w, r["left"] + r["width"] + pad_px), min(h, r["top"] + r["height"] + pad_px)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)
    return out

MAX_PIXMAP_BYTES = 150_000_000  # 150 MB safety limit

def render_pdf_page_to_pil(doc, page_index, dpi):
    page = doc.load_page(page_index)
    # Estimate pixmap size; fall back to lower DPI if too large
    rect = page.rect
    for try_dpi in [dpi, 200, 150, 100]:
        scale = try_dpi / 72.0
        est_bytes = int(rect.width * scale) * int(rect.height * scale) * 3
        if est_bytes <= MAX_PIXMAP_BYTES:
            mat = fitz.Matrix(scale, scale)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    # Absolute minimum
    mat = fitz.Matrix(72 / 72.0, 72 / 72.0)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

def images_to_pdf(pil_images, out_path):
    if not pil_images:
        raise ValueError("No pages to write.")
    rgb_imgs = [im.convert("RGB") for im in pil_images]
    rgb_imgs[0].save(out_path, save_all=True, append_images=rgb_imgs[1:])

def generate_case_id(filename):
    h = hashlib.sha256(filename.encode()).hexdigest()[:12]
    return f"CASE_{h.upper()}"

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Build patient mapping
# ══════════════════════════════════════════════════════════════════════════════
log_time("STEP 1", "Building patient mapping")
step_start = time.time()

mapping_rows = []
for pdf_path in sorted(RAW_DIR.rglob("*.pdf")):
    mapping_rows.append({
        "case_id": generate_case_id(pdf_path.stem),
        "original_filename": pdf_path.name,
        "original_path": str(pdf_path),
    })
mapping_df = pd.DataFrame(mapping_rows)
mapping_df.to_csv(DEID_DIR / "patient_case_id_mapping.csv", index=False)
log_time("STEP 1 DONE", f"{len(mapping_df)} cases mapped", step_start)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Deidentify all PDFs with per-file timing
# ══════════════════════════════════════════════════════════════════════════════
log_time("STEP 2", f"Deidentifying {len(mapping_df)} PDFs")
step_start = time.time()

compiled_rules = compile_rules(DEFAULT_RULES)
config = DeidConfig(dpi=300, contextual_numeric_redaction=True, redact_after_label_tokens=8)
pdf_log_rows = []
out_dir = DEID_DIR / "pdfs"

pdf_paths = sorted(RAW_DIR.rglob("*.pdf"))
total_pdfs = len(pdf_paths)
total_pages_processed = 0
total_redactions_applied = 0
skipped_existing = 0

# Resume support: check which case_ids already have output PDFs
existing_outputs = {p.stem for p in out_dir.glob("*.pdf")}

for pdf_i, pdf_path in enumerate(tqdm(pdf_paths, desc="Deidentifying PDFs")):
    pdf_start = time.time()
    case_id = generate_case_id(pdf_path.stem)

    # Skip already-processed PDFs (resume support)
    if case_id in existing_outputs:
        skipped_existing += 1
        continue

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        pdf_log_rows.append({"case_id": case_id, "file": str(pdf_path),
                             "status": "ERROR_OPEN", "error": str(e)})
        continue

    redacted_pages = []
    file_redactions = 0
    page_errors = 0
    for p in range(doc.page_count):
        try:
            pil_img = render_pdf_page_to_pil(doc, p, config.dpi)
            ocr_df = ocr_tokens_with_boxes(pil_img, config.ocr_lang)
            redactions = get_redaction_boxes(ocr_df, compiled_rules, config)
            file_redactions += len(redactions)
            cv_img = pil_to_cv(pil_img)
            cv_redacted = apply_redactions_to_image(cv_img, redactions, config.pad_px)
            redacted_pages.append(cv_to_pil(cv_redacted))
        except Exception as e:
            page_errors += 1
            print(f"  WARNING: page {p+1} of {pdf_path.name} failed: {e}")
            # Insert a blank page placeholder
            redacted_pages.append(Image.new("RGB", (612, 792), (255, 255, 255)))

    out_path = out_dir / f"{case_id}.pdf"
    try:
        images_to_pdf(redacted_pages, out_path)
    except Exception as e:
        pdf_log_rows.append({"case_id": case_id, "file": pdf_path.name,
                             "status": "ERROR_WRITE", "error": str(e)})
        doc.close()
        continue

    pdf_elapsed = time.time() - pdf_start
    total_pages_processed += doc.page_count
    total_redactions_applied += file_redactions

    pdf_log_rows.append({
        "case_id": case_id,
        "file": pdf_path.name,
        "status": "DONE" if page_errors == 0 else f"DONE_WITH_{page_errors}_PAGE_ERRORS",
        "pages": doc.page_count,
        "redactions": file_redactions,
        "page_errors": page_errors,
        "time_s": round(pdf_elapsed, 2),
        "output_file": str(out_path),
    })
    doc.close()

    # Progress every 50 PDFs
    if (pdf_i + 1) % 50 == 0:
        elapsed_total = time.time() - step_start
        processed_count = pdf_i + 1 - skipped_existing
        rate = processed_count / elapsed_total if elapsed_total > 0 else 1
        remaining_count = total_pdfs - pdf_i - 1
        remaining_s = remaining_count / rate if rate > 0 else 0
        log_time("PROGRESS",
                 f"{pdf_i+1}/{total_pdfs} PDFs ({skipped_existing} resumed) | "
                 f"{total_pages_processed} pages | "
                 f"{total_redactions_applied} redactions | "
                 f"ETA: {remaining_s/60:.0f} min")

# Save PDF log
pd.DataFrame(pdf_log_rows).to_csv(DEID_DIR / "deid_pdf_log.csv", index=False)
log_time("STEP 2 DONE",
         f"{total_pdfs} PDFs, {total_pages_processed} pages, "
         f"{total_redactions_applied} redactions", step_start)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Deidentify Validation Excel
# ══════════════════════════════════════════════════════════════════════════════
log_time("STEP 3", "Deidentifying validation Excel")
step_start = time.time()

EXCEL_PATH = RAW_DIR / "merged_llm_summary_validation_datasheet.xlsx"
df_raw = pd.read_excel(EXCEL_PATH)
status_cols = [c for c in df_raw.columns if "_status_" in c.lower()]
phi_candidates = [c for c in df_raw.columns if c not in status_cols
                  and df_raw[c].dtype == "object"
                  and c.lower() not in ["case_id", "observation_id"]]

compiled = compile_rules(DEFAULT_RULES)

def redact_cell(val):
    if pd.isna(val) or not isinstance(val, str):
        return val
    out = val
    for name, pat in compiled:
        out = pat.sub(f"[{name}]", out)
    return out

df_deid = df_raw.copy()
for col in phi_candidates:
    df_deid[col] = df_deid[col].apply(redact_cell)

# Save to BOTH expected locations for downstream notebooks
deid_excel_path = DEID_DIR / "validation_datasheet_deidentified.xlsx"
df_deid.to_excel(deid_excel_path, index=False)

# Also save where NB02 expects it
deid_excel_nb02 = RAW_DIR / "merged_llm_summary_validation_datasheet_deidentified.xlsx"
df_deid.to_excel(deid_excel_nb02, index=False)

log_time("STEP 3 DONE",
         f"Excel deidentified ({df_deid.shape}), saved to 2 locations", step_start)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Save timing report
# ══════════════════════════════════════════════════════════════════════════════
total_elapsed = time.time() - GLOBAL_START
log_time("ALL DONE", f"Total wall clock: {total_elapsed/3600:.2f} hours")

df_timing = pd.DataFrame(timing_rows)
df_timing.to_csv(OUTPUT_DIR / "nb01_timing_log.csv", index=False)
print(f"\nTiming log saved: {OUTPUT_DIR / 'nb01_timing_log.csv'}")
print(f"PDF deid log saved: {DEID_DIR / 'deid_pdf_log.csv'}")
print(f"Patient mapping saved: {DEID_DIR / 'patient_case_id_mapping.csv'}")
print(f"Deidentified Excel saved: {deid_excel_path}")
print(f"Deidentified Excel (NB02 path): {deid_excel_nb02}")
print(f"\n{'='*60}")
print(f"TOTAL TIME: {total_elapsed:.0f}s ({total_elapsed/3600:.2f} hours)")
print(f"{'='*60}")
