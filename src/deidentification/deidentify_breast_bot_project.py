"""
Deidentify All Source Documents in Breast Bot Project
Processes all PDFs in surgeon/case folder structure with HIPAA Safe Harbor 18 PHI identifiers
Maintains original folder architecture: surgeon_name -> case_folder -> deidentified_files
"""

import os
import re
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict

import argparse

import fitz  # PyMuPDF
import pytesseract
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

# ── Tesseract configuration ───────────────────────────────────────────────────
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Users\jamesr4\AppData\Local\miniforge3\Library\bin\tesseract.exe"
)
os.environ["TESSDATA_PREFIX"] = (
    r"C:\Users\jamesr4\AppData\Local\miniforge3\share\tessdata"
)


# ── Configuration ─────────────────────────────────────────────────────────────
_DEFAULT_OUTPUT = Path(r"C:\Users\jamesr4\loc\data_private\breast_bot_deidentified")


# ── HIPAA Safe Harbor 18 PHI Identifiers ─────────────────────────────────────
@dataclass
class RedactionRule:
    name: str
    pattern: str
    flags: int = re.IGNORECASE


DEFAULT_RULES: List[RedactionRule] = [
    # 1. Names - handled via CONTEXT_AFTER_LABEL
    
    # 2. Geographic subdivisions smaller than state
    RedactionRule(
        "ADDRESS",
        r"\b\d{1,6}\s+[A-Z0-9][A-Z0-9\s.-]{2,}\s+(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Place|Pl|Way|Circle|Cir)\b"
    ),
    RedactionRule(
        "CITY",
        r"\b(?:New\s+York|Brooklyn|Manhattan|Queens|Bronx|Staten\s+Island|Los\s+Angeles|Chicago|Houston|Phoenix|Philadelphia|San\s+Antonio|San\s+Diego|Dallas|San\s+Jose|Boston|Seattle|Denver|Miami|Atlanta|Detroit|Baltimore|Portland|Minneapolis|Cleveland|Pittsburgh|Cincinnati|Newark|Buffalo|Rochester|Syracuse|Albany|Yonkers|White\s+Plains)\b"
    ),
    RedactionRule("ZIP", r"\b\d{5}(?:-\d{4})?\b"),
    
    # 3. Dates (except year) - all date formats
    RedactionRule("DATE_MDY", r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
    RedactionRule(
        "DATE_TEXT",
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b"
    ),
    RedactionRule(
        "DATE_MMDDYY",
        r"\b(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12][0-9]|3[01])[/-]\d{2,4}\b"
    ),
    RedactionRule("DATE_DOB", r"\b(?:DOB|Date\s*of\s*Birth)[\s:]*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
    
    # 4. Telephone numbers
    RedactionRule(
        "PHONE",
        r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    ),
    
    # 5. Fax numbers
    RedactionRule(
        "FAX",
        r"\b(?:fax|facsimile)[\s:]*(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    ),
    
    # 6. Email addresses
    RedactionRule(
        "EMAIL",
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"
    ),
    
    # 7. Social Security numbers
    RedactionRule("SSN", r"\b\d{3}-\d{2}-\d{4}\b"),
    
    # 8. Medical record numbers
    RedactionRule(
        "MRN_LABEL",
        r"\b(MRN|Medical\s*Record\s*#?|Med\s*Rec|Patient\s*ID|Record\s*Number)\b"
    ),
    RedactionRule("MRN_NUMBER", r"\b\d{6,10}\b"),
    
    # 9. Health plan beneficiary numbers
    RedactionRule(
        "HEALTH_PLAN",
        r"\b(?:member|policy|subscriber|beneficiary)\s*(?:id|#|number|no)[\s:]*[A-Z0-9]{6,}\b"
    ),
    
    # 10. Account numbers
    RedactionRule(
        "ACCOUNT",
        r"\b(?:account|acct)\s*(?:id|#|number|no)[\s:]*[A-Z0-9]{6,}\b"
    ),
    
    # 11. Certificate/license numbers
    RedactionRule(
        "CERTIFICATE",
        r"\b(?:certificate|license|permit)\s*(?:id|#|number|no)[\s:]*[A-Z0-9]{6,}\b"
    ),
    
    # 12. Vehicle identifiers
    RedactionRule(
        "VEHICLE",
        r"\b(?:VIN|vehicle\s*id|plate)[\s:]*[A-Z0-9]{6,}\b"
    ),
    
    # 13. Device identifiers/serial numbers
    RedactionRule(
        "DEVICE",
        r"\b(?:device|serial|equipment)\s*(?:id|#|number|no)[\s:]*[A-Z0-9]{6,}\b"
    ),
    
    # 14. Web URLs
    RedactionRule("URL", r"\bhttps?://\S+\b|\bwww\.\S+\b"),
    
    # 15. IP addresses
    RedactionRule("IP_ADDRESS", r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    
    # 16. Biometric identifiers - manual review required
    # 17. Full-face photos - manual review required
    
    # 18. Unique identifying codes
    RedactionRule("UNIQUE_ID", r"\b[A-Z]{2,}\d{6,}\b"),
]

# Context labels for names and other identifiers
CONTEXT_LABELS = {
    "name", "patient", "dob", "dateofbirth", "birth", "mrn", "acct", "account",
    "accession", "doctor", "physician", "surgeon", "provider", "initials",
    "id", "identifier", "number", "no", "ssn", "social", "attending",
    "resident", "fellow", "nurse", "technician", "radiologist", "pathologist"
}


@dataclass
class DeidConfig:
    dpi: int = 300
    ocr_lang: str = "eng"
    pad_px: int = 4
    contextual_numeric_redaction: bool = True
    redact_after_label_tokens: int = 8
    enable_broad_numeric_redaction: bool = False


# ── Helper Functions ──────────────────────────────────────────────────────────
def pil_to_cv(img: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV array"""
    arr = np.array(img)
    if arr.ndim == 2:
        return arr
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def cv_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert OpenCV array to PIL Image"""
    if arr.ndim == 2:
        return Image.fromarray(arr)
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def compile_rules(rules: List[RedactionRule]) -> List[Tuple[str, re.Pattern]]:
    """Compile regex patterns from rules"""
    return [(r.name, re.compile(r.pattern, r.flags)) for r in rules]


def normalize_token(t: str) -> str:
    """Normalize token for matching context labels"""
    return re.sub(r"[^a-z0-9]+", "", t.lower())


def ocr_tokens_with_boxes(pil_img: Image.Image, lang: str) -> pd.DataFrame:
    """Extract text tokens and bounding boxes via OCR"""
    data = pytesseract.image_to_data(
        pil_img, lang=lang, output_type=pytesseract.Output.DATAFRAME
    )
    data = data.dropna(subset=["text"])
    data["text"] = data["text"].astype(str)
    data = data[data["text"].str.strip().ne("")]
    data["conf"] = pd.to_numeric(data["conf"], errors="coerce").fillna(-1)
    return data


def token_matches_any_rule(token: str, compiled_rules):
    """Check if token matches any redaction rule"""
    for name, pat in compiled_rules:
        if pat.search(token):
            return name
    return None


def get_redaction_boxes(ocr_df, compiled_rules, config):
    """Identify all text regions to redact"""
    redactions = []
    ocr_df = ocr_df.copy()
    ocr_df["norm"] = ocr_df["text"].map(normalize_token)
    
    # Rule-based redaction
    for idx, row in ocr_df.iterrows():
        token = row["text"]
        rule = token_matches_any_rule(token, compiled_rules)
        
        # Skip broad numeric redaction unless enabled
        if rule == "MRN_NUMBER" and not config.enable_broad_numeric_redaction:
            rule = None
        
        if rule:
            redactions.append({
                "rule": rule,
                "text": token,
                "left": int(row["left"]),
                "top": int(row["top"]),
                "width": int(row["width"]),
                "height": int(row["height"])
            })
    
    # Contextual redaction (names after labels like "Patient:", "Doctor:", etc.)
    if config.contextual_numeric_redaction:
        for _, line_df in ocr_df.groupby(["block_num", "par_num", "line_num"]):
            line_df = line_df.sort_values("word_num")
            norms = line_df["norm"].tolist()
            rows = line_df.to_dict("records")
            
            for i, n in enumerate(norms):
                if n in CONTEXT_LABELS:
                    # Redact next N tokens after context label
                    for j in range(i + 1, min(i + 1 + config.redact_after_label_tokens, len(rows))):
                        rr = rows[j]
                        if normalize_token(rr["text"]) == "":
                            continue
                        redactions.append({
                            "rule": "CONTEXT_AFTER_LABEL",
                            "text": rr["text"],
                            "left": int(rr["left"]),
                            "top": int(rr["top"]),
                            "width": int(rr["width"]),
                            "height": int(rr["height"])
                        })
    
    # Remove duplicates
    seen = set()
    uniq = []
    for r in redactions:
        k = (r["left"], r["top"], r["width"], r["height"], r["text"], r["rule"])
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    
    return uniq


def apply_redactions_to_image(cv_img, redactions, pad_px):
    """Apply black boxes to redact PHI regions"""
    out = cv_img.copy()
    h, w = out.shape[:2]
    
    for r in redactions:
        x1 = max(0, r["left"] - pad_px)
        y1 = max(0, r["top"] - pad_px)
        x2 = min(w, r["left"] + r["width"] + pad_px)
        y2 = min(h, r["top"] + r["height"] + pad_px)
        
        # Draw black rectangle
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)
    
    return out


def render_pdf_page_to_pil(doc, page_index, dpi):
    """Render PDF page to PIL Image at specified DPI"""
    page = doc.load_page(page_index)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def images_to_pdf(pil_images, out_path):
    """Save list of PIL images as PDF"""
    if not pil_images:
        raise ValueError("No pages to write.")
    rgb_imgs = [im.convert("RGB") for im in pil_images]
    rgb_imgs[0].save(out_path, save_all=True, append_images=rgb_imgs[1:])


def generate_case_id(surgeon: str, case_folder: str, filename: str) -> str:
    """Generate deterministic case_id from path components"""
    combined = f"{surgeon}/{case_folder}/{filename}"
    h = hashlib.sha256(combined.encode()).hexdigest()[:12]
    return f"CASE_{h.upper()}"


# ── Main Deidentification Logic ───────────────────────────────────────────────
def deidentify_pdf(
    pdf_path: Path,
    output_path: Path,
    case_id: str,
    rules: List[RedactionRule],
    config: DeidConfig
) -> Dict:
    """Deidentify a single PDF file"""
    compiled_rules = compile_rules(rules)
    
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        return {
            "case_id": case_id,
            "status": "ERROR_OPEN",
            "error": str(e),
            "pages": 0,
            "total_redactions": 0
        }
    
    redacted_pages = []
    total_redactions = 0
    page_details = []
    
    for p in range(doc.page_count):
        try:
            # Render page to image
            pil_img = render_pdf_page_to_pil(doc, p, config.dpi)
            
            # OCR to get text and bounding boxes
            ocr_df = ocr_tokens_with_boxes(pil_img, config.ocr_lang)
            
            # Identify redaction regions
            redactions = get_redaction_boxes(ocr_df, compiled_rules, config)
            total_redactions += len(redactions)
            
            # Apply black boxes
            cv_img = pil_to_cv(pil_img)
            cv_redacted = apply_redactions_to_image(cv_img, redactions, config.pad_px)
            redacted_pages.append(cv_to_pil(cv_redacted))
            
            # Log page details
            page_details.append({
                "page": p + 1,
                "redactions": len(redactions),
                "rules": ",".join(sorted(set(r["rule"] for r in redactions)))[:200]
            })
            
        except Exception as e:
            n_pages = doc.page_count
            doc.close()
            return {
                "case_id": case_id,
                "status": f"ERROR_PAGE_{p+1}",
                "error": str(e),
                "pages": n_pages,
                "total_redactions": total_redactions
            }
    
    # Save redacted PDF
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        images_to_pdf(redacted_pages, output_path)
    except Exception as e:
        n_pages = doc.page_count
        doc.close()
        return {
            "case_id": case_id,
            "status": "ERROR_WRITE",
            "error": str(e),
            "pages": n_pages,
            "total_redactions": total_redactions
        }
    
    doc.close()
    
    return {
        "case_id": case_id,
        "status": "SUCCESS",
        "pages": len(redacted_pages),
        "total_redactions": total_redactions,
        "page_details": page_details
    }


def process_breast_bot_project(source_root: Path = None, output_root: Path = None):
    """Process all PDFs in Breast Bot Project folder structure"""

    # Resolve source root
    if source_root is None:
        _ONEDRIVE = Path(r"C:\Users\jamesr4\OneDrive - Memorial Sloan Kettering Cancer Center")
        _matches = [d for d in _ONEDRIVE.iterdir() if d.is_dir() and "Moo" in d.name and "Breast Bot" in d.name]
        if not _matches:
            raise FileNotFoundError(f"Breast Bot Project folder not found in {_ONEDRIVE}")
        source_root = _matches[0]

    if output_root is None:
        output_root = _DEFAULT_OUTPUT

    mapping_csv = output_root / "case_id_mapping.csv"
    log_csv = output_root / "deidentification_log.csv"
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("BREAST BOT PROJECT DEIDENTIFICATION")
    print("=" * 80)
    print(f"Source: {source_root}")
    print(f"Output: {output_root}")
    print("\nUsing HIPAA Safe Harbor 18 PHI Identifiers")
    print(f"Redaction method: Black bounding boxes with {DeidConfig().pad_px}px padding")
    print("=" * 80)
    
    # Scan for all PDFs
    all_pdfs = []
    surgeon_folders = [d for d in source_root.iterdir() if d.is_dir()]
    
    print(f"\nScanning {len(surgeon_folders)} surgeon folders...")
    
    for surgeon_folder in surgeon_folders:
        surgeon_name = surgeon_folder.name
        case_folders = [d for d in surgeon_folder.iterdir() if d.is_dir()]
        
        for case_folder in case_folders:
            case_name = case_folder.name
            pdf_files = list(case_folder.glob("*.pdf"))
            
            for pdf_file in pdf_files:
                all_pdfs.append({
                    "surgeon": surgeon_name,
                    "case": case_name,
                    "filename": pdf_file.name,
                    "source_path": pdf_file,
                    "relative_path": pdf_file.relative_to(source_root)
                })
    
    print(f"Found {len(all_pdfs)} PDF files across all cases")
    
    if len(all_pdfs) == 0:
        print("No PDF files found. Exiting.")
        return
    
    # Create mapping and log structures
    mapping_rows = []
    log_rows = []
    
    # Process each PDF
    config = DeidConfig(
        dpi=300,
        contextual_numeric_redaction=True,
        redact_after_label_tokens=8
    )
    
    print(f"\nProcessing {len(all_pdfs)} PDFs...")
    print("=" * 80)
    
    for pdf_info in tqdm(all_pdfs, desc="Deidentifying PDFs"):
        surgeon = pdf_info["surgeon"]
        case = pdf_info["case"]
        filename = pdf_info["filename"]
        source_path = pdf_info["source_path"]
        
        # Generate case_id
        case_id = generate_case_id(surgeon, case, filename)
        
        # Create output path maintaining folder structure
        output_path = output_root / surgeon / case / f"{case_id}.pdf"
        
        # Skip if already processed
        if output_path.exists():
            mapping_rows.append({
                "case_id": case_id,
                "surgeon": surgeon,
                "case_folder": case,
                "original_filename": filename,
                "original_path": str(source_path),
                "deidentified_path": str(output_path),
                "status": "SKIPPED"
            })
            continue

        # Deidentify
        result = deidentify_pdf(
            pdf_path=source_path,
            output_path=output_path,
            case_id=case_id,
            rules=DEFAULT_RULES,
            config=config
        )
        
        # Add to mapping
        mapping_rows.append({
            "case_id": case_id,
            "surgeon": surgeon,
            "case_folder": case,
            "original_filename": filename,
            "original_path": str(source_path),
            "deidentified_path": str(output_path),
            "status": result["status"]
        })
        
        # Add to log
        log_rows.append({
            "case_id": case_id,
            "surgeon": surgeon,
            "case_folder": case,
            "original_filename": filename,
            "status": result["status"],
            "pages": result.get("pages", 0),
            "total_redactions": result.get("total_redactions", 0),
            "error": result.get("error", "")
        })
        
        # Add page-level details if available
        if "page_details" in result:
            for page_detail in result["page_details"]:
                log_rows.append({
                    "case_id": case_id,
                    "surgeon": surgeon,
                    "case_folder": case,
                    "original_filename": filename,
                    "status": "PAGE_DETAIL",
                    "page": page_detail["page"],
                    "page_redactions": page_detail["redactions"],
                    "rules_applied": page_detail["rules"]
                })
    
    # Save mapping CSV
    mapping_df = pd.DataFrame(mapping_rows)
    mapping_df.to_csv(mapping_csv, index=False)
    print(f"\n[OK] Mapping saved: {mapping_csv}")
    print(f"   Total cases: {len(mapping_df)}")
    
    # Save log CSV
    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(log_csv, index=False)
    print(f"[OK] Log saved: {log_csv}")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("DEIDENTIFICATION SUMMARY")
    print("=" * 80)
    
    success_count = len(mapping_df[mapping_df["status"] == "SUCCESS"])
    error_count = len(mapping_df[mapping_df["status"] != "SUCCESS"])
    total_redactions = log_df[log_df["status"] == "SUCCESS"]["total_redactions"].sum()
    
    print(f"Total PDFs processed: {len(mapping_df)}")
    print(f"  Successful: {success_count}")
    print(f"  Errors:     {error_count}")
    print(f"Total redactions applied: {total_redactions:,.0f}")
    
    # Surgeon breakdown
    print("\nBreakdown by surgeon:")
    surgeon_summary = mapping_df.groupby("surgeon").agg({
        "case_id": "count",
        "status": lambda x: (x == "SUCCESS").sum()
    }).rename(columns={"case_id": "total_files", "status": "successful"})
    print(surgeon_summary.to_string())
    
    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"Deidentified files: {output_root}")
    print(f"Mapping: {mapping_csv}")
    print(f"Log: {log_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deidentify PDFs using HIPAA Safe Harbor 18 PHI rules")
    parser.add_argument(
        "--source", type=Path, default=None,
        help="Root folder containing surgeon/case/PDF structure (default: OneDrive Breast Bot folder)"
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output root folder for deidentified PDFs (default: breast_bot_deidentified)"
    )
    args = parser.parse_args()
    process_breast_bot_project(source_root=args.source, output_root=args.output)
