"""
Source Document Feature Extraction with OCR (Optimized)
======================================================
Extracts text from each case's source PDFs with OCR fallback for scanned documents.
Optimized for memory usage and batch processing.

Usage:
    python source_document_feature_extraction_v3_ocr.py [--batch-size 10]
"""

import os
import sys
import json
import re
import hashlib
import logging
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# PDF / DOCX text extraction
# ---------------------------------------------------------------------------
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import docx  # python-docx
except ImportError:
    docx = None

# OCR support
try:
    import easyocr
    OCR_AVAILABLE = True
    log = logging.getLogger(__name__)
    log.info("EasyOCR available for scanned PDF processing")
except ImportError:
    OCR_AVAILABLE = False
    log = logging.getLogger(__name__)
    log.warning("EasyOCR not installed. To enable OCR for scanned PDFs: pip install easyocr")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "source_doc_features_v3_ocr"

VALIDATION_RAW = RAW_DIR / "merged_llm_summary_validation_datasheet.xlsx"
VALIDATION_DEIDENTIFIED = PROCESSED_DIR / "merged_llm_summary_validation_datasheet_deidentified.xlsx"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Surgeon name → folder name mapping
# ---------------------------------------------------------------------------
SURGEON_NAME_TO_DIR = {
    "Barrio, Andrea": "Andrea Barrio",
    "Choi, Danny": "Danny Choi",
    "Capko, Deborah": "Deborah Capko",
    "Allen, Lisa": "Lisa Allen",
    "Goel, Neha": "Neha Goel",
    "Bayard, Solange": "Solange Bayard",
    "Downs-Canner, Stephanie": "Stephanie Downs-Canner",
    "Pawloski": "Pawloski",
}

# ---------------------------------------------------------------------------
# Element definitions
# ---------------------------------------------------------------------------
ELEMENT_TRIPLES = [
    ("lesion_size_status", "Lesion Size"),
    ("laterality_status", "Lesion Laterality"),
    ("lesion_location_status", "Lesion Location"),
    ("calcifications_asymmetry_status", "Calcifications / Asymmetry"),
    ("additional_enhancement_mri_status", "Additional Enhancement (MRI)"),
    ("extent_status", "Extent"),
    ("accurate_clip_placement_status", "Accurate Clip Placement"),
    ("workup_recommendation_status", "Workup Recommendation"),
    ("Lymph node_status", "Lymph Node"),
    ("chronology_preserved_status", "Chronology Preserved"),
    ("biopsy_method_status", "Biopsy Method"),
    ("invasive_component_size_pathology_status", "Invasive Component Size (Pathology)"),
    ("histologic_diagnosis_status", "Histologic Diagnosis"),
    ("receptor_status", "Receptor Status"),
]

RADIOLOGY_ELEMENTS = {
    "Lesion Size", "Lesion Laterality", "Lesion Location",
    "Calcifications / Asymmetry", "Additional Enhancement (MRI)",
    "Extent", "Accurate Clip Placement", "Workup Recommendation",
    "Lymph Node", "Chronology Preserved", "Biopsy Method",
}

PATHOLOGY_ELEMENTS = {
    "Invasive Component Size (Pathology)",
    "Histologic Diagnosis",
    "Receptor Status",
}


# ===================================================================
# OCR Manager (optimized)
# ===================================================================
class OCRManager:
    """Manages OCR processing with memory optimization."""
    
    def __init__(self):
        self.reader = None
        self.initialized = False
        
    def initialize(self):
        """Initialize OCR reader on demand."""
        if not OCR_AVAILABLE:
            return False
        
        if not self.initialized:
            try:
                self.reader = easyocr.Reader(['en'], gpu=False)  # Force CPU to avoid GPU memory issues
                self.initialized = True
                log.info("OCR initialized (CPU mode)")
                return True
            except Exception as exc:
                log.error(f"Failed to initialize OCR: {exc}")
                return False
        return True
    
    def extract_text_from_image(self, image) -> str:
        """Extract text from a PIL image using OCR."""
        if not self.initialize():
            return ""
        
        try:
            result = self.reader.readtext(image)
            text = " ".join([item[1] for item in result])
            return text
        except Exception as e:
            log.debug(f"OCR extraction failed: {e}")
            return ""


# Global OCR manager
ocr_manager = OCRManager()


# ===================================================================
# Text extraction with OCR fallback
# ===================================================================
def extract_text_from_pdf(path: Path, use_ocr: bool = True) -> Tuple[str, str]:
    """
    Extract text from a PDF file with OCR fallback.
    Returns: (extracted_text, extraction_method)
    """
    # Try pypdf first
    if PdfReader is not None:
        try:
            reader = PdfReader(str(path))
            pages_text = []
            for page in reader.pages:
                text = page.extract_text() or ""
                pages_text.append(text)
            combined = "\n".join(pages_text).strip()
            if len(combined) > 50:  # Reasonable amount of text
                return combined, 'pypdf'
        except Exception as exc:
            log.debug("pypdf failed for %s: %s", path.name, exc)

    # Try pdfplumber
    if pdfplumber is not None:
        try:
            with pdfplumber.open(str(path)) as pdf:
                pages_text = []
                for page in pdf.pages:
                    text = page.extract_text() or ""
                    pages_text.append(text)
                combined = "\n".join(pages_text).strip()
                if len(combined) > 50:
                    return combined, 'pdfplumber'
        except Exception as exc:
            log.debug("pdfplumber failed for %s: %s", path.name, exc)

    # OCR fallback
    if use_ocr and OCR_AVAILABLE:
        try:
            log.debug("Attempting OCR for %s", path.name)
            with pdfplumber.open(str(path)) as pdf:
                all_text = []
                
                for page_num, page in enumerate(pdf.pages):
                    try:
                        # Convert page to image
                        if hasattr(page, 'to_image'):
                            img = page.to_image()
                            # Extract text using OCR
                            page_text = ocr_manager.extract_text_from_image(img.original)
                            if page_text:
                                all_text.append(f"--- Page {page_num + 1} (OCR) ---\n{page_text}")
                    except Exception as e:
                        log.debug("OCR failed for page %d of %s: %s", page_num + 1, path.name, e)
                
                combined = "\n\n".join(all_text).strip()
                if len(combined) > 10:
                    log.info("OCR extracted %d chars from %s", len(combined), path.name)
                    return combined, 'ocr'
        except Exception as exc:
            log.warning("OCR processing failed for %s: %s", path.name, exc)

    return "", 'failed'


def extract_text_from_docx(path: Path) -> Tuple[str, str]:
    """Extract text from a DOCX file."""
    if docx is None:
        return "", "docx_not_available"
    try:
        doc = docx.Document(str(path))
        text = "\n".join(p.text for p in doc.paragraphs).strip()
        return text, "docx" if text else "docx_empty"
    except Exception as exc:
        log.warning("Failed to read DOCX %s: %s", path.name, exc)
        return "", "docx_failed"


# ===================================================================
# Utility functions
# ===================================================================
def classify_report(filename: str) -> str:
    """Classify a source file into a report category."""
    fn = filename.lower()
    if "hpi_ai" in fn:
        return "hpi_ai"
    if "hpi_human" in fn:
        return "hpi_human"
    if "path" in fn or "biopsy" in fn or "bopsy" in fn:
        return "pathology"
    if "imaging" in fn or "mammo" in fn or "mri" in fn or "us" in fn:
        return "radiology"
    if "genetic" in fn or "biomarker" in fn:
        return "genetics"
    return "other"


def compute_text_features(text: str) -> dict:
    """Compute basic text features from extracted text."""
    if not text:
        return {
            "char_count": 0,
            "word_count": 0,
            "line_count": 0,
            "sentence_count": 0,
            "avg_word_length": 0.0,
            "numeric_token_count": 0,
            "unique_word_count": 0,
            "lexical_diversity": 0.0,
        }

    words = text.split()
    word_count = len(words)
    unique_words = set(w.lower() for w in words)
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    numeric_tokens = [w for w in words if re.search(r"\d", w)]

    return {
        "char_count": len(text),
        "word_count": word_count,
        "line_count": text.count("\n") + 1,
        "sentence_count": len(sentences),
        "avg_word_length": round(np.mean([len(w) for w in words]), 2) if words else 0.0,
        "numeric_token_count": len(numeric_tokens),
        "unique_word_count": len(unique_words),
        "lexical_diversity": round(len(unique_words) / word_count, 4) if word_count else 0.0,
    }


def annotation_outcome(source_val, annotator_val) -> str:
    """Classify a single annotation into TP/FN/FP/TN/unknown."""
    if pd.isna(source_val) or pd.isna(annotator_val):
        return "not_evaluated"
    source_val = int(source_val) if not isinstance(source_val, str) else source_val
    if source_val == 1 and annotator_val == 1:
        return "TP"
    if source_val == 1 and annotator_val == 2:
        return "FN"
    if source_val == 0 and annotator_val == 3:
        return "FP"
    if source_val == 0 and (annotator_val == "N/A" or str(annotator_val).strip().upper() == "N/A"):
        return "TN"
    return "unknown"


def is_correct(outcome: str) -> Optional[bool]:
    """Return True for TP/TN, False for FP/FN, None for not_evaluated."""
    if outcome in ("TP", "TN"):
        return True
    if outcome in ("FP", "FN"):
        return False
    return None


def is_fabrication(outcome: str) -> bool:
    """Return True if the annotation is a fabrication (FP)."""
    return outcome == "FP"


def build_case_folder_map(raw_dir: Path) -> dict:
    """Build a mapping from (surgeon_dir_name, patient_initials, case_type) → folder path."""
    case_folders = {}
    for surgeon_dir in raw_dir.iterdir():
        if not surgeon_dir.is_dir():
            continue
        for case_dir in surgeon_dir.iterdir():
            if not case_dir.is_dir():
                continue
            parts = case_dir.name.split("_")
            if len(parts) >= 2:
                patient_init = parts[1]
                case_type = "invasive" if "invasive" in case_dir.name.lower() else "DCIS"
                key = (surgeon_dir.name, patient_init, case_type)
                case_folders[key] = case_dir
    return case_folders


# ===================================================================
# Main extraction pipeline
# ===================================================================
def extract_case_features(case_dir: Path, use_ocr: bool = True) -> dict:
    """Extract features from all source documents in a case folder."""
    files_info = []
    all_text_combined = []
    report_type_counts = defaultdict(int)
    report_type_word_counts = defaultdict(int)
    extraction_methods = defaultdict(int)

    for f in sorted(case_dir.iterdir()):
        if f.name.startswith(".") or f.name.startswith("_"):
            continue
        if f.suffix.lower() not in (".pdf", ".docx", ".doc", ".txt"):
            continue

        report_type = classify_report(f.name)
        
        # Extract text with OCR fallback
        if f.suffix.lower() == ".pdf":
            text, method = extract_text_from_pdf(f, use_ocr=use_ocr)
        else:
            text, method = extract_text_from_docx(f)
        
        features = compute_text_features(text)

        file_info = {
            "filename": f.name,
            "report_type": report_type,
            "file_size_bytes": f.stat().st_size,
            "extraction_method": method,
            "text_content": text,
            "text_hash": hashlib.md5(text.encode()).hexdigest()[:12] if text else None,
            **features,
        }
        files_info.append(file_info)

        # Track extraction methods
        extraction_methods[method] += 1

        # Only include source documents (not hpi_ai/hpi_human) in combined text
        if report_type not in ("hpi_ai", "hpi_human"):
            all_text_combined.append(text)
            report_type_counts[report_type] += 1
            report_type_word_counts[report_type] += features["word_count"]

    combined_source_text = "\n\n".join(t for t in all_text_combined if t)
    combined_features = compute_text_features(combined_source_text)

    # Count source document types
    n_radiology = report_type_counts.get("radiology", 0)
    n_pathology = report_type_counts.get("pathology", 0)
    n_genetics = report_type_counts.get("genetics", 0)
    n_source_docs = n_radiology + n_pathology + n_genetics + report_type_counts.get("other", 0)

    has_hpi_ai = any(fi["report_type"] == "hpi_ai" for fi in files_info)
    has_hpi_human = any(fi["report_type"] == "hpi_human" for fi in files_info)

    return {
        "case_folder": case_dir.name,
        "total_files": len(files_info),
        "n_source_documents": n_source_docs,
        "n_radiology_reports": n_radiology,
        "n_pathology_reports": n_pathology,
        "n_genetics_reports": n_genetics,
        "has_hpi_ai": has_hpi_ai,
        "has_hpi_human": has_hpi_human,
        "text_extraction_successful": len(combined_source_text) > 100,
        "combined_source_text": combined_source_text,
        "combined_source_text_features": combined_features,
        "report_type_word_counts": dict(report_type_word_counts),
        "extraction_methods": dict(extraction_methods),
        "files": files_info,
    }


def build_annotation_record(
    row: pd.Series,
    case_features: dict,
    case_idx: int,
) -> dict:
    """Build a single case-level JSON record combining source doc features with annotation outcomes."""
    elements_data = {}
    total_correct_human = 0
    total_correct_ai = 0
    total_evaluated = 0
    total_fabrication_human = 0
    total_fabrication_ai = 0

    for col_prefix, display_name in ELEMENT_TRIPLES:
        source_col = f"{col_prefix}_source"
        human_col = f"{col_prefix}_human"
        ai_col = f"{col_prefix}_ai"

        source_val = row.get(source_col)
        human_val = row.get(human_col)
        ai_val = row.get(ai_col)

        human_outcome = annotation_outcome(source_val, human_val)
        ai_outcome = annotation_outcome(source_val, ai_val)

        human_correct = is_correct(human_outcome)
        ai_correct = is_correct(ai_outcome)

        domain = "radiology" if display_name in RADIOLOGY_ELEMENTS else "pathology"

        elements_data[display_name] = {
            "domain": domain,
            "source_present": int(source_val) if pd.notna(source_val) else None,
            "human_annotation": human_val if pd.notna(human_val) else None,
            "ai_annotation": ai_val if pd.notna(ai_val) else None,
            "human_outcome": human_outcome,
            "ai_outcome": ai_outcome,
            "human_correct": human_correct,
            "ai_correct": ai_correct,
            "human_fabrication": is_fabrication(human_outcome),
            "ai_fabrication": is_fabrication(ai_outcome),
        }

        if human_correct is not None:
            total_evaluated += 1
            if human_correct:
                total_correct_human += 1
            if is_fabrication(human_outcome):
                total_fabrication_human += 1
        if ai_correct is not None:
            if ai_correct:
                total_correct_ai += 1
            if is_fabrication(ai_outcome):
                total_fabrication_ai += 1

    case_accuracy_human = total_correct_human / total_evaluated if total_evaluated > 0 else None
    case_accuracy_ai = total_correct_ai / total_evaluated if total_evaluated > 0 else None

    record = {
        "case_index": case_idx,
        "tumor_type": "invasive" if row.get("tumor_invasive_dcis") == 1 else "DCIS",
        "complex_case": bool(row.get("complex_case_status", 0)),
        "surgeon_id": row.get("surgeon_id", row.get("surgeon", "unknown")),
        "source_document_features": case_features,
        "elements": elements_data,
        "case_level_summary": {
            "total_elements_evaluated": total_evaluated,
            "total_correct_human": total_correct_human,
            "total_correct_ai": total_correct_ai,
            "case_accuracy_human": round(case_accuracy_human, 4) if case_accuracy_human is not None else None,
            "case_accuracy_ai": round(case_accuracy_ai, 4) if case_accuracy_ai is not None else None,
            "total_fabrication_human": total_fabrication_human,
            "total_fabrication_ai": total_fabrication_ai,
        },
    }

    return record


# ===================================================================
# Main
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description="Extract features with OCR support")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Number of cases to process before saving checkpoint")
    parser.add_argument("--no-ocr", action="store_true",
                        help="Disable OCR processing")
    parser.add_argument("--resume", type=int,
                        help="Resume from case index")
    args = parser.parse_args()
    
    log.info("Starting source document feature extraction with OCR...")
    
    if args.no_ocr:
        log.info("OCR disabled by --no-ocr flag")
    elif not OCR_AVAILABLE:
        log.warning("EasyOCR not installed. OCR will be skipped.")
        log.info("To install OCR support: pip install easyocr")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load validation data
    if VALIDATION_RAW.exists():
        df_raw = pd.read_excel(VALIDATION_RAW)
        log.info("Loaded raw validation data: %d rows", len(df_raw))
    else:
        log.error("Raw validation file not found: %s", VALIDATION_RAW)
        sys.exit(1)

    # Load deidentified data for surgeon_id
    if VALIDATION_DEIDENTIFIED.exists():
        df_deident = pd.read_excel(VALIDATION_DEIDENTIFIED)
        log.info("Loaded deidentified validation data: %d rows", len(df_deident))
    else:
        df_deident = None
        log.warning("Deidentified validation file not found; using raw data only.")

    # Build case folder map
    case_folder_map = build_case_folder_map(RAW_DIR)
    log.info("Found %d case folders in raw directory", len(case_folder_map))

    # Process cases in batches
    all_records = []
    matched_count = 0
    unmatched_count = 0
    extraction_stats = defaultdict(int)
    
    start_idx = args.resume if args.resume is not None else 0
    
    for idx in range(start_idx, len(df_raw)):
        row = df_raw.iloc[idx]
        surgeon = row.get("surgeon", "")
        patient_init = row.get("patient_initials", "")
        tumor_type = "invasive" if row.get("tumor_invasive_dcis") == 1 else "DCIS"

        surgeon_dir = SURGEON_NAME_TO_DIR.get(surgeon, surgeon)
        key = (surgeon_dir, patient_init, tumor_type)

        case_features = None
        if key in case_folder_map:
            case_dir = case_folder_map[key]
            case_features = extract_case_features(case_dir, use_ocr=not args.no_ocr)
            matched_count += 1
            
            # Track extraction methods
            for method, count in case_features.get("extraction_methods", {}).items():
                extraction_stats[method] += 1
        else:
            unmatched_count += 1

        # Merge surgeon_id from deidentified data if available
        merged_row = row.to_dict()
        if df_deident is not None and idx < len(df_deident):
            merged_row["surgeon_id"] = df_deident.iloc[idx].get("surgeon_id", "unknown")

        record = build_annotation_record(
            pd.Series(merged_row),
            case_features,
            case_idx=idx,
        )
        all_records.append(record)

        # Save individual case JSON
        case_json_path = OUTPUT_DIR / f"case_{idx:04d}.json"
        with open(case_json_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, default=str)

        # Batch save checkpoint
        if (idx + 1) % args.batch_size == 0:
            checkpoint_path = OUTPUT_DIR / f"checkpoint_{idx:04d}.json"
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(all_records, f, indent=2, default=str)
            log.info("Checkpoint saved at case %d", idx)

        # Progress update
        if (idx + 1) % 10 == 0:
            log.info("Processed %d/%d cases", idx + 1, len(df_raw))

    # Final save
    combined_path = OUTPUT_DIR / "all_cases_features.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, default=str)
    log.info("Saved combined features to %s", combined_path)

    # Summary
    log.info(
        "Processed %d cases: %d matched to folders, %d unmatched",
        len(all_records), matched_count, unmatched_count,
    )

    log.info("Extraction method usage:")
    for method, count in extraction_stats.items():
        log.info("  %s: %d files", method, count)

    matched_records = [r for r in all_records if r["source_document_features"] is not None]
    if matched_records:
        successful = sum(1 for r in matched_records 
                        if r["source_document_features"].get("text_extraction_successful", False))
        log.info("Text extraction success: %d/%d cases (%.1f%%)", 
                successful, len(matched_records), 100*successful/len(matched_records))

    log.info("Done.")


if __name__ == "__main__":
    main()
