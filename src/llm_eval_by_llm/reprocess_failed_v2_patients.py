"""
Reprocess failed v2 prompt test patients.

This script reprocesses only the patients that failed or had malformed JSON:
- MM_39103258: API credits ran out
- KS_39111396: API credits ran out
- MC_39102467: Malformed JSON (plain text response)
- RW_39091629: Malformed JSON (syntax error)
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import json
import base64
from anthropic import Anthropic
from pdf2image import convert_from_path
from io import BytesIO
from tqdm import tqdm
import cv2
import numpy as np

load_dotenv(override=True)

# Paths
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT",
    r"C:\Users\jamesr4\OneDrive - Memorial Sloan Kettering Cancer Center\Documents\GitHub\llm_summarization_br_ca"))
DATA_PRIVATE_DIR = Path(os.getenv("DATA_PRIVATE_DIR", r"C:\Users\jamesr4\loc\data_private"))
RAW_DIR = DATA_PRIVATE_DIR / "raw"
OUTPUT_DIR = DATA_PRIVATE_DIR / "v2_prompt_test"
PROMPT_FILE = PROJECT_ROOT / "prompts" / "updated_developer_prompt_v2.txt"

# API setup
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
MODEL = "claude-sonnet-4-20250514"

# Patients to reprocess
PATIENTS_TO_REPROCESS = [
    {"mrn": 39103258, "initials": "MM", "folder": "Lee/ML_MM_INV"},
    {"mrn": 39111396, "initials": "KS", "folder": "Downs-Canner/SDC_KS_invasive"},
    {"mrn": 39102467, "initials": "MC", "folder": "Goel/NG_MC_invasive"},
    {"mrn": 39091629, "initials": "RW", "folder": "Tadros/AT_RW_INV"}
]

def detect_blur(image_array, threshold=100.0):
    """Detect if image is blurry using Laplacian variance."""
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold, laplacian_var

def pdf_to_base64_images(pdf_path):
    """Convert PDF to base64 encoded images."""
    images = convert_from_path(pdf_path, dpi=300)
    base64_images = []
    blur_detected = []
    
    for img in images:
        img_array = np.array(img)
        is_blurry, blur_score = detect_blur(img_array)
        blur_detected.append(is_blurry)
        
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        base64_images.append(img_base64)
    
    return base64_images, blur_detected

def extract_text_from_pdf_with_vision(pdf_path):
    """Extract text from PDF using Claude Sonnet 4.5 vision API."""
    try:
        base64_images, blur_detected = pdf_to_base64_images(pdf_path)
        
        content = []
        for img_b64 in base64_images:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": img_b64
                }
            })
        
        content.append({
            "type": "text",
            "text": "Extract all text from these medical document images. Preserve formatting, structure, and all details exactly as shown."
        })
        
        response = client.messages.create(
            model=MODEL,
            max_tokens=8192,
            messages=[{"role": "user", "content": content}]
        )
        
        extracted_text = response.content[0].text
        
        return {
            "success": True,
            "text": extracted_text,
            "blur_detected": any(blur_detected),
            "page_count": len(base64_images),
            "error": None
        }
    
    except Exception as e:
        return {
            "success": False,
            "text": "",
            "blur_detected": False,
            "page_count": 0,
            "error": str(e)
        }

def generate_summary_with_v2_prompt(combined_text, v2_prompt):
    """Generate summary using v2 prompt."""
    try:
        user_prompt = f"""Please extract structured clinical information from the following source documents.

SOURCE DOCUMENTS:
{combined_text}

Extract and structure the information as JSON following the schema provided in the system prompt."""
        
        response = client.messages.create(
            model=MODEL,
            max_tokens=8192,
            system=v2_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )
        
        summary_text = response.content[0].text
        
        return {
            "success": True,
            "summary": summary_text,
            "error": None
        }
    
    except Exception as e:
        return {
            "success": False,
            "summary": "",
            "error": str(e)
        }

def process_patient(patient_info, v2_prompt):
    """Process a single patient."""
    mrn = patient_info["mrn"]
    initials = patient_info["initials"]
    folder_name = patient_info["folder"]
    
    print(f"\n{'=' * 70}")
    print(f"Patient: {initials} (MRN: {mrn})")
    print(f"Folder: {folder_name}")
    
    patient_folder = RAW_DIR / folder_name
    
    if not patient_folder.exists():
        print(f"  ERROR: Folder not found: {patient_folder}")
        return None
    
    pdf_files = list(patient_folder.glob("*.pdf"))
    print(f"Documents: {len(pdf_files)}")
    
    if not pdf_files:
        print("  ERROR: No PDF files found")
        return None
    
    extraction_log = []
    combined_text = ""
    blur_detected_any = False
    
    for pdf_file in pdf_files:
        print(f"  Extracting: {pdf_file.name}")
        result = extract_text_from_pdf_with_vision(pdf_file)
        
        extraction_log.append({
            "document": pdf_file.name,
            "success": result["success"],
            "blur_detected": result["blur_detected"],
            "page_count": result["page_count"],
            "error": result["error"]
        })
        
        if result["success"]:
            combined_text += f"\n\n=== {pdf_file.name} ===\n{result['text']}"
            if result["blur_detected"]:
                blur_detected_any = True
        else:
            print(f"    ERROR: {result['error']}")
    
    if not combined_text:
        print("  ERROR: No text extracted from any document")
        return {
            "mrn": mrn,
            "patient_initials": initials,
            "folder_name": folder_name,
            "num_documents": len(pdf_files),
            "extraction_log": extraction_log,
            "summary_success": False,
            "summary": "",
            "summary_error": "No text extracted from any document",
            "blur_detected_any": blur_detected_any
        }
    
    print("  Generating summary with v2 prompt...")
    summary_result = generate_summary_with_v2_prompt(combined_text, v2_prompt)
    
    if not summary_result["success"]:
        print(f"    ERROR: {summary_result['error']}")
    
    return {
        "mrn": mrn,
        "patient_initials": initials,
        "folder_name": folder_name,
        "num_documents": len(pdf_files),
        "extraction_log": extraction_log,
        "summary_success": summary_result["success"],
        "summary": summary_result["summary"],
        "summary_error": summary_result["error"],
        "blur_detected_any": blur_detected_any
    }

def main():
    print("=" * 70)
    print("REPROCESSING FAILED V2 PROMPT TEST PATIENTS")
    print("=" * 70)
    
    with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
        v2_prompt = f.read()
    
    print(f"\nReprocessing {len(PATIENTS_TO_REPROCESS)} patients:")
    for p in PATIENTS_TO_REPROCESS:
        print(f"  - {p['initials']} (MRN: {p['mrn']})")
    
    results = []
    
    for patient_info in PATIENTS_TO_REPROCESS:
        result = process_patient(patient_info, v2_prompt)
        if result:
            results.append(result)
            
            initials = result['patient_initials']
            mrn = result['mrn']
            output_file = OUTPUT_DIR / f"{initials}_{mrn}_v2_summary.json"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
            
            print(f"  Saved: {output_file.name}")
    
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total patients reprocessed: {len(results)}")
    print(f"Summaries generated successfully: {sum(r['summary_success'] for r in results)}")
    print(f"Patients with blur detected: {sum(r['blur_detected_any'] for r in results)}")

if __name__ == "__main__":
    main()
