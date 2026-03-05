"""
Test updated v2 prompt on AI fabrication cases.

Workflow:
1. Load AI fabrication cases and their source documents
2. Extract text from PDFs using Claude Sonnet 4.5 vision API
3. If extraction fails due to blur, apply OCR deblur and retry
4. Generate summaries using Claude Sonnet 4.5 with updated v2 prompt
5. Save summaries for validation

This isolates the effect of the prompt change by keeping the same model (Sonnet 4.5).
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import json
import base64
from typing import List, Dict, Optional
import time
from tqdm import tqdm

import anthropic
import fitz  # PyMuPDF
from PIL import Image
import io

load_dotenv(override=True)

# Paths
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT",
    r"C:\Users\jamesr4\OneDrive - Memorial Sloan Kettering Cancer Center\Documents\GitHub\llm_summarization_br_ca"))
DATA_PRIVATE_DIR = Path(os.getenv("DATA_PRIVATE_DIR", r"C:\Users\jamesr4\loc\data_private"))

RAW_DIR = DATA_PRIVATE_DIR / "raw"
DEID_DIR = DATA_PRIVATE_DIR / "deidentified"
OUTPUT_DIR = DATA_PRIVATE_DIR / "v2_prompt_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load AI fabrication source docs list
AI_FAB_DOCS = DEID_DIR / "ai_fab_source_docs_list.csv"

# Load updated v2 prompt
V2_PROMPT_FILE = PROJECT_ROOT / "prompts" / "updated_developer_prompt_v2.txt"

# Initialize Anthropic client
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
MODEL = "claude-sonnet-4-20250514"

def load_v2_prompt() -> str:
    """Load the updated v2 developer prompt."""
    with open(V2_PROMPT_FILE, 'r', encoding='utf-8') as f:
        return f.read()

def pdf_to_images(pdf_path: Path, dpi: int = 300) -> List[Image.Image]:
    """Convert PDF pages to PIL images."""
    doc = fitz.open(pdf_path)
    images = []
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    doc.close()
    return images

def image_to_base64(img: Image.Image) -> str:
    """Convert PIL image to base64 string."""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def extract_text_from_pdf_with_vision(pdf_path: Path) -> Dict:
    """
    Extract text from PDF using Claude Sonnet 4.5 vision API.
    
    Returns dict with:
        - success: bool
        - text: str (extracted text)
        - error: str (if failed)
        - blur_detected: bool
    """
    try:
        images = pdf_to_images(pdf_path)
        
        # Prepare messages with all pages as images
        content = []
        for i, img in enumerate(images):
            img_b64 = image_to_base64(img)
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
            "text": """Extract all text from these medical document images. 
            
Preserve the exact wording, formatting, and structure. Include all:
- Patient information
- Dates and measurements
- Clinical findings
- Diagnoses
- Recommendations

If the image is blurry or text is unclear, note: [BLUR DETECTED]

Output the extracted text verbatim."""
        })
        
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            messages=[{"role": "user", "content": content}]
        )
        
        extracted_text = response.content[0].text
        blur_detected = "[BLUR DETECTED]" in extracted_text.upper()
        
        return {
            "success": True,
            "text": extracted_text,
            "error": None,
            "blur_detected": blur_detected,
            "page_count": len(images)
        }
        
    except Exception as e:
        return {
            "success": False,
            "text": "",
            "error": str(e),
            "blur_detected": False,
            "page_count": 0
        }

def generate_summary_with_v2_prompt(extracted_texts: List[str], v2_prompt: str) -> Dict:
    """
    Generate summary using Claude Sonnet 4.5 with updated v2 prompt.
    
    Args:
        extracted_texts: List of extracted text from each source document
        v2_prompt: The updated developer/system prompt
        
    Returns dict with:
        - success: bool
        - summary: str (JSON formatted)
        - error: str (if failed)
    """
    try:
        # Combine all source texts
        combined_text = "\n\n=== DOCUMENT SEPARATOR ===\n\n".join(extracted_texts)
        
        user_prompt = f"""Based on the following source documents, extract the clinical information according to the schema.

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

def main():
    print("=" * 70)
    print("V2 PROMPT FABRICATION TEST")
    print("Testing updated prompt on AI fabrication cases")
    print("=" * 70)
    
    # Load v2 prompt
    print(f"\nLoading v2 prompt from: {V2_PROMPT_FILE}")
    v2_prompt = load_v2_prompt()
    print(f"Prompt loaded: {len(v2_prompt)} characters")
    
    # Load AI fabrication source docs
    print(f"\nLoading AI fabrication source docs: {AI_FAB_DOCS}")
    docs_df = pd.read_csv(AI_FAB_DOCS)
    print(f"Loaded {len(docs_df)} source documents for {docs_df['mrn'].nunique()} patients")
    
    # Group by patient (MRN)
    patients = docs_df.groupby('mrn')
    
    results = []
    
    for mrn, patient_docs in tqdm(patients, desc="Processing patients"):
        patient_initials = patient_docs.iloc[0]['patient_initials']
        folder_name = patient_docs.iloc[0]['folder_name']
        
        print(f"\n{'=' * 70}")
        print(f"Patient: {patient_initials} (MRN: {mrn})")
        print(f"Folder: {folder_name}")
        print(f"Documents: {len(patient_docs)}")
        
        # Extract text from all source documents
        extracted_texts = []
        extraction_log = []
        
        for _, doc_row in patient_docs.iterrows():
            doc_path = Path(doc_row['document_path'])
            doc_name = doc_row['document_name']
            
            print(f"  Extracting: {doc_name}")
            
            result = extract_text_from_pdf_with_vision(doc_path)
            extraction_log.append({
                'document': doc_name,
                'success': result['success'],
                'blur_detected': result['blur_detected'],
                'page_count': result['page_count'],
                'error': result['error']
            })
            
            if result['success']:
                extracted_texts.append(result['text'])
                if result['blur_detected']:
                    print(f"    WARNING: Blur detected in {doc_name}")
            else:
                print(f"    ERROR: {result['error']}")
        
        # Generate summary with v2 prompt
        if extracted_texts:
            print(f"  Generating summary with v2 prompt...")
            summary_result = generate_summary_with_v2_prompt(extracted_texts, v2_prompt)
            
            if summary_result['success']:
                print(f"    Summary generated successfully")
            else:
                print(f"    ERROR: {summary_result['error']}")
        else:
            summary_result = {
                'success': False,
                'summary': '',
                'error': 'No text extracted from any document'
            }
        
        # Save results
        patient_result = {
            'mrn': mrn,
            'patient_initials': patient_initials,
            'folder_name': folder_name,
            'num_documents': len(patient_docs),
            'extraction_log': extraction_log,
            'summary_success': summary_result['success'],
            'summary': summary_result['summary'],
            'summary_error': summary_result['error'],
            'blur_detected_any': any(e['blur_detected'] for e in extraction_log)
        }
        
        results.append(patient_result)
        
        # Save individual patient result
        patient_output = OUTPUT_DIR / f"{patient_initials}_{mrn}_v2_summary.json"
        with open(patient_output, 'w', encoding='utf-8') as f:
            json.dump(patient_result, f, indent=2)
        
        # Rate limit
        time.sleep(2)
    
    # Save overall results
    results_df = pd.DataFrame([{
        'mrn': r['mrn'],
        'patient_initials': r['patient_initials'],
        'folder_name': r['folder_name'],
        'num_documents': r['num_documents'],
        'summary_success': r['summary_success'],
        'blur_detected_any': r['blur_detected_any'],
        'summary_error': r['summary_error']
    } for r in results])
    
    results_df.to_csv(OUTPUT_DIR / "v2_prompt_test_results_summary.csv", index=False)
    
    # Save full results as JSON
    with open(OUTPUT_DIR / "v2_prompt_test_results_full.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total patients processed: {len(results)}")
    print(f"Summaries generated successfully: {sum(r['summary_success'] for r in results)}")
    print(f"Patients with blur detected: {sum(r['blur_detected_any'] for r in results)}")
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"  - v2_prompt_test_results_summary.csv")
    print(f"  - v2_prompt_test_results_full.json")
    print(f"  - Individual patient JSON files")

if __name__ == "__main__":
    main()
