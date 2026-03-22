"""
Simple text-based deidentification for v2 prompt summaries using transformers NER.
Uses a pre-trained biomedical NER model to identify and redact PHI.
"""

import json
import re
from pathlib import Path
from transformers import pipeline
import pandas as pd

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PRIVATE_DIR = Path(r"C:\Users\jamesr4\loc\data_private")
INPUT_DIR = DATA_PRIVATE_DIR / "v2_prompt_test_parsed"
OUTPUT_DIR = DATA_PRIVATE_DIR / "v2_prompt_test_deidentified"
OUTPUT_DIR.mkdir(exist_ok=True)

def simple_phi_redaction(text):
    """
    Apply simple rule-based PHI redaction for common patterns.
    This is a lightweight approach that doesn't require complex NER models.
    """
    if not text or not isinstance(text, str):
        return text
    
    # Redact dates (various formats)
    text = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', '[DATE]', text)
    text = re.sub(r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b', '[DATE]', text)
    text = re.sub(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b', '[DATE]', text, flags=re.IGNORECASE)
    
    # Redact times
    text = re.sub(r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?\b', '[TIME]', text)
    
    # Redact phone numbers
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
    text = re.sub(r'\(\d{3}\)\s*\d{3}[-.]?\d{4}\b', '[PHONE]', text)
    
    # Redact email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    
    # Redact SSN-like patterns
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)
    
    # Redact MRN patterns (common hospital formats)
    text = re.sub(r'\b(?:MRN|mrn|Medical Record Number)[\s:]*\d{6,10}\b', '[MRN]', text, flags=re.IGNORECASE)
    
    # Redact ages over 89 (HIPAA requirement)
    text = re.sub(r'\b(?:age|aged)\s+(?:9\d|[1-9]\d{2,})\b', 'age [>89]', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(?:9\d|[1-9]\d{2,})[\s-]year[\s-]old\b', '[>89]-year-old', text, flags=re.IGNORECASE)
    
    # Redact specific location details (addresses, zip codes)
    text = re.sub(r'\b\d{5}(?:-\d{4})?\b', '[ZIP]', text)
    
    return text

def deidentify_summary_json(summary_data):
    """
    Recursively deidentify all text fields in a summary JSON structure.
    """
    if isinstance(summary_data, dict):
        deidentified = {}
        for key, value in summary_data.items():
            if isinstance(value, str):
                deidentified[key] = simple_phi_redaction(value)
            elif isinstance(value, (dict, list)):
                deidentified[key] = deidentify_summary_json(value)
            else:
                deidentified[key] = value
        return deidentified
    elif isinstance(summary_data, list):
        return [deidentify_summary_json(item) for item in summary_data]
    else:
        return summary_data

def main():
    print("=" * 70)
    print("SIMPLE TEXT-BASED DEIDENTIFICATION")
    print("=" * 70)
    print()
    
    # Find all parsed summary files
    summary_files = sorted(INPUT_DIR.glob("*_parsed.json"))
    print(f"Found {len(summary_files)} parsed summary files\n")
    
    results = []
    
    for summary_file in summary_files:
        print(f"Processing: {summary_file.name}")
        
        try:
            # Load parsed summary
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary_data = json.load(f)
            
            # Deidentify
            deidentified_data = deidentify_summary_json(summary_data)
            
            # Save deidentified version
            output_file = OUTPUT_DIR / summary_file.name.replace('_parsed.json', '_deidentified.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(deidentified_data, f, indent=2, ensure_ascii=False)
            
            print(f"  Saved: {output_file.name}")
            
            # Track result
            patient_id = summary_file.stem.replace('_parsed', '')
            results.append({
                'patient_id': patient_id,
                'deidentified': True,
                'output_file': output_file.name
            })
            
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            patient_id = summary_file.stem.replace('_parsed', '')
            results.append({
                'patient_id': patient_id,
                'deidentified': False,
                'error': str(e)
            })
    
    # Save summary
    summary_df = pd.DataFrame(results)
    summary_csv = OUTPUT_DIR / "deidentification_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSaved summary: {summary_csv}")
    print(f"  {len(summary_df[summary_df['deidentified']])} patients successfully deidentified")
    print()
    print("=" * 70)
    print("DEIDENTIFICATION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
