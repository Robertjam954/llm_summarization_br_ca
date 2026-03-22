"""
Apply text-based deidentification to v2 prompt summaries using robust-deid.

This script:
1. Loads the generated v2 summaries
2. Applies transformer-based deidentification using the ehr_deidentification package
3. Saves deidentified summaries for validation

Uses OBI-RoBERTa model trained on i2b2 dataset for medical note deidentification.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import json
from typing import List, Dict
from tqdm import tqdm

load_dotenv(override=True)

# Add ehr_deidentification to path
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT",
    r"C:\Users\jamesr4\OneDrive - Memorial Sloan Kettering Cancer Center\Documents\GitHub\llm_summarization_br_ca"))
sys.path.insert(0, str(PROJECT_ROOT / "ehr_deidentification" / "src"))

from robust_deid.ner_datasets import DatasetFormatter
from robust_deid.sequence_tagging import SequenceTagger
from robust_deid.deid import TextDeid

# Paths
DATA_PRIVATE_DIR = Path(os.getenv("DATA_PRIVATE_DIR", r"C:\Users\jamesr4\loc\data_private"))
V2_TEST_DIR = DATA_PRIVATE_DIR / "v2_prompt_test"
DEID_OUTPUT_DIR = DATA_PRIVATE_DIR / "v2_prompt_test_deidentified"
DEID_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model from huggingface
MODEL_NAME = "obi/deid_roberta_i2b2"

def load_v2_summaries() -> List[Dict]:
    """Load all v2 summary JSON files."""
    summaries = []
    for json_file in V2_TEST_DIR.glob("*_v2_summary.json"):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            summaries.append(data)
    return summaries

def prepare_text_for_deid(summary_text: str, note_id: str) -> Dict:
    """
    Prepare summary text in format required by robust-deid.
    
    Format:
    {
        "text": "...",
        "meta": {"note_id": "..."},
        "spans": []
    }
    """
    return {
        "text": summary_text,
        "meta": {"note_id": note_id},
        "spans": []
    }

def deidentify_summaries(summaries: List[Dict]) -> List[Dict]:
    """
    Apply text-based deidentification to summaries using robust-deid.
    
    Args:
        summaries: List of summary dicts with 'summary' field
        
    Returns:
        List of summaries with deidentified text
    """
    print("\nInitializing deidentification model...")
    print(f"Loading model: {MODEL_NAME}")
    
    # Initialize the deidentification pipeline
    tagger = SequenceTagger(
        model=MODEL_NAME,
        device='cpu'  # Use GPU if available by changing to 'cuda'
    )
    
    deid = TextDeid(
        notation='BIO',
        span_constraint='super_strict'
    )
    
    print("Model loaded successfully")
    
    deidentified_summaries = []
    
    for summary_data in tqdm(summaries, desc="Deidentifying summaries"):
        mrn = summary_data['mrn']
        patient_initials = summary_data['patient_initials']
        
        if not summary_data['summary_success']:
            # Skip failed summaries
            deidentified_summaries.append({
                **summary_data,
                'deidentified_summary': '',
                'deid_success': False,
                'deid_error': 'Original summary generation failed'
            })
            continue
        
        try:
            # Prepare text for deidentification
            note_id = f"{patient_initials}_{mrn}"
            deid_input = prepare_text_for_deid(
                summary_data['summary'], 
                note_id
            )
            
            # Run deidentification
            predictions = tagger([deid_input])
            deid_output = deid([deid_input], predictions=predictions)
            
            # Extract deidentified text
            deidentified_text = deid_output[0]['deid_text']
            phi_spans = predictions[0]
            
            deidentified_summaries.append({
                **summary_data,
                'deidentified_summary': deidentified_text,
                'deid_success': True,
                'deid_error': None,
                'phi_spans_detected': len(phi_spans),
                'phi_spans': phi_spans
            })
            
        except Exception as e:
            deidentified_summaries.append({
                **summary_data,
                'deidentified_summary': '',
                'deid_success': False,
                'deid_error': str(e)
            })
    
    return deidentified_summaries

def main():
    print("=" * 70)
    print("TEXT-BASED DEIDENTIFICATION FOR V2 SUMMARIES")
    print("Using robust-deid with OBI-RoBERTa model")
    print("=" * 70)
    
    # Load v2 summaries
    print(f"\nLoading v2 summaries from: {V2_TEST_DIR}")
    summaries = load_v2_summaries()
    print(f"Loaded {len(summaries)} summaries")
    
    if not summaries:
        print("No summaries found. Please run v2_prompt_fabrication_test.py first.")
        return
    
    # Apply deidentification
    deidentified_summaries = deidentify_summaries(summaries)
    
    # Save results
    print(f"\nSaving deidentified summaries to: {DEID_OUTPUT_DIR}")
    
    for deid_summary in deidentified_summaries:
        patient_initials = deid_summary['patient_initials']
        mrn = deid_summary['mrn']
        
        # Save individual deidentified summary
        output_file = DEID_OUTPUT_DIR / f"{patient_initials}_{mrn}_deidentified.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(deid_summary, f, indent=2)
    
    # Save summary CSV
    summary_df = pd.DataFrame([{
        'mrn': s['mrn'],
        'patient_initials': s['patient_initials'],
        'summary_success': s['summary_success'],
        'deid_success': s.get('deid_success', False),
        'phi_spans_detected': s.get('phi_spans_detected', 0),
        'deid_error': s.get('deid_error', '')
    } for s in deidentified_summaries])
    
    summary_df.to_csv(DEID_OUTPUT_DIR / "deidentification_summary.csv", index=False)
    
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total summaries processed: {len(deidentified_summaries)}")
    print(f"Successfully deidentified: {sum(s.get('deid_success', False) for s in deidentified_summaries)}")
    print(f"Total PHI spans detected: {sum(s.get('phi_spans_detected', 0) for s in deidentified_summaries)}")
    print(f"\nResults saved to: {DEID_OUTPUT_DIR}")

if __name__ == "__main__":
    main()
