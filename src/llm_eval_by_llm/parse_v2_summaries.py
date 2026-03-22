"""
Parse v2 prompt summaries to extract structured features.

The summaries are stored as JSON strings within the wrapper JSON.
This script parses them into proper structured format for validation.
"""

import os
import json
import re
from pathlib import Path
import pandas as pd

# Paths
DATA_PRIVATE_DIR = Path(r"C:\Users\jamesr4\loc\data_private")
V2_TEST_DIR = DATA_PRIVATE_DIR / "v2_prompt_test"
PARSED_OUTPUT_DIR = DATA_PRIVATE_DIR / "v2_prompt_test_parsed"
PARSED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_json_from_summary(summary_string: str) -> dict:
    """
    Extract JSON from summary string.
    The summary may be wrapped in ```json ... ``` markdown code blocks.
    """
    # Remove markdown code blocks if present
    summary_string = summary_string.strip()
    if summary_string.startswith("```json"):
        summary_string = summary_string[7:]  # Remove ```json
    if summary_string.startswith("```"):
        summary_string = summary_string[3:]  # Remove ```
    if summary_string.endswith("```"):
        summary_string = summary_string[:-3]  # Remove trailing ```
    
    summary_string = summary_string.strip()
    
    # Parse JSON
    try:
        return json.loads(summary_string)
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        print(f"String preview: {summary_string[:200]}")
        return None

def parse_v2_summaries():
    """Parse all v2 summary JSON files."""
    
    summary_files = list(V2_TEST_DIR.glob("*_v2_summary.json"))
    print(f"Found {len(summary_files)} summary files")
    
    parsed_summaries = []
    
    for summary_file in summary_files:
        print(f"\nProcessing: {summary_file.name}")
        
        with open(summary_file, 'r', encoding='utf-8') as f:
            wrapper = json.load(f)
        
        mrn = wrapper['mrn']
        patient_initials = wrapper['patient_initials']
        summary_success = wrapper['summary_success']
        
        if not summary_success:
            print(f"  Skipping - summary generation failed")
            continue
        
        # Extract and parse the summary string
        summary_string = wrapper['summary']
        parsed_summary = extract_json_from_summary(summary_string)
        
        if parsed_summary is None:
            print(f"  ERROR: Could not parse summary JSON")
            continue
        
        # Create parsed output
        parsed_output = {
            'mrn': mrn,
            'patient_initials': patient_initials,
            'folder_name': wrapper['folder_name'],
            'num_documents': wrapper['num_documents'],
            'extraction_log': wrapper['extraction_log'],
            'parsed_summary': parsed_summary
        }
        
        # Save individual parsed summary
        output_file = PARSED_OUTPUT_DIR / f"{patient_initials}_{mrn}_parsed.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(parsed_output, f, indent=2)
        
        print(f"  Saved: {output_file.name}")
        
        # Extract features for CSV
        if 'lesions' in parsed_summary and len(parsed_summary['lesions']) > 0:
            lesion = parsed_summary['lesions'][0]  # Use first lesion
            
            feature_row = {
                'mrn': mrn,
                'patient_initials': patient_initials,
                'lesion_id': lesion.get('lesion_id', 'L1'),
                'feature_1_lesion_size': lesion.get('feature_1_lesion_size', {}).get('value', ''),
                'feature_2_lesion_location': lesion.get('feature_2_lesion_location', {}).get('value', ''),
                'feature_3_calcifications_asymmetry': lesion.get('feature_3_calcifications_asymmetry', {}).get('value', ''),
                'feature_4_additional_enhancement_mri': lesion.get('feature_4_additional_enhancement_mri', {}).get('value', ''),
                'feature_5_extent': lesion.get('feature_5_extent', {}).get('value', ''),
                'feature_6_accurate_clip_placement': lesion.get('feature_6_accurate_clip_placement', {}).get('value', ''),
                'feature_7_workup_recommendation': lesion.get('feature_7_workup_recommendation', {}).get('value', ''),
                'feature_8_lymph_node': lesion.get('feature_8_lymph_node', {}).get('value', ''),
                'feature_10_biopsy_method': lesion.get('feature_10_biopsy_method', {}).get('value', ''),
                'feature_11_invasive_component_size_pathology': lesion.get('feature_11_invasive_component_size_pathology', {}).get('value', ''),
                'feature_12_histologic_diagnosis': lesion.get('feature_12_histologic_diagnosis', {}).get('value', ''),
                'feature_9_chronology_preserved': parsed_summary.get('feature_9_chronology_preserved', '')
            }
            
            # Receptor status is nested
            receptor = lesion.get('feature_13_receptor_status', {})
            feature_row['feature_13_receptor_ER'] = receptor.get('ER', {}).get('value', '')
            feature_row['feature_13_receptor_PR'] = receptor.get('PR', {}).get('value', '')
            feature_row['feature_13_receptor_HER2_IHC'] = receptor.get('HER2_IHC', {}).get('value', '')
            feature_row['feature_13_receptor_HER2_ISH'] = receptor.get('HER2_ISH', {}).get('value', '')
            
            parsed_summaries.append(feature_row)
    
    # Save features CSV
    if parsed_summaries:
        features_df = pd.DataFrame(parsed_summaries)
        features_csv = PARSED_OUTPUT_DIR / "v2_summaries_features_extracted.csv"
        features_df.to_csv(features_csv, index=False)
        print(f"\nSaved features CSV: {features_csv}")
        print(f"  {len(features_df)} patients with extracted features")
    
    print(f"\nParsing complete!")
    print(f"  Parsed summaries saved to: {PARSED_OUTPUT_DIR}")

if __name__ == "__main__":
    parse_v2_summaries()
