"""
Map patients to their folder directories using patient initials.

This script:
1. Scans the raw folder structure (attending folders -> patient folders)
2. Extracts patient initials from folder names (format: <surgeon>_<patient>_inv/dcis)
3. Lists all source documents for each patient
4. Maps patients to case_ids from patient_case_id_mapping.csv
5. Creates comprehensive patient mapping file
"""

import pandas as pd
from pathlib import Path
import re

# File paths
DATA_PRIVATE_DIR = Path(r"C:\Users\jamesr4\loc\data_private")
RAW_DIR = DATA_PRIVATE_DIR / "raw"
DEID_DIR = DATA_PRIVATE_DIR / "deidentified"

CASE_MAPPING_FILE = DEID_DIR / "patient_case_id_mapping.csv"
OUTPUT_FILE = DEID_DIR / "patient_folder_mapping.csv"

def extract_patient_info(folder_name):
    """
    Extract surgeon initials, patient initials, and tumor type from folder name.
    Format: <surgeon>_<patient>_inv or <surgeon>_<patient>_dcis
    """
    pattern = r'^([A-Z]+)_([A-Z]+)_(inv|dcis|invasive)$'
    match = re.match(pattern, folder_name, re.IGNORECASE)
    
    if match:
        surgeon_initials = match.group(1)
        patient_initials = match.group(2)
        tumor_type = match.group(3).lower()
        
        # Normalize tumor type
        if tumor_type in ['inv', 'invasive']:
            tumor_type = 'invasive'
        
        return {
            'surgeon_initials': surgeon_initials,
            'patient_initials': patient_initials,
            'tumor_type': tumor_type,
            'folder_name': folder_name
        }
    return None

def main():
    print("=" * 70)
    print("MAPPING PATIENTS TO FOLDER DIRECTORIES")
    print("=" * 70)
    
    # Scan raw directory for attending folders
    print(f"\nScanning raw directory: {RAW_DIR}")
    
    patient_data = []
    
    # Iterate through attending folders
    for attending_folder in sorted(RAW_DIR.iterdir()):
        if not attending_folder.is_dir():
            continue
        
        attending_name = attending_folder.name
        print(f"\nProcessing attending: {attending_name}")
        
        # Iterate through patient folders within attending folder
        for patient_folder in sorted(attending_folder.iterdir()):
            if not patient_folder.is_dir():
                continue
            
            folder_name = patient_folder.name
            
            # Extract patient info from folder name
            patient_info = extract_patient_info(folder_name)
            
            if patient_info is None:
                print(f"  WARNING: Skipping folder (doesn't match pattern): {folder_name}")
                continue
            
            # List all source documents in this patient folder
            source_docs = []
            for file in sorted(patient_folder.glob("*.pdf")):
                source_docs.append(file.name)
            
            if not source_docs:
                print(f"  WARNING: No PDFs found in: {folder_name}")
            
            # Create patient record
            patient_record = {
                'attending_name': attending_name,
                'surgeon_initials': patient_info['surgeon_initials'],
                'patient_initials': patient_info['patient_initials'],
                'tumor_type': patient_info['tumor_type'],
                'folder_name': folder_name,
                'folder_path': str(patient_folder),
                'num_source_docs': len(source_docs),
                'source_documents': '|'.join(source_docs)
            }
            
            patient_data.append(patient_record)
            
            print(f"  OK: {folder_name}: {len(source_docs)} documents")
    
    # Create DataFrame
    print(f"\n{'=' * 70}")
    print(f"Found {len(patient_data)} patient folders")
    
    df = pd.DataFrame(patient_data)
    
    # Load case_id mapping
    print(f"\nLoading case_id mapping from: {CASE_MAPPING_FILE}")
    case_mapping_df = pd.read_csv(CASE_MAPPING_FILE)
    
    print(f"Loaded {len(case_mapping_df)} case_id mappings")
    
    # Extract patient initials from original_path in case mapping
    # Path format: .../Barrio/AB_CB_invasive/imaging_internal_read.pdf
    def extract_patient_from_path(path):
        """Extract patient folder name from path"""
        path_obj = Path(path)
        # Get parent directory name (patient folder)
        patient_folder = path_obj.parent.name
        return patient_folder
    
    case_mapping_df['folder_name'] = case_mapping_df['original_path'].apply(
        extract_patient_from_path
    )
    
    # Group by folder_name to get unique case_ids per patient
    case_groups = case_mapping_df.groupby('folder_name').agg({
        'case_id': lambda x: '|'.join(sorted(set(x))),
        'original_filename': 'count'
    }).reset_index()
    
    case_groups.columns = ['folder_name', 'case_ids', 'num_case_ids']
    
    print(f"\nGrouped into {len(case_groups)} unique patient folders")
    
    # Merge with patient data
    print("\nMerging patient data with case_ids...")
    df_merged = df.merge(
        case_groups[['folder_name', 'case_ids', 'num_case_ids']], 
        on='folder_name', 
        how='left'
    )
    
    # Check for patients without case_ids
    missing_case_ids = df_merged[df_merged['case_ids'].isna()]
    if len(missing_case_ids) > 0:
        print(f"\nWARNING: {len(missing_case_ids)} patients without case_ids:")
        for _, row in missing_case_ids.iterrows():
            print(f"  - {row['folder_name']}")
    
    # Save to file
    print(f"\nSaving patient folder mapping to: {OUTPUT_FILE}")
    df_merged.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total patients mapped: {len(df_merged)}")
    print(f"Patients with case_ids: {len(df_merged[df_merged['case_ids'].notna()])}")
    print(f"Total source documents: {df_merged['num_source_docs'].sum()}")
    print(f"\nOutput saved to: {OUTPUT_FILE}")
    
    # Display sample
    print(f"\nSample rows:")
    print(df_merged[['patient_initials', 'tumor_type', 'num_source_docs', 
                     'case_ids']].head(10).to_string())

if __name__ == "__main__":
    main()
