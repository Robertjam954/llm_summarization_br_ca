"""
Identify source documents for AI fabrications patients.

This script:
1. Loads ai_fabrications_dataset.xlsx to get MRNs
2. Maps MRNs to patient folders using patient_folder_mapping.csv
3. Extracts list of source documents for those patients
4. Creates a file list for deidentification processing
"""

import pandas as pd
from pathlib import Path

# File paths
DATA_PRIVATE_DIR = Path(r"C:\Users\jamesr4\loc\data_private")
RAW_DIR = DATA_PRIVATE_DIR / "raw"
DEID_DIR = DATA_PRIVATE_DIR / "deidentified"

AI_FAB_FILE = RAW_DIR / "ai_fabrications_dataset.xlsx"
PATIENT_MAPPING_FILE = DEID_DIR / "patient_folder_mapping.csv"
OUTPUT_FILE = DEID_DIR / "ai_fab_source_docs_list.csv"

def main():
    print("=" * 70)
    print("IDENTIFYING SOURCE DOCS FOR AI FABRICATIONS PATIENTS")
    print("=" * 70)
    
    # Load AI fabrications dataset
    print(f"\nLoading AI fabrications dataset: {AI_FAB_FILE}")
    ai_fab_df = pd.read_excel(AI_FAB_FILE)
    print(f"Loaded {len(ai_fab_df)} AI fabrication cases")
    
    # Get unique patient initials
    ai_fab_patients = ai_fab_df[['mrn', 'patient_initials', 'surgeon']].drop_duplicates()
    print(f"\nUnique patients: {len(ai_fab_patients)}")
    print(ai_fab_patients[['patient_initials', 'surgeon', 'mrn']].to_string())
    
    # Load patient folder mapping
    print(f"\nLoading patient folder mapping: {PATIENT_MAPPING_FILE}")
    patient_mapping_df = pd.read_csv(PATIENT_MAPPING_FILE)
    print(f"Loaded {len(patient_mapping_df)} patient folders")
    
    # Map AI fabrications patients to their folders using patient_initials
    print("\nMapping AI fabrications patients to folders...")
    
    # Merge on patient_initials
    ai_fab_with_folders = ai_fab_patients.merge(
        patient_mapping_df,
        on='patient_initials',
        how='left'
    )
    
    print(f"\nMatched {len(ai_fab_with_folders[ai_fab_with_folders['folder_name'].notna()])} patients to folders")
    
    # Check for unmatched patients
    unmatched = ai_fab_with_folders[ai_fab_with_folders['folder_name'].isna()]
    if len(unmatched) > 0:
        print(f"\nWARNING: {len(unmatched)} patients not matched:")
        print(unmatched[['patient_initials', 'surgeon', 'mrn']].to_string())
    
    # Create list of all source documents
    matched_patients = ai_fab_with_folders[ai_fab_with_folders['folder_name'].notna()]
    
    print("\n" + "=" * 70)
    print("AI FABRICATIONS PATIENTS AND THEIR SOURCE DOCUMENTS")
    print("=" * 70)
    
    all_source_docs = []
    
    for _, row in matched_patients.iterrows():
        print(f"\nPatient: {row['patient_initials']} (MRN: {row['mrn']})")
        print(f"  Folder: {row['folder_name']}")
        print(f"  Path: {row['folder_path']}")
        print(f"  Documents ({row['num_source_docs']}):")
        
        source_docs = row['source_documents'].split('|')
        for doc in source_docs:
            doc_path = Path(row['folder_path']) / doc
            print(f"    - {doc}")
            all_source_docs.append({
                'mrn': row['mrn'],
                'patient_initials': row['patient_initials'],
                'folder_name': row['folder_name'],
                'folder_path': row['folder_path'],
                'document_name': doc,
                'document_path': str(doc_path),
                'case_ids': row['case_ids']
            })
    
    # Save to CSV
    print(f"\n{'=' * 70}")
    print(f"Total source documents: {len(all_source_docs)}")
    
    output_df = pd.DataFrame(all_source_docs)
    output_df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\nSaved source documents list to: {OUTPUT_FILE}")
    print(f"\nSummary:")
    print(f"  - {len(matched_patients)} patients")
    print(f"  - {len(all_source_docs)} source documents")
    print(f"\nReady for deidentification processing!")

if __name__ == "__main__":
    main()
