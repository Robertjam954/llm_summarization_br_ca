"""
Filter validation datasheet by AI and Human fabrications from classification file.

This script:
1. Loads MRNs from "Fabricated by AI" sheet (sheet 1) and "Fabrication by Human" sheet (sheet 2)
2. Filters the validation datasheet by those MRNs
3. Saves two separate datasets: ai_fabrications_dataset.xlsx and human_fabrications_dataset.xlsx
"""

import pandas as pd
from pathlib import Path

# File paths
DATA_PRIVATE_DIR = Path(r"C:\Users\jamesr4\loc\data_private")
RAW_DIR = DATA_PRIVATE_DIR / "raw"

VALIDATION_FILE = RAW_DIR / "merged_llm_summary_validation_datasheet.xlsx"
CLASSIFICATIONS_FILE = RAW_DIR / "Fabrications_classified.xlsx"

OUTPUT_AI_FILE = RAW_DIR / "ai_fabrications_dataset.xlsx"
OUTPUT_HUMAN_FILE = RAW_DIR / "human_fabrications_dataset.xlsx"

def main():
    print("Loading validation datasheet...")
    validation_df = pd.read_excel(VALIDATION_FILE)
    print(f"Loaded {len(validation_df)} rows from validation datasheet")
    
    # Check if MRN column exists
    mrn_col = None
    for col in validation_df.columns:
        if 'mrn' in col.lower():
            mrn_col = col
            break
    
    if mrn_col is None:
        print("Available columns:", validation_df.columns.tolist())
        raise ValueError("No MRN column found in validation datasheet")
    
    print(f"Using MRN column: '{mrn_col}'")
    
    # Load AI fabrications (sheet 1)
    print("\nLoading AI fabrications (sheet 1: 'Fabricated by AI')...")
    try:
        ai_fab_df = pd.read_excel(CLASSIFICATIONS_FILE, sheet_name="Fabricated by AI")
    except:
        # Try with index 0 if sheet name doesn't work
        ai_fab_df = pd.read_excel(CLASSIFICATIONS_FILE, sheet_name=0)
    
    print(f"AI fabrications sheet has {len(ai_fab_df)} rows")
    print(f"Columns: {ai_fab_df.columns.tolist()}")
    
    # Find MRN column in AI fabrications
    ai_mrn_col = None
    for col in ai_fab_df.columns:
        if 'mrn' in col.lower():
            ai_mrn_col = col
            break
    
    if ai_mrn_col is None:
        raise ValueError("No MRN column found in AI fabrications sheet")
    
    ai_mrns = set(ai_fab_df[ai_mrn_col].dropna().astype(str))
    print(f"Found {len(ai_mrns)} unique MRNs in AI fabrications")
    
    # Load Human fabrications (sheet 2)
    print("\nLoading Human fabrications (sheet 2: 'Fabrication by Human')...")
    try:
        human_fab_df = pd.read_excel(CLASSIFICATIONS_FILE, sheet_name="Fabrication by Human")
    except:
        # Try with index 1 if sheet name doesn't work
        human_fab_df = pd.read_excel(CLASSIFICATIONS_FILE, sheet_name=1)
    
    print(f"Human fabrications sheet has {len(human_fab_df)} rows")
    print(f"Columns: {human_fab_df.columns.tolist()}")
    
    # Find MRN column in Human fabrications
    human_mrn_col = None
    for col in human_fab_df.columns:
        if 'mrn' in col.lower():
            human_mrn_col = col
            break
    
    if human_mrn_col is None:
        raise ValueError("No MRN column found in Human fabrications sheet")
    
    human_mrns = set(human_fab_df[human_mrn_col].dropna().astype(str))
    print(f"Found {len(human_mrns)} unique MRNs in Human fabrications")
    
    # Filter validation datasheet by AI fabrications
    print("\nFiltering validation datasheet by AI fabrications...")
    validation_df[mrn_col] = validation_df[mrn_col].astype(str)
    ai_fabrications_dataset = validation_df[validation_df[mrn_col].isin(ai_mrns)].copy()
    print(f"AI fabrications dataset: {len(ai_fabrications_dataset)} rows")
    
    # Filter validation datasheet by Human fabrications
    print("Filtering validation datasheet by Human fabrications...")
    human_fabrications_dataset = validation_df[validation_df[mrn_col].isin(human_mrns)].copy()
    print(f"Human fabrications dataset: {len(human_fabrications_dataset)} rows")
    
    # Save filtered datasets
    print(f"\nSaving AI fabrications to: {OUTPUT_AI_FILE}")
    ai_fabrications_dataset.to_excel(OUTPUT_AI_FILE, index=False)
    
    print(f"Saving Human fabrications to: {OUTPUT_HUMAN_FILE}")
    human_fabrications_dataset.to_excel(OUTPUT_HUMAN_FILE, index=False)
    
    print("\nDone! Created:")
    print(f"  - {OUTPUT_AI_FILE.name} ({len(ai_fabrications_dataset)} rows)")
    print(f"  - {OUTPUT_HUMAN_FILE.name} ({len(human_fabrications_dataset)} rows)")
    print(f"  - Original validation datasheet preserved ({len(validation_df)} rows)")

if __name__ == "__main__":
    main()
