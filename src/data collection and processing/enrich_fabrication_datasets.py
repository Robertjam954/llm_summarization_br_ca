"""
Enrich fabrication datasets by joining classification columns from Fabrications_classified.xlsx

This script:
1. Joins AI fabrication columns (comment, critical_error, justification) from sheet 1 
   to ai_fabrications_dataset.xlsx by MRN
2. Joins Human fabrication columns (comment, critical_error) from sheet 2 
   to human_fabrications_dataset.xlsx by MRN
"""

import pandas as pd
from pathlib import Path

# File paths
DATA_PRIVATE_DIR = Path(r"C:\Users\jamesr4\loc\data_private")
RAW_DIR = DATA_PRIVATE_DIR / "raw"

AI_FAB_FILE = RAW_DIR / "ai_fabrications_dataset.xlsx"
HUMAN_FAB_FILE = RAW_DIR / "human_fabrications_dataset.xlsx"
CLASSIFICATIONS_FILE = RAW_DIR / "Fabrications_classified.xlsx"

def main():
    # ========== AI Fabrications Dataset ==========
    print("=" * 60)
    print("ENRICHING AI FABRICATIONS DATASET")
    print("=" * 60)
    
    print("\nLoading ai_fabrications_dataset.xlsx...")
    ai_fab_df = pd.read_excel(AI_FAB_FILE)
    print(f"Loaded {len(ai_fab_df)} rows")
    
    print("\nLoading sheet 1 ('Fabricated by AI') from classifications...")
    try:
        ai_class_df = pd.read_excel(
            CLASSIFICATIONS_FILE, 
            sheet_name="Fabricated by AI"
        )
    except:
        ai_class_df = pd.read_excel(CLASSIFICATIONS_FILE, sheet_name=0)
    
    print(f"Loaded {len(ai_class_df)} rows")
    print(f"Columns: {ai_class_df.columns.tolist()}")
    
    # Find column names (case-insensitive)
    col_map = {}
    for col in ai_class_df.columns:
        col_lower = col.lower()
        if 'mrn' in col_lower:
            col_map['mrn'] = col
        elif 'comment' in col_lower:
            col_map['comment'] = col
        elif 'critical' in col_lower and 'error' in col_lower:
            col_map['critical_error'] = col
        elif 'justification' in col_lower:
            col_map['justification'] = col
    
    print(f"\nMapped columns: {col_map}")
    
    # Select and rename columns for join
    ai_join_cols = ['mrn', 'comment', 'critical_error', 'justification']
    ai_class_subset = ai_class_df[[
        col_map['mrn'], 
        col_map['comment'], 
        col_map['critical_error'], 
        col_map['justification']
    ]].copy()
    
    ai_class_subset.columns = [
        'mrn', 
        'ai_fab_comment', 
        'ai_fab_critical_error', 
        'ai_fab_justification'
    ]
    
    # Convert MRN to string for join
    ai_fab_df['mrn'] = ai_fab_df['mrn'].astype(str)
    ai_class_subset['mrn'] = ai_class_subset['mrn'].astype(str)
    
    # Left join
    print("\nPerforming left join on 'mrn'...")
    ai_enriched = ai_fab_df.merge(
        ai_class_subset, 
        on='mrn', 
        how='left'
    )
    
    print(f"Enriched dataset: {len(ai_enriched)} rows")
    print(f"New columns added: ai_fab_comment, ai_fab_critical_error, "
          "ai_fab_justification")
    
    # Save enriched AI dataset
    print(f"\nSaving enriched AI dataset to: {AI_FAB_FILE}")
    ai_enriched.to_excel(AI_FAB_FILE, index=False)
    print("Saved!")
    
    # ========== Human Fabrications Dataset ==========
    print("\n" + "=" * 60)
    print("ENRICHING HUMAN FABRICATIONS DATASET")
    print("=" * 60)
    
    print("\nLoading human_fabrications_dataset.xlsx...")
    human_fab_df = pd.read_excel(HUMAN_FAB_FILE)
    print(f"Loaded {len(human_fab_df)} rows")
    
    print("\nLoading sheet 2 ('Fabrication by Human') from classifications...")
    try:
        human_class_df = pd.read_excel(
            CLASSIFICATIONS_FILE, 
            sheet_name="Fabrication by Human"
        )
    except:
        human_class_df = pd.read_excel(CLASSIFICATIONS_FILE, sheet_name=1)
    
    print(f"Loaded {len(human_class_df)} rows")
    print(f"Columns: {human_class_df.columns.tolist()}")
    
    # Find column names (case-insensitive)
    col_map_human = {}
    for col in human_class_df.columns:
        col_lower = col.lower()
        if 'mrn' in col_lower:
            col_map_human['mrn'] = col
        elif 'comment' in col_lower:
            col_map_human['comment'] = col
        elif 'critical' in col_lower and 'error' in col_lower:
            col_map_human['critical_error'] = col
    
    print(f"\nMapped columns: {col_map_human}")
    
    # Select and rename columns for join
    human_class_subset = human_class_df[[
        col_map_human['mrn'], 
        col_map_human['comment'], 
        col_map_human['critical_error']
    ]].copy()
    
    human_class_subset.columns = [
        'mrn', 
        'human_fab_comment', 
        'human_fab_critical_error'
    ]
    
    # Convert MRN to string for join
    human_fab_df['mrn'] = human_fab_df['mrn'].astype(str)
    human_class_subset['mrn'] = human_class_subset['mrn'].astype(str)
    
    # Left join
    print("\nPerforming left join on 'mrn'...")
    human_enriched = human_fab_df.merge(
        human_class_subset, 
        on='mrn', 
        how='left'
    )
    
    print(f"Enriched dataset: {len(human_enriched)} rows")
    print(f"New columns added: human_fab_comment, human_fab_critical_error")
    
    # Save enriched Human dataset
    print(f"\nSaving enriched Human dataset to: {HUMAN_FAB_FILE}")
    human_enriched.to_excel(HUMAN_FAB_FILE, index=False)
    print("Saved!")
    
    print("\n" + "=" * 60)
    print("DONE! Both datasets enriched with classification data")
    print("=" * 60)

if __name__ == "__main__":
    main()
