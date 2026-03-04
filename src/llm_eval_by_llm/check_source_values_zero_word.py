"""Check source annotation values for zero-word cases."""

import json
import pandas as pd
from pathlib import Path
from collections import Counter

# Load data
VALIDATION_RAW = Path(r"C:\Users\jamesr4\OneDrive - Memorial Sloan Kettering Cancer Center\Documents\Research\Projects\moo\llm_summary\data\raw\merged_llm_summary_validation_datasheet.xlsx")
FEATURES_DIR = Path(r"C:\Users\jamesr4\OneDrive - Memorial Sloan Kettering Cancer Center\Documents\Research\Projects\moo\llm_summary\data\processed\source_doc_features")

df = pd.read_excel(VALIDATION_RAW)
with open(FEATURES_DIR / "all_cases_features.json") as f:
    cases = json.load(f)

# Get zero-word cases
zero_word_cases = []
for case in cases:
    if case.get("source_document_features"):
        word_count = case["source_document_features"]["combined_source_text_features"]["word_count"]
        if word_count == 0:
            zero_word_cases.append(case)

print(f"Analyzing {len(zero_word_cases)} zero-word cases")

# Check source values for key elements
elements = [
    ("lesion_size_status", "Lesion Size"),
    ("receptor_status", "Receptor Status"),
    ("histologic_diagnosis_status", "Histologic Diagnosis"),
    ("invasive_component_size_pathology_status", "Invasive Component Size (Pathology)"),
]

print("\n=== SOURCE VALUE DISTRIBUTIONS FOR ZERO-WORD CASES ===")

for col_prefix, display_name in elements:
    print(f"\n{display_name}:")
    values = []
    
    for case in zero_word_cases:
        case_idx = case["case_index"]
        row = df.iloc[case_idx]
        source_val = row.get(f"{col_prefix}_source")
        values.append(source_val)
    
    # Count unique values
    value_counts = Counter(values)
    for val, count in sorted(value_counts.items()):
        print(f"  Value {val}: {count} cases ({100*count/len(values):.1f}%)")

# Check if these are DCIS cases (might not have pathology)
print("\n=== TUMOR TYPE DISTRIBUTION ===")
tumor_types = []
for case in zero_word_cases:
    case_idx = case["case_index"]
    row = df.iloc[case_idx]
    tumor_val = row.get("tumor_invasive_dcis")
    tumor_types.append(tumor_val)

tumor_counts = Counter(tumor_types)
print(f"DCIS (0): {tumor_counts.get(0, 0)} cases")
print(f"Invasive (1): {tumor_counts.get(1, 0)} cases")
print(f"Other: {sum(1 for t in tumor_types if t not in [0, 1])} cases")

# Check a few example cases
print("\n=== EXAMPLE ZERO-WORD CASES ===")
for i, case in enumerate(zero_word_cases[:5]):
    case_idx = case["case_index"]
    row = df.iloc[case_idx]
    folder = case["source_document_features"]["case_folder"]
    
    print(f"\nCase {case_idx}: {folder}")
    print(f"  Tumor type: {'DCIS' if row.get('tumor_invasive_dcis') == 0 else 'Invasive'}")
    print(f"  Source values:")
    for col_prefix, display_name in elements:
        val = row.get(f"{col_prefix}_source")
        print(f"    {display_name}: {val}")

# Check if source documents exist for these cases
print("\n=== SOURCE DOCUMENT COUNTS ===")
doc_counts = []
for case in zero_word_cases:
    n_docs = case["source_document_features"]["n_source_documents"]
    doc_counts.append(n_docs)

doc_counter = Counter(doc_counts)
for n_docs, count in sorted(doc_counter.items()):
    print(f"  {n_docs} source documents: {count} cases")

# Check file types in these cases
print("\n=== FILE TYPES IN ZERO-WORD CASES ===")
file_types = Counter()
for case in zero_word_cases:
    for file_info in case["source_document_features"]["files"]:
        file_types[file_info["filename"].split('.')[-1]] += 1

for ext, count in sorted(file_types.items()):
    print(f"  .{ext}: {count} files")
