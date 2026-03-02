"""Check missing annotations for cases with 0 extracted words (scanned PDFs)."""

import json
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Load validation data and features
VALIDATION_RAW = Path(r"C:\Users\jamesr4\OneDrive - Memorial Sloan Kettering Cancer Center\Documents\Research\Projects\moo\llm_summary\data\raw\merged_llm_summary_validation_datasheet.xlsx")
FEATURES_DIR = Path(r"C:\Users\jamesr4\OneDrive - Memorial Sloan Kettering Cancer Center\Documents\Research\Projects\moo\llm_summary\data\processed\source_doc_features")

df = pd.read_excel(VALIDATION_RAW)
with open(FEATURES_DIR / "all_cases_features.json") as f:
    cases = json.load(f)

# Find cases with 0 words (scanned PDFs)
zero_word_cases = []
good_word_cases = []

for case in cases:
    if not case.get("source_document_features"):
        continue
    
    word_count = case["source_document_features"]["combined_source_text_features"]["word_count"]
    if word_count == 0:
        zero_word_cases.append(case)
    elif word_count > 100:
        good_word_cases.append(case)

print(f"Cases with 0 words (scanned PDFs): {len(zero_word_cases)}")
print(f"Cases with >100 words: {len(good_word_cases)}")

# Element columns to check
element_cols = [
    ("lesion_size_status", "Lesion Size"),
    ("laterality_status", "Lesion Laterality"),
    ("lesion_location_status", "Lesion Location"),
    ("calcifications_asymmetry_status", "Calcifications / Asymmetry"),
    ("additional_enhancement_mri_status", "Additional Enhancement (MRI)"),
    ("extent_status", "Extent"),
    ("accurate_clip_placement_status", "Accurate Clip Placement"),
    ("workup_recommendation_status", "Workup Recommendation"),
    ("Lymph node_status", "Lymph Node"),
    ("chronology_preserved_status", "Chronology Preserved"),
    ("biopsy_method_status", "Biopsy Method"),
    ("invasive_component_size_pathology_status", "Invasive Component Size (Pathology)"),
    ("histologic_diagnosis_status", "Histologic Diagnosis"),
    ("receptor_status", "Receptor Status"),
]

# Check missing annotations (value=2) in zero-word cases
print("\n=== MISSING ANNOTATIONS (value=2) IN ZERO-WORD CASES ===")
missing_in_zero = defaultdict(int)
missing_cases = defaultdict(list)

for case in zero_word_cases:
    case_idx = case["case_index"]
    row = df.iloc[case_idx]
    folder = case["source_document_features"]["case_folder"]
    
    for col_prefix, display_name in element_cols:
        source_val = row.get(f"{col_prefix}_source")
        if source_val == 2:  # Missing annotation
            missing_in_zero[display_name] += 1
            missing_cases[display_name].append({
                "case_index": case_idx,
                "folder": folder,
                "surgeon": row.get("surgeon"),
                "patient": row.get("patient_initials")
            })

# Print results
print("Missing annotations by element:")
for element, count in sorted(missing_in_zero.items(), key=lambda x: x[1], reverse=True):
    pct = 100 * count / len(zero_word_cases)
    print(f"  {element}: {count}/{len(zero_word_cases)} ({pct:.1f}%)")

# Focus on receptor status
print("\n=== RECEPTOR STATUS MISSING IN ZERO-WORD CASES ===")
if "Receptor Status" in missing_cases:
    print(f"Cases with missing receptor status: {len(missing_cases['Receptor Status'])}")
    for case in missing_cases["Receptor Status"]:
        print(f"  Case {case['case_index']}: {case['folder']} ({case['surgeon']} - {case['patient']})")
else:
    print("No missing receptor status in zero-word cases")

# Compare with good-word cases
print("\n=== COMPARISON: ZERO-WORD vs GOOD-WORD CASES ===")
def count_missing_in_cases(case_list, label):
    missing = defaultdict(int)
    for case in case_list:
        case_idx = case["case_index"]
        row = df.iloc[case_idx]
        
        for col_prefix, display_name in element_cols:
            source_val = row.get(f"{col_prefix}_source")
            if source_val == 2:
                missing[display_name] += 1
    
    print(f"\n{label} ({len(case_list)} cases):")
    for element, count in sorted(missing.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            pct = 100 * count / len(case_list)
            print(f"  {element}: {count} ({pct:.1f}%)")
    return missing

missing_zero = count_missing_in_cases(zero_word_cases, "Zero-word cases")
missing_good = count_missing_in_cases(good_word_cases, "Good-word cases")

# Specific pathology elements check
print("\n=== PATHOLOGY ELEMENTS: MISSING RATE COMPARISON ===")
pathology_elements = ["Invasive Component Size (Pathology)", "Histologic Diagnosis", "Receptor Status"]
for element in pathology_elements:
    zero_rate = 100 * missing_zero.get(element, 0) / len(zero_word_cases) if zero_word_cases else 0
    good_rate = 100 * missing_good.get(element, 0) / len(good_word_cases) if good_word_cases else 0
    print(f"\n{element}:")
    print(f"  Zero-word cases: {zero_rate:.1f}% ({missing_zero.get(element, 0)}/{len(zero_word_cases)})")
    print(f"  Good-word cases: {good_rate:.1f}% ({missing_good.get(element, 0)}/{len(good_word_cases)})")
    
    if zero_rate > good_rate:
        print(f"  → Higher missing rate in zero-word cases by {zero_rate - good_rate:.1f}%")

# List all zero-word cases with their pathology elements
print("\n=== ALL ZERO-WORD CASES WITH PATHOLOGY ELEMENTS ===")
for case in zero_word_cases:
    case_idx = case["case_index"]
    row = df.iloc[case_idx]
    folder = case["source_document_features"]["case_folder"]
    
    pathology_missing = []
    for col_prefix, display_name in pathology_elements:
        source_val = row.get(f"{col_prefix}_source")
        if source_val == 2:
            pathology_missing.append(display_name)
    
    if pathology_missing:
        print(f"Case {case_idx}: {folder}")
        print(f"  Missing: {', '.join(pathology_missing)}")
