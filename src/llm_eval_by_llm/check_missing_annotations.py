"""Check if cases with failed text extraction have missing annotations (value=2)."""

import json
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict

# Load the original validation data to check source values
VALIDATION_RAW = Path(r"C:\Users\jamesr4\OneDrive - Memorial Sloan Kettering Cancer Center\Documents\Research\Projects\moo\llm_summary\data\raw\merged_llm_summary_validation_datasheet.xlsx")
FEATURES_DIR = Path(r"C:\Users\jamesr4\OneDrive - Memorial Sloan Kettering Cancer Center\Documents\Research\Projects\moo\llm_summary\data\processed\source_doc_features")

# Load validation data
df = pd.read_excel(VALIDATION_RAW)
print(f"Loaded validation data: {len(df)} rows")

# Load case features to identify which cases had failed text extraction
with open(FEATURES_DIR / "all_cases_features.json") as f:
    cases = json.load(f)

# Identify cases with failed text extraction
failed_cases = []
successful_cases = []

for case in cases:
    if not case.get("source_document_features"):
        continue
    
    if not case["source_document_features"].get("text_extraction_successful", True):
        failed_cases.append(case)
    else:
        successful_cases.append(case)

print(f"\nCases with failed text extraction: {len(failed_cases)}")
print(f"Cases with successful text extraction: {len(successful_cases)}")

# Check for missing annotations (value=2) in failed cases
print("\n=== MISSING ANNOTATIONS (value=2) IN FAILED EXTRACTION CASES ===")
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

missing_counts = defaultdict(int)
missing_cases_by_element = defaultdict(list)

# Check each failed case
for case in failed_cases:
    case_idx = case["case_index"]
    row = df.iloc[case_idx]
    
    for col_prefix, display_name in element_cols:
        source_val = row.get(f"{col_prefix}_source")
        if source_val == 2:  # Missing annotation
            missing_counts[display_name] += 1
            missing_cases_by_element[display_name].append({
                "case_index": case_idx,
                "case_folder": case["source_document_features"]["case_folder"] if case.get("source_document_features") else "No folder",
                "surgeon": row.get("surgeon", "Unknown"),
                "patient_initials": row.get("patient_initials", "Unknown")
            })

# Print missing annotation counts
print("\nMissing annotations (value=2) by element:")
for element, count in sorted(missing_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {element}: {count} cases")

# Focus on receptor status
print("\n=== RECEPTOR STATUS DETAILS ===")
if "Receptor Status" in missing_cases_by_element:
    print(f"Cases with missing receptor status: {len(missing_cases_by_element['Receptor Status'])}")
    for case_info in missing_cases_by_element["Receptor Status"]:
        print(f"  Case {case_info['case_index']}: {case_info['case_folder']} ({case_info['surgeon']} - {case_info['patient_initials']})")
else:
    print("No cases with missing receptor status in failed extraction group")

# Compare with successful cases
print("\n=== COMPARISON: MISSING ANNOTATIONS IN SUCCESSFUL vs FAILED CASES ===")

def count_missing(case_list, label):
    """Count missing annotations in a list of cases."""
    missing = defaultdict(int)
    total = 0
    
    for case in case_list:
        case_idx = case["case_index"]
        row = df.iloc[case_idx]
        total += 1
        
        for col_prefix, display_name in element_cols:
            source_val = row.get(f"{col_prefix}_source")
            if source_val == 2:
                missing[display_name] += 1
    
    print(f"\n{label} ({total} cases):")
    for element, count in sorted(missing.items(), key=lambda x: x[1], reverse=True):
        pct = 100 * count / total if total > 0 else 0
        print(f"  {element}: {count} ({pct:.1f}%)")
    
    return missing

missing_successful = count_missing(successful_cases, "Successful extraction")
missing_failed = count_missing(failed_cases, "Failed extraction")

# Check if missing annotations correlate with extraction failure
print("\n=== CORRELATION ANALYSIS ===")
print("Elements with higher missing rate in failed cases:")
for element in element_cols:
    _, display_name = element
    if display_name in missing_successful and display_name in missing_failed:
        rate_success = 100 * missing_successful[display_name] / len(successful_cases) if successful_cases else 0
        rate_failed = 100 * missing_failed[display_name] / len(failed_cases) if failed_cases else 0
        diff = rate_failed - rate_success
        if diff > 10:  # More than 10% difference
            print(f"  {display_name}: {rate_success:.1f}% vs {rate_failed:.1f}% (diff: {diff:.1f}%)")

# Check pathology-specific elements
pathology_elements = ["Invasive Component Size (Pathology)", "Histologic Diagnosis", "Receptor Status"]
print("\n=== PATHOLOGY ELEMENTS MISSING RATE ===")
for element in pathology_elements:
    success_rate = 100 * missing_successful.get(element, 0) / len(successful_cases) if successful_cases else 0
    failed_rate = 100 * missing_failed.get(element, 0) / len(failed_cases) if failed_cases else 0
    print(f"{element}:")
    print(f"  Successful extraction: {success_rate:.1f}%")
    print(f"  Failed extraction: {failed_rate:.1f}%")
