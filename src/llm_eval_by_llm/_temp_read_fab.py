import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(
    r"C:\Users\jamesr4\OneDrive - Memorial Sloan Kettering Cancer Center"
    r"\Documents\GitHub\llm_summarization_br_ca"
)
DATA_PRIVATE_DIR = Path(r"C:\Users\jamesr4\loc\data_private")
DEID_DIR = DATA_PRIVATE_DIR / "deidentified"
TEXT_DIR = DATA_PRIVATE_DIR / "extracted_text"
RAW_DIR = DATA_PRIVATE_DIR / "raw"

# 1. Read validation datasheet and case mapping
val = pd.read_excel(DEID_DIR / "validation_datasheet_deidentified.xlsx")
mapping = pd.read_csv(DEID_DIR / "patient_case_id_mapping.csv")

# 2. Find AI fabrication cases (status == 3)
ai_status_cols = [c for c in val.columns if c.endswith("_status_ai")]
fab_mask = val[ai_status_cols].apply(
    lambda row: any(str(v) == '3' for v in row), axis=1
)
fab_cases = val[fab_mask].copy()

# 3. Map surgeon+initials to raw folders
SURGEON_DIR_MAP = {
    "Barrio, Andrea": "Barrio",
    "Capko, Deborah": "Capko",
    "Kirstein, Laurie": "Kirstein",
    "Tadros, Audree": "Tadros",
    "El-Tamer, Mahmoud": "EL Tamer",
    "Heerdt, Alexandra": "Heerdt",
    "Lee, Min": "Lee",
    "Montag, Giacomo": "Montag",
    "Moo, Tracy-Ann": "Moo",
}

# 4. For each fab case, find source docs in raw dir
print("=== Fabrication Cases -> Source Docs ===\n")
for idx, row in fab_cases.iterrows():
    surgeon = row['surgeon']
    initials = row['patient_initials']
    fab_elements = [
        c.replace("_status_ai", "")
        for c in ai_status_cols if str(row[c]) == '3'
    ]
    surgeon_dir = SURGEON_DIR_MAP.get(surgeon, surgeon.split(",")[0].strip())
    surgeon_path = RAW_DIR / surgeon_dir

    # Find matching patient folder
    matched_folder = None
    if surgeon_path.exists():
        for case_dir in surgeon_path.iterdir():
            if case_dir.is_dir():
                parts = case_dir.name.split("_")
                # Format: SurgeonInitials_PatientInitials_...
                if len(parts) >= 2 and parts[1] == initials:
                    matched_folder = case_dir
                    break

    print(f"Row {idx}: {surgeon} / {initials}")
    print(f"  Fabricated: {fab_elements}")
    if matched_folder:
        pdfs = list(matched_folder.glob("*.pdf"))
        print(f"  Source folder: {matched_folder.name}")
        print(f"  PDFs: {[p.name for p in pdfs]}")
        # Check for existing deid PDFs via mapping
        map_rows = mapping[
            mapping['original_path'].str.contains(
                matched_folder.name, na=False
            )
        ]
        if not map_rows.empty:
            case_ids = map_rows['case_id'].unique()
            print(f"  Case IDs: {case_ids.tolist()}")
    else:
        print(f"  WARNING: No matching folder in {surgeon_path}")
    print()
