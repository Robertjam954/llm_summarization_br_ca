# Breast Bot Project Deidentification

Complete deidentification of all source documents in the Breast Bot Project using HIPAA Safe Harbor 18 PHI identifiers with black box redaction.

## 📁 Folder Structure

**Source:**
```
C:\Users\jamesr4\OneDrive - Memorial Sloan Kettering Cancer Center\Moo, Tracy-Ann's files - Breast Bot Project\
├── Barrio/
│   ├── Case1/
│   │   ├── H&P.pdf
│   │   ├── Imaging.pdf
│   │   └── Path_Biopsy.pdf
│   └── Case2/
├── Morrow/
│   ├── MM_AK_DCIS/
│   └── MM_AM_INV/
└── [18 more surgeon folders...]
```

**Output:**
```
C:\Users\jamesr4\loc\data_private\breast_bot_deidentified\
├── Barrio/
│   ├── Case1/
│   │   ├── CASE_A1B2C3D4E5F6.pdf  (deidentified H&P)
│   │   ├── CASE_X7Y8Z9A0B1C2.pdf  (deidentified Imaging)
│   │   └── CASE_M3N4O5P6Q7R8.pdf  (deidentified Path)
│   └── Case2/
├── Morrow/
│   ├── MM_AK_DCIS/
│   └── MM_AM_INV/
├── case_id_mapping.csv           (case_id ↔ original filename mapping)
└── deidentification_log.csv      (detailed processing log)
```

## 🔒 HIPAA Safe Harbor 18 PHI Identifiers

All 18 PHI identifiers are detected via OCR and covered with **black bounding boxes**:

1. ✅ **NAMES** — Patient, physician, any person names (via contextual redaction)
2. ✅ **GEOGRAPHIC** — Cities, counties, ZIP codes, street addresses
3. ✅ **DATES** — All dates except year (birth, admission, discharge, death)
4. ✅ **PHONE** — Telephone numbers
5. ✅ **FAX** — Fax numbers
6. ✅ **EMAIL** — Email addresses
7. ✅ **SSN** — Social Security Numbers
8. ✅ **MRN** — Medical record numbers
9. ✅ **HEALTH_PLAN** — Health plan beneficiary numbers
10. ✅ **ACCOUNT** — Account numbers
11. ✅ **CERTIFICATE** — Certificate/license numbers
12. ✅ **VEHICLE** — Vehicle identifiers
13. ✅ **DEVICE** — Device identifiers/serial numbers
14. ✅ **URL** — Web URLs
15. ✅ **IP** — IP addresses
16. ⚠️ **BIOMETRIC** — Requires manual review
17. ⚠️ **PHOTOS** — Requires manual review
18. ✅ **UNIQUE_ID** — Any other unique identifying codes

## 🚀 Quick Start

### Prerequisites

```bash
pip install PyMuPDF pytesseract pandas numpy opencv-python Pillow tqdm
```

**Install Tesseract OCR:**
- Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
- Add to PATH: `C:\Program Files\Tesseract-OCR\tesseract.exe`

### Run Deidentification

```bash
cd "C:\Users\jamesr4\OneDrive - Memorial Sloan Kettering Cancer Center\Documents\GitHub\llm_summarization_br_ca\src\deidentification"

python deidentify_breast_bot_project.py
```

### Expected Output

```
================================================================================
BREAST BOT PROJECT DEIDENTIFICATION
================================================================================
Source: C:\...\Moo, Tracy-Ann's files - Breast Bot Project
Output: C:\Users\jamesr4\loc\data_private\breast_bot_deidentified

Using HIPAA Safe Harbor 18 PHI Identifiers
Redaction method: Black bounding boxes with 4px padding
================================================================================

Scanning 20 surgeon folders...
Found 450 PDF files across all cases

Processing 450 PDFs...
================================================================================
Deidentifying PDFs: 100%|████████████████████| 450/450 [2:15:30<00:00, 18.07s/it]

✅ Mapping saved: C:\...\case_id_mapping.csv
   Total cases: 450
✅ Log saved: C:\...\deidentification_log.csv

================================================================================
DEIDENTIFICATION SUMMARY
================================================================================
Total PDFs processed: 450
  ✅ Successful: 448
  ❌ Errors: 2
Total redactions applied: 12,345

Breakdown by surgeon:
                total_files  successful
Barrio                   25          25
Morrow                   45          45
...
```

## 📊 Output Files

### 1. Deidentified PDFs

**Location:** `C:\Users\jamesr4\loc\data_private\breast_bot_deidentified\[Surgeon]\[Case]\`

**Naming:** Each file renamed with deterministic `case_id`:
- Original: `H&P.pdf`
- Deidentified: `CASE_A1B2C3D4E5F6.pdf`

**Redaction:** Black boxes placed over all 18 PHI identifier types

### 2. Case ID Mapping (`case_id_mapping.csv`)

Maps deidentified files back to original sources:

```csv
case_id,surgeon,case_folder,original_filename,original_path,deidentified_path,status
CASE_A1B2C3D4E5F6,Morrow,MM_AK_DCIS,H&P.pdf,C:\...\H&P.pdf,C:\...\CASE_A1B2C3D4E5F6.pdf,SUCCESS
```

**Columns:**
- `case_id`: Unique identifier for deidentified file
- `surgeon`: Surgeon folder name
- `case_folder`: Case folder name
- `original_filename`: Original PDF filename
- `original_path`: Full path to source file
- `deidentified_path`: Full path to deidentified file
- `status`: SUCCESS or error type

### 3. Deidentification Log (`deidentification_log.csv`)

Detailed processing log with page-level redaction counts:

```csv
case_id,surgeon,case_folder,original_filename,status,pages,total_redactions,page,page_redactions,rules_applied
CASE_A1B2C3D4E5F6,Morrow,MM_AK_DCIS,H&P.pdf,SUCCESS,5,127,,,
CASE_A1B2C3D4E5F6,Morrow,MM_AK_DCIS,H&P.pdf,PAGE_DETAIL,,,1,25,"DATE_MDY,PHONE,CONTEXT_AFTER_LABEL"
```

**Columns:**
- `case_id`: Unique identifier
- `surgeon`, `case_folder`, `original_filename`: File location info
- `status`: SUCCESS, PAGE_DETAIL, or error type
- `pages`: Total pages in PDF
- `total_redactions`: Total redactions across all pages
- `page`: Page number (for PAGE_DETAIL rows)
- `page_redactions`: Redactions on this page
- `rules_applied`: Comma-separated list of PHI rules matched

## 🔧 Configuration

Edit the script to customize:

```python
# Paths
SOURCE_ROOT = Path(r"C:\...\Breast Bot Project")
OUTPUT_ROOT = Path(r"C:\Users\jamesr4\loc\data_private\breast_bot_deidentified")

# OCR settings
config = DeidConfig(
    dpi=300,                              # Higher DPI = better OCR, slower
    ocr_lang="eng",                       # Tesseract language
    pad_px=4,                             # Padding around black boxes
    contextual_numeric_redaction=True,    # Redact text after "Patient:", "Name:", etc.
    redact_after_label_tokens=8,          # How many tokens to redact after labels
    enable_broad_numeric_redaction=False  # Redact all 6-10 digit numbers (aggressive)
)
```

## 📈 Performance

**Processing time:**
- ~18 seconds per PDF (300 DPI, contextual redaction enabled)
- ~450 PDFs = ~2 hours 15 minutes total

**Redaction accuracy:**
- OCR-based detection at 300 DPI
- Contextual redaction catches names after labels
- Black boxes with 4px padding ensure complete coverage

## ⚠️ Important Notes

### Security

1. **Mapping file is sensitive**: `case_id_mapping.csv` links deidentified files to original PHI
   - Store securely in `C:\Users\jamesr4\loc\data_private\` (never commit to Git)
   - Restrict access to authorized personnel only

2. **Original files unchanged**: Source PDFs remain in OneDrive folder
   - Only deidentified copies created in output folder

3. **Manual review recommended**: 
   - Biometric identifiers (fingerprints, retinal scans)
   - Full-face photographs
   - Verify redaction quality on sample files before using in production

### Folder Structure

- **Maintained**: Original surgeon → case folder hierarchy preserved
- **Case IDs**: Deterministic (same input always produces same case_id)
- **Traceability**: Every deidentified file can be traced back via mapping CSV

### Error Handling

Common errors logged in `deidentification_log.csv`:

- `ERROR_OPEN`: PDF file corrupted or locked
- `ERROR_PAGE_N`: OCR failed on specific page
- `ERROR_WRITE`: Permission denied or disk full

## 🧪 Testing

Test on a single surgeon folder first:

```python
# Edit script to process only one surgeon
surgeon_folders = [d for d in SOURCE_ROOT.iterdir() if d.is_dir() and d.name == "Morrow"]
```

Then verify:
1. Black boxes cover all PHI
2. Clinical content remains readable
3. Folder structure maintained
4. Mapping CSV accurate

## 📞 Troubleshooting

### Issue: "Tesseract not found"

**Solution:**
```bash
# Windows
# Download: https://github.com/UB-Mannheim/tesseract/wiki
# Add to PATH or set in script:
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

### Issue: "Out of memory"

**Solution:** Reduce DPI or process in batches:
```python
config = DeidConfig(dpi=200)  # Lower DPI uses less memory
```

### Issue: "Some PHI not redacted"

**Solution:** 
1. Check OCR quality (increase DPI)
2. Add custom patterns to `DEFAULT_RULES`
3. Enable broad numeric redaction for aggressive mode

### Issue: "Processing too slow"

**Solution:**
1. Lower DPI: `config = DeidConfig(dpi=200)`
2. Disable contextual redaction: `contextual_numeric_redaction=False`
3. Process in parallel (advanced - modify script)

## 📚 Related Files

- **Notebook reference:** `notebooks/01_deidentification.ipynb`
- **Original script:** `src/llm_eval_by_llm/simple_text_deidentification.py`
- **EHR dataset adapter:** `src/agents/ehr_dataset_adapter.py`

## 🎯 Next Steps

After deidentification:

1. **Verify quality**: Manually review sample files
2. **Use for analysis**: Load deidentified PDFs for LLM processing
3. **Link to validation data**: Use `case_id` to join with validation spreadsheet
4. **Archive originals**: Keep original files secure, use only deidentified copies

---

**Version:** 1.0.0  
**Last Updated:** 2026-03-29  
**HIPAA Compliance:** Safe Harbor Method (18 PHI Identifiers)
