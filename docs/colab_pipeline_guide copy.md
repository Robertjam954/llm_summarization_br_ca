# Running the Pipeline from Google Colab

**For use with a pre-deidentified dataset (NB02–NB07)**

This guide covers how to clone the repo, configure paths for the Colab environment, upload your deidentified data to Google Drive, and run each analysis notebook.

---

## Overview

| Step | Action | When |
|---|---|---|
| 1 | Set up Google Drive folder structure | Once |
| 2 | Upload deidentified dataset to Drive | Once (or when data changes) |
| 3 | Clone repo + install dependencies | Every new Colab session |
| 4 | Mount Drive + configure paths | Every new Colab session |
| 5 | Run NB02–NB07 | As needed |
| 6 | Download or commit outputs | After each session |

> **NB01 (deidentification) is skipped** — this guide assumes you already have the deidentified validation Excel and, optionally, deidentified PDF text files. NB01 must be run locally (see local setup in `README.md`).

---

## Part 1: One-Time Google Drive Setup

Create the following folder structure in your Google Drive. You only need to do this once.

```
My Drive/
└── llm_data_private/
    ├── raw/                          ← place deidentified Excel here
    ├── deidentified/                 ← redacted PDFs + case mapping (from NB01)
    ├── extracted_text/               ← per-case .txt files (from NB04 local run)
    └── extracted_text_comparison/    ← method comparison outputs (from NB04)
```

**Upload the deidentified validation Excel:**

1. Go to [drive.google.com](https://drive.google.com)
2. Navigate to `My Drive/llm_data_private/raw/`
3. Upload: `merged_llm_summary_validation_datasheet_deidentified.xlsx`

If you have extracted text files from a local NB04 run, upload the `.txt` files to `extracted_text/`. NB05 and NB07 require these.

---

## Part 2: Store API Keys in Colab Secrets

Colab Secrets persist across sessions and avoid hardcoding keys in notebooks.

1. In any Colab notebook: **Tools → Secrets** (left sidebar lock icon)
2. Add the following secrets:

| Secret Name | Value |
|---|---|
| `ANTHROPIC_API_KEY` | Your Anthropic API key (required for NB04 Claude extraction) |
| `OPENAI_API_KEY` | Your OpenAI key (if used) |

---

## Part 3: Session Setup Cell

**Paste this into a new cell at the top of any notebook before running it in Colab.** Run it once per session.

```python
# ── Colab Session Setup ────────────────────────────────────────────────────────
# Run this cell once at the start of each Colab session before executing the notebook.

import os
import sys
from pathlib import Path

# 1. Clone the repo (skips if already present)
if not Path("/content/llm_summarization_br_ca").exists():
    !git clone https://github.com/Robertjam954/llm_summarization_br_ca.git /content/llm_summarization_br_ca

# 2. Install dependencies
!pip install -q -r /content/llm_summarization_br_ca/requirements.txt

# 3. Add repo to Python path (so src/ imports work)
repo_root = "/content/llm_summarization_br_ca"
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# 4. Mount Google Drive
from google.colab import drive
drive.mount("/content/drive", force_remount=False)

# 5. Set path environment variables (overrides .env defaults)
os.environ["PROJECT_ROOT"]     = "/content/llm_summarization_br_ca"
os.environ["DATA_PRIVATE_DIR"] = "/content/drive/MyDrive/llm_data_private"

# 6. Load API keys from Colab Secrets
try:
    from google.colab import userdata
    os.environ["ANTHROPIC_API_KEY"] = userdata.get("ANTHROPIC_API_KEY") or ""
    os.environ["OPENAI_API_KEY"]    = userdata.get("OPENAI_API_KEY") or ""
except Exception:
    pass  # secrets not set or running outside Colab

# 7. Write .env so load_dotenv() in notebook cells picks up paths automatically
env_path = Path(repo_root) / ".env"
env_path.write_text(
    f'PROJECT_ROOT={os.environ["PROJECT_ROOT"]}\n'
    f'DATA_PRIVATE_DIR={os.environ["DATA_PRIVATE_DIR"]}\n'
    f'ANTHROPIC_API_KEY={os.environ.get("ANTHROPIC_API_KEY","")}\n'
    f'OPENAI_API_KEY={os.environ.get("OPENAI_API_KEY","")}\n'
)

print("✓ Repo:        ", repo_root)
print("✓ Drive:        /content/drive/MyDrive/")
print("✓ Data private:", os.environ["DATA_PRIVATE_DIR"])
print("✓ .env written to:", env_path)
```

---

## Part 4: Running Each Notebook

Open notebooks from `/content/llm_summarization_br_ca/notebooks/` in Colab (File → Open → GitHub or upload).

### NB01 — Deidentification
**Skip in Colab.** Run locally only. NB01 processes raw PHI-containing PDFs and must not be run in a cloud environment.

---

### NB02 — Missing Data Analysis
**Runtime:** CPU · ~2 min

**Requires:** `llm_data_private/raw/merged_llm_summary_validation_datasheet_deidentified.xlsx`

```python
# After the session setup cell, the notebook's existing path cell resolves correctly.
# Run all cells top to bottom.
```

**Outputs to** `PROJECT_ROOT/reports/`:
- `missingness_heatmap_by_feature_annotator.png`
- `missingness_bar_human_vs_ai.png`
- `missingness_per_feature_column.csv`
- `missingness_domain_summary.csv`

---

### NB03 — EDA, Classification & Diagnostic Metrics
**Runtime:** CPU · ~5 min (bootstrap CIs at n=2000 per metric)

**Requires:** `llm_data_private/raw/merged_llm_summary_validation_datasheet_deidentified.xlsx`

**Also requires** `metrics_utils.py` — verify it is in `src/` and importable.

```python
# If metrics_utils import fails, run:
import sys
sys.path.insert(0, "/content/llm_summarization_br_ca/src")
```

**Outputs to** `reports/`:
- `element_level_metrics.csv`
- `element_pvalues_one_sided.csv`
- `fabrication_rate_element_level.csv`
- `domain_level_aggregated_metrics.csv`
- Faceted diagnostic metric plots (`.png`)

---

### NB04 — Source Document Text Extraction
**Runtime:** CPU/GPU · variable (depends on number of PDFs)

**Requires:** Source PDFs in `llm_data_private/raw/`

> If you only have the deidentified Excel (no source PDFs), **skip NB04**. NB05 and NB07 will still run if you upload pre-extracted `.txt` files to `llm_data_private/extracted_text/`.

**For Claude Vision / Transcription methods** — `ANTHROPIC_API_KEY` must be set (Step 2 above).

**Outputs to** `llm_data_private/`:
- `extracted_text/<case_id>.txt` — per-case deidentified text
- `deidentified/case_document_mapping.csv`

**Outputs to** `reports/`:
- `text_extraction_quality.png`
- `extraction_method_comparison.csv`

---

### NB05 — Feature Extraction, OCR Quality & BERT
**Runtime:** GPU recommended (BERT encoding) · ~10–20 min with GPU

**Requires:** `.txt` files in `llm_data_private/extracted_text/` (from NB04)

**Enable GPU:** Runtime → Change runtime type → T4 GPU

```python
# Optional: install sentence-transformers if not in requirements.txt
!pip install -q sentence-transformers
```

**Outputs to** `data/features/` (committed):
- `page_level_ocr_quality.csv`
- `case_level_ocr_quality.csv`
- `bert_document_embeddings.csv`
- `case_text_features.csv`
- `case_all_features.csv`

---

### NB06 — Metadata & Data Dictionary
**Runtime:** CPU · ~1 min

**Requires:** `llm_data_private/raw/merged_llm_summary_validation_datasheet_deidentified copy.xlsx`

**Outputs to** `reports/`:
- `data_dictionary.xlsx` (styled, 3 sheets)
- `variable_names.xlsx`

---

### NB07 — Validation Methods Comparison
**Runtime:** GPU recommended (BERT TF-Hub) · ~20–40 min full run

**Requires:**
- `llm_data_private/raw/merged_llm_summary_validation_datasheet_deidentified.xlsx`
- `.txt` files in `llm_data_private/extracted_text/` (for ML/DL validators)

**Enable GPU:** Runtime → Change runtime type → T4 GPU

```python
# Install XGBoost and TF-Hub if needed
!pip install -q xgboost tensorflow tensorflow-hub
```

**Outputs to** `reports/`:
- `vectorization_benchmark.csv` + plot
- `ml_validation_cv_results.csv`
- `dl_validation_cv_results.csv`
- `shap_feature_rankings_by_vec_method.csv` + plot
- `validation_methods_comparison.csv` + heatmap + domain plot

---

## Part 5: Saving Outputs

Outputs written to `PROJECT_ROOT/reports/` are inside the cloned repo at `/content/llm_summarization_br_ca/reports/`. Options for persisting them:

### Option A — Commit and push from Colab

```python
import subprocess, os

os.chdir("/content/llm_summarization_br_ca")

# Configure git identity (once per session)
subprocess.run(["git", "config", "user.email", "your-email@mskcc.org"])
subprocess.run(["git", "config", "user.name", "Your Name"])

# Stage outputs
subprocess.run(["git", "add", "reports/", "data/features/"])
subprocess.run(["git", "commit", "-m", "Add analysis outputs from Colab run"])

# Push — requires a personal access token (PAT) with repo scope
# Set your PAT in Colab Secrets as GITHUB_PAT
from google.colab import userdata
pat = userdata.get("GITHUB_PAT")
remote_url = f"https://{pat}@github.com/Robertjam954/llm_summarization_br_ca.git"
subprocess.run(["git", "remote", "set-url", "origin", remote_url])
subprocess.run(["git", "push"])
```

### Option B — Copy outputs to Google Drive

```python
from pathlib import Path
import shutil

src = Path("/content/llm_summarization_br_ca/reports")
dst = Path("/content/drive/MyDrive/llm_reports_colab")
dst.mkdir(parents=True, exist_ok=True)

for f in src.glob("*"):
    shutil.copy2(f, dst / f.name)
    print(f"Copied: {f.name}")
```

### Option C — Download directly

```python
from google.colab import files
import zipfile
from pathlib import Path

# Zip the reports folder and download
with zipfile.ZipFile("/content/reports.zip", "w") as zf:
    for f in Path("/content/llm_summarization_br_ca/reports").glob("*"):
        zf.write(f, f.name)
files.download("/content/reports.zip")
```

---

## Notebook Execution Order

For a full run from a fresh deidentified dataset:

```
NB02 (missing data)
    ↓
NB03 (EDA + metrics)       ← primary analysis outputs
    ↓
NB04 (text extraction)     ← only if source PDFs available
    ↓
NB05 (features + BERT)     ← requires NB04 .txt outputs
    ↓
NB06 (data dictionary)     ← independent, run any time
    ↓
NB07 (validation methods)  ← requires NB04 .txt outputs
```

If source PDFs are not available (Excel-only), run: **NB02 → NB03 → NB06** independently, and skip NB04/NB05/NB07.

---

## Quick Troubleshooting

| Issue | Fix |
|---|---|
| `ModuleNotFoundError: metrics_utils` | `sys.path.insert(0, "/content/llm_summarization_br_ca/src")` |
| `FileNotFoundError` on DATA_PATH | Verify Drive is mounted and Excel is in `llm_data_private/raw/` |
| `RuntimeError: CUDA out of memory` (NB07 BERT) | Reduce `batch_size` from 16 to 8 in the `model.fit()` call |
| Colab session disconnects mid-run | Enable `Runtime → Run all` and keep browser tab active; or use Colab Pro |
| `load_dotenv()` not picking up paths | Re-run the session setup cell to rewrite `.env` |
| Git push authentication fails | Add `GITHUB_PAT` to Colab Secrets (repo-scope PAT from github.com/settings/tokens) |

---

*Last updated: March 2026 | Memorial Sloan Kettering | Goel Lab*
