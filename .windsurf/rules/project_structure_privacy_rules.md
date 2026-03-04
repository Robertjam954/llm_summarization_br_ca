# Project Rules — LLM Summarization BR/CA

## Project Identity
- **Name:** Prompt-Technique Evaluation for Feature-Level Human vs LLM Clinical Feature Extraction
- **Institution:** Memorial Sloan Kettering Cancer Center | Goel Lab
- **Repo:** github.com/Robertjam954/llm_summarization_br_ca

## Working Directory
`C:\Users\jamesr4\OneDrive - Memorial Sloan Kettering Cancer Center\Documents\GitHub\llm_summarization_br_ca`

## Path Definitions

| Variable | Path | Committed? | Contains |
|---|---|---|---|
| `PROJECT_ROOT` | `C:\Users\jamesr4\OneDrive...\llm_summarization_br_ca` | ✅ Yes (OneDrive + GitHub) | Source code, notebooks, non-PHI processed CSVs, reports |
| `PROCESSED_DIR` | `PROJECT_ROOT\data\processed` | ✅ Yes | Non-sensitive CSVs only — metrics, prompt library, analysis outputs |
| `DATA_PRIVATE_DIR` | `C:\Users\jamesr4\loc\data_private` | ❌ Never | Raw PDFs (PHI), deidentified PDFs, extracted text, case mappings |
| `OUTPUT_DIR` / `REPORTS_DIR` | `PROJECT_ROOT\reports` | ✅ Yes | Exported tables and figures for manuscript/presentations |

**Sub-paths within `DATA_PRIVATE_DIR` (all sensitive, local only):**
- `data_private\raw\` — source PDFs with patient PHI
- `data_private\deidentified\` — redacted PDFs + `case_document_mapping.csv`
- `data_private\extracted_text\` — per-case deidentified `.txt` files
- `data_private\extracted_text_comparison\` — method comparison text outputs

**Loaded in every notebook via:**
```python
from dotenv import load_dotenv
load_dotenv()
PROJECT_ROOT     = Path(os.getenv("PROJECT_ROOT",     r"..."))
DATA_PRIVATE_DIR = Path(os.getenv("DATA_PRIVATE_DIR", r"C:\Users\jamesr4\loc\data_private"))
```

## Tech Stack
- Python 3.12 (`.python-version`)
- Package manager: `uv` (`uv sync` to install deps)
- Config: `pyproject.toml`
- Env vars: `python-dotenv` via `.env` (never commit)

## Key Dependencies
- `anthropic` — primary LLM API (Claude)
- `pandas`, `numpy`, `scikit-learn`, `xgboost` — data & ML
- `PyMuPDF`, `pytesseract`, `opencv-python` — PDF/OCR processing
- `sentence-transformers` — embeddings
- `deepeval`, `pydantic` — evaluation framework
- Optional: `[dl]` tensorflow, `[h2o]` h2o

## Project Domain
- **Task:** LLM vs human extraction of 14 clinical elements from breast cancer surgical documents (radiology + pathology reports)
- **Dataset:** 200 patient cases × 45 columns (de-identified)
- **Annotator codes:** 1=Correct, 2=Omission, 3=Fabrication, N/A=Not applicable
- **Primary safety outcome:** Fabrication rate = FP / (FP + TN)
- **Prompt variants evaluated:** zero-shot, chain-of-thought, RAG, few-shot, program-aided, ReAct, BFOP, 2pop-mCODE

## Directory Structure
```
data/                      # raw PDFs (never commit PHI), processed outputs, splits
prompts/                   # library/ (versioned templates), generated/ (agent-created)
models/                    # configs/ (model specs, params, system prompts)
eval/                      # schemas/ (label defs), metrics/ (TP/FN/FP/TN code)
reports/                   # auto-generated tables, figures, dashboards
experiments/               # runs/ (run_id, git_commit_hash, prompt_id, model_id)
src/                       # core pipeline modules
tools/                     # CLI utilities, preprocessing scripts
notebooks/                 # Jupyter notebooks for analysis
docs/                      # protocol.md, data_dictionary.md, risk_and_safety.md, executive_summary.md
  docs/manuscript/         # project outline, methods, prompt documentation
  docs/manuscript_components/  # abstract, appendix, supplementary methods, cover letter
references/                # academic papers, reference materials (see REFERENCES_INDEX.md)
conferences/               # conference submissions by conference name
  conferences/acs_clinical_congress/  # ACS Clinical Congress abstract drafts
```

## Directory Structure Rules
- **Always append new folders** to the directory structure above when created
- `docs/manuscript_components/` — for manuscript submission files (abstract, appendix, supplementary, cover letter); exclude `lit review/` subfolder (kept separately)
- `conferences/<conference_name>/` — one subfolder per conference; use lowercase with underscores

## Data Privacy Rules
- **NEVER** commit patient data, MRNs, or identifiable information
- **NEVER** sync `DATA_PRIVATE_DIR` (`C:\Users\jamesr4\loc\data_private`) to OneDrive, GitHub, or any cloud storage
- Raw PDFs, deidentified PDFs, extracted text, and case mappings → `DATA_PRIVATE_DIR` only
- Non-PHI processed CSVs (metrics, analysis outputs) → `data/processed/` (safe to commit)
- Exported tables and figures → `reports/` (safe to commit)
- API keys → `.env` only (gitignored; see `.env.example` for template)

## Code Conventions
- Follow existing module structure in `src/`
- Experiment tracking: always log `run_id`, `git_commit_hash`, `prompt_id`, `model_id`, `dataset_snapshot_id`
- Use `python-dotenv` for all API key access
- Prefer `uv add <package>` over pip for dependency management
