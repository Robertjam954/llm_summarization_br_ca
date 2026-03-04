# Project Rules — LLM Summarization BR/CA

## Project Identity
- **Name:** Prompt-Technique Evaluation for Feature-Level Human vs LLM Clinical Feature Extraction
- **Institution:** Memorial Sloan Kettering Cancer Center | Goel Lab
- **Repo:** github.com/Robertjam954/llm_summarization_br_ca

## Working Directory
`C:\Users\jamesr4\OneDrive - Memorial Sloan Kettering Cancer Center\Documents\GitHub\llm_summarization_br_ca`

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
- Raw data goes in `data/raw/` (gitignored)
- Only de-identified outputs in `data/processed/`
- API keys go in `.env` only (gitignored)

## Code Conventions
- Follow existing module structure in `src/`
- Experiment tracking: always log `run_id`, `git_commit_hash`, `prompt_id`, `model_id`, `dataset_snapshot_id`
- Use `python-dotenv` for all API key access
- Prefer `uv add <package>` over pip for dependency management
