# LLM Clinical Feature Extraction — Validation & Evaluation

**Memorial Sloan Kettering | Goel Lab**

Prompt-technique evaluation for feature-level human vs. LLM validation of clinical summary extraction from scanned breast cancer radiology and pathology reports.

---

## Motivation

Large Language Models are increasingly used to extract structured clinical features from unstructured medical records. In breast cancer care, accurate feature extraction is critical for surgical planning, tumor board review, and multidisciplinary documentation. However, LLM-generated summaries may **fabricate** clinical features not present in the source documents or **omit** features that are present — both failure modes pose direct patient safety risks.

This project provides a rigorous, reproducible framework for quantifying these failure modes and comparing validation approaches: **human annotator**, **LLM-as-judge**, **ML (XGBoost)**, and **deep learning (BERT)**.

## Key Questions

1. Does the AI fabrication rate significantly exceed the human fabrication rate for any clinical element?
2. Which clinical elements are most fragile (highest fabrication / omission rates)?
3. Which document-level features (OCR quality, lexical diversity, negation frequency) best predict AI extraction errors?
4. Does RAG-based retrieval reduce hallucination compared to full-document prompting?
5. Are Radiology elements more or less fragile than Pathology elements?
6. How do prompt iterations change diagnostic metrics across extraction runs?
7. How do text vectorization methods affect ML-based validation performance?

## Dataset

**200 patient cases × 45 columns** covering 14 clinical elements, each scored by a human annotator and an AI annotator against ground-truth source documents.

| Variable Group | Count | Description |
|---|---|---|
| Source (ground truth) | 14 | Binary (0/1): feature present in source documents |
| Human annotator | 14 | Coded 1/2/3/N/A: human extraction status |
| AI annotator | 14 | Coded 1/2/3/N/A: LLM extraction status |
| Covariates | 2 | `tumor_invasive_dcis`, `complex_case_status` |
| Identifier | 1 | `surgeon_id` (de-identified, 20 unique) |

**Annotator coding:** 1 = Correct, 2 = Omission, 3 = Fabrication, N/A = Not applicable

**Primary safety outcome:** Fabrication rate = FP / (FP + TN)

## Notebooks

| # | Notebook | Description |
|---|---|---|
| 01 | `01_deidentification.ipynb` | PHI redaction from scanned PDFs + validation Excel via regex + bounding-box OCR |
| 02 | `02_missing_data_analysis.ipynb` | Feature- and observation-level missingness analysis with heatmaps and domain breakdown |
| 03 | `03_eda_classification_diagnostic_metrics.ipynb` | EDA, confusion matrices, element/domain diagnostic metrics, bootstrap CIs, McNemar inference |
| 04 | `04_source_doc_text_extraction.ipynb` | Multi-method text extraction (pytesseract, Claude Vision, Claude Transcription) with quality comparison |
| 05 | `05_feature_extraction_ocr_bert.ipynb` | OCR image quality scoring, BERT embeddings, text features, H2O feature interactions |
| 06 | `06_metadata_data_dictionary.ipynb` | Auto-generated data dictionary + variable names Excel with styled formatting |
| 07 | `07_validation_methods_comparison.ipynb` | Vectorization benchmark (5 methods), XGBoost + BERT validation, SHAP importance, stratified comparison |

## Repository Structure

```
llm_summarization_br_ca/                          ← PROJECT_ROOT (OneDrive + GitHub)
├── notebooks/              7 Jupyter notebooks (01–07) with MSK | Goel Lab headers
├── data/
│   ├── processed/          Non-PHI CSVs: metrics, prompt library, analysis outputs  ← committed
│   ├── features/           BERT embeddings, OCR quality, text features (NB05)        ← committed
│   └── splits/             Train/test definitions                                     ← committed
├── docs/
│   ├── executive_summary.md    Full technical executive summary + code map
│   ├── dataset_metadata.md     YAML front-matter dataset specification
│   ├── manuscript/             Project outline, methods, prompt documentation
│   └── manuscript_components/  Abstract, appendix, supplementary methods, cover letter
├── conferences/
│   └── acs_clinical_congress/  ACS Clinical Congress abstract drafts
├── reports/                Exported figures and tables for manuscript/presentations   ← committed
├── prompts/
│   ├── prompt_library.csv      9 prompt versions with metadata
│   ├── library/                Frozen prompt templates
│   └── generated/              Agent-derived prompts
├── references/             Academic papers, infographics, setup guide (see REFERENCES_INDEX.md)
├── src/                    Source code scaffold (modeling, services, config)
├── eval/                   Evaluation schemas and metric definitions
├── models/                 Model configurations
├── experiments/            Run tracking (run_id, commit hash, prompt_id, results)
├── tools/
│   ├── colab/              BERT fine-tuning notebook for Cloud TPUs
│   └── watch_repo.sh       Local macOS notifications for remote repo changes
├── .env.example            Path variables + API key template (copy to .env, never commit .env)
├── pyproject.toml          Project metadata + dependencies
├── requirements.txt        Pip-compatible dependency list
└── LICENCE                 MIT

C:\Users\jamesr4\loc\data_private\               ← DATA_PRIVATE_DIR (local only — never committed)
├── raw\                    Source PDFs with PHI (input to NB01)
├── deidentified\           Redacted PDFs + case_document_mapping.csv (NB01 output)
├── extracted_text\         Per-case deidentified .txt files (NB04 output)
└── extracted_text_comparison\  Method comparison text outputs (NB04 multi-method)
```

> **Path model:** All notebooks load `PROJECT_ROOT` and `DATA_PRIVATE_DIR` from `.env` via `python-dotenv`.
> Non-PHI processed data → `data/processed/` (committed). Sensitive/raw data → `data_private/` (local only).

## Methods Overview

### Validation Approaches (Notebook 07)

| Method | Description |
|---|---|
| **Human** | Expert annotator element-level scoring (1/2/3/N/A) |
| **LLM** | Base-model extraction with 9 prompt variants (zero-shot, CoT, RAG, few-shot, ReAct, etc.) |
| **ML (XGBoost)** | Multi-class classifier (correct/omission/fabrication) per vectorization method, 5-fold stratified CV |
| **DL (BERT)** | TF-Hub BERT fine-tuning for 3-class sequence classification, 5-fold CV |

### Vectorization Methods Benchmarked

- DictVectorizer — token-frequency dicts to sparse matrix
- FeatureHasher — hash trick to fixed-size (2^18) vector
- CountVectorizer — built-in tokenizer + word counts
- HashingVectorizer — built-in tokenizer + hashing
- TfidfVectorizer — TF-IDF weighted features

Results are stratified by **evaluation method × vectorization method × domain** (Radiology vs. Pathology).

## Getting Started

### Prerequisites

- Python 3.11+ (3.12 recommended)
- macOS, Linux, or WSL
- API keys for Claude (Anthropic) if running NB04 extraction methods

### Setup

```bash
git clone https://github.com/Robertjam954/llm_summarization_br_ca.git
cd llm_summarization_br_ca

# Option A: uv (recommended)
uv sync

# Option B: pip
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Configure paths and API keys
cp .env.example .env
# Edit .env — set PROJECT_ROOT, DATA_PRIVATE_DIR, and API keys
```

### Running Notebooks

1. Copy `.env.example` → `.env` and fill in `PROJECT_ROOT` and `DATA_PRIVATE_DIR`
2. Place raw source PDFs in `DATA_PRIVATE_DIR\raw\` (never in the repo)
3. Open any notebook in VS Code or JupyterLab and select the `.venv` kernel
4. All paths resolve automatically from `.env` via `python-dotenv`

### Monitoring Remote Changes

```bash
# Start background watcher (macOS notifications every 5 min)
nohup ./tools/watch_repo.sh &

# Custom interval (e.g., 60 seconds)
nohup ./tools/watch_repo.sh 60 &
```

## Documentation

- **Executive Summary & Code Map:** [`docs/executive_summary.md`](docs/executive_summary.md)
- **Dataset Specification:** [`docs/dataset_metadata.md`](docs/dataset_metadata.md)
- **References Index:** [`references/REFERENCES_INDEX.md`](references/REFERENCES_INDEX.md)
- **Environment Setup:** [`references/environment_setup.md`](references/environment_setup.md)

## License

MIT — see [LICENCE](LICENCE) for details.

---

*Memorial Sloan Kettering Cancer Center | Goel Lab*