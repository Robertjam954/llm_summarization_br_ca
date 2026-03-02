# LLM Clinical Feature Extraction — Validation & Evaluation

**Memorial Sloan Kettering | Goel Lab**

Prompt-technique evaluation for feature-level human vs. LLM validation of clinical summary extraction from scanned breast cancer radiology and pathology reports.

---

## Overview

Large Language Models are increasingly deployed to extract structured clinical features from unstructured medical records. In breast cancer care, accurate extraction of 14 key clinical elements is critical for surgical planning, tumor board review, and multidisciplinary documentation. This project provides a rigorous, reproducible framework for quantifying LLM performance compared to human expert annotation, with particular focus on **fabrication (hallucination) risks** that pose direct patient safety threats.

## Primary Outcomes (Classification Metrics)

| Metric | Definition | Clinical Significance |
|--------|------------|----------------------|
| **Correct Rate** | TP / (TP + FN + FP) | Proportion of accurately extracted features |
| **Omission Rate** | FN / (TP + FN) | Missed features present in source documents |
| **Fabrication Rate** | FP / (FP + TN) | **Hallucinated features** not present in source (primary safety risk) |

**Annotator coding:** 1 = Correct, 2 = Omission, 3 = Fabrication, N/A = Not applicable

## Secondary Outcomes (Safety & Performance)

| Metric | Domain | Application |
|--------|--------|-------------|
| **Sensitivity (Recall)** | Diagnostic | Ability to detect present features |
| **Specificity** | Diagnostic | Ability to reject absent features |
| **Positive Predictive Value** | Diagnostic | Reliability of positive extractions |
| **Negative Predictive Value** | Diagnostic | Reliability of negative extractions |
| **Cohen's κ** | Agreement | Human vs AI inter-rater reliability |
| **Domain-stratified performance** | Safety | Radiology vs Pathology element comparison |
| **Prompt technique impact** | Optimization | Effect of prompting strategies on outcomes |

## Key Questions

1. **Safety**: Does AI fabrication rate significantly exceed human fabrication rate for any clinical element?
2. **Reliability**: Which elements are most fragile (highest fabrication/omission rates)?
3. **Predictors**: Which document features (OCR quality, lexical diversity, negation frequency) predict AI errors?
4. **Retrieval**: Does RAG-based retrieval reduce hallucination vs full-document prompting?
5. **Domain**: Are radiology elements more/less fragile than pathology elements?
6. **Optimization**: How do prompt iterations affect diagnostic metrics across runs?
7. **Vectorization**: How do text vectorization methods affect ML-based validation performance?
8. **Ontology**: Does mCodeGPT DAG extraction (RLS/BFOP/2POP) reduce fabrication vs flat prompting?
9. **Method**: Which DAG-based prompt method achieves best completeness-accuracy balance?
10. **Prediction**: Can embeddings + document quality predict extraction correctness/fabrication?

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

**Clinical Elements (14):**
- *Radiology*: Lesion Size, Laterality, Location, Calcifications/Asymmetry, MRI Enhancement, Extent, Clip Placement, Workup Recommendation, Lymph Node Status
- *Pathology*: Chronology Preservation, Biopsy Method, Invasive Component Size, Histologic Diagnosis, Receptor Status

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
| 08 | `08_ocr_image_quality_deblur.ipynb` | Blur detection scan, deblur pipeline, before/after OCR quality comparison |
| 09 | `09_mcodegpt_dag_extraction.ipynb` | mCodeGPT DAG extraction with RLS/BFOP/2POP methods for 14 breast cancer elements |
| 10 | `10_openai_predictive_model.ipynb` | OpenAI embeddings + document quality → predict extraction outcomes; multi-embedding (OpenAI, TF-IDF+SVD, Sentence-BERT) × multi-algorithm (Logistic Regression, Gradient Boosting, SVM, TabNet) comparison across correct/omission/fabrication |

## Repository Structure

```
llm_summarization_br_ca/                          ← PROJECT_ROOT (OneDrive + GitHub)
├── notebooks/              10 Jupyter notebooks (01–10) with MSK | Goel Lab headers
├── study_records/          Study protocol, hypothesis, aims (PI-reviewed documents)
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
│   ├── comprehensive_model_comparison.csv    — 3 embeddings × 4 algorithms × 3 outcomes
│   ├── model_comparison_*_auc.png            — AUC bar plots per outcome
│   ├── model_comparison_*_heatmap.png        — AUC heatmaps (embedding × algorithm)
├── prompts/
│   ├── prompt_library.csv      9 prompt versions with metadata
│   ├── library/                Frozen prompt templates
│   └── generated/              Agent-derived prompts
├── references/             Academic papers, infographics, setup guide (see REFERENCES_INDEX.md)
├── src/
│   ├── llm_eval_by_human/  Primary analysis scripts (metrics_utils.py, main_analysis.py)
│   ├── llm_eval_by_ml/     XGBoost/SHAP/vectorization scripts (NB07 templates)
│   ├── llm_eval_by_llm/    LLM extraction pipeline (source_document_feature_extraction v1–v3)
│   ├── classifier_models_prompt_optimization/  Classifier benchmarks (PCA, SVM, XGB, etc.)
│   ├── data collection and processing/         Deidentification scripts (R + Python)
│   ├── prompt_eng/         Prompt drafts, mCODE structure, developer prompt
│   ├── misc_scripts/       Utility scripts (OCR, ROUGE/BLEU, vector store, Mistral)
│   └── notebooks_legacy/   Pre-refactor notebooks and markdown analysis notes
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

### Prompt Engineering Techniques

Extraction prompts employ a multi-technique approach based on the Anthropic Prompt Engineering Tutorial:

| Version | Label | Techniques |
|---|---|---|
| v1 | Initial extraction | Basic structured prompt, JSON output |
| v2 | Zero-shot + CoT + anti-fabrication | Role assignment, 5-step chain-of-thought, 6 anti-fabrication rules, confidence scoring, strict JSON |
| v3 | Few-shot enhanced | v2 + 2–3 clinical examples per modality (standard, conflict, missing data) |
| v3+ | Few-shot + RAG + self-consistency | v3 + RAG verification loop + self-consistency on critical features (receptor status, invasive size) |

**Core techniques:** Role assignment (Ch. 3), chain-of-thought reasoning (Ch. 6), avoiding hallucinations (Ch. 8), formatted output (Ch. 5), few-shot prompting (Ch. 7), data/instruction separation (Ch. 4), clear directives (Ch. 2).

**Advanced methods:** RAG verification (re-query vector store to confirm traceability), self-consistency (3× extraction on safety-critical features), tree-of-thought (design phase for ambiguous edge cases), confidence calibration (0.0–1.0 per extraction).

See [`docs/executive_summary.md` Appendix H](docs/executive_summary.md) for the full taxonomy.

### Predictive Modeling (Notebook 10)

| Embedding Method | Description |
|---|---|
| **OpenAI text-embedding-3-large** | 3072-dim dense embeddings → PCA(50) reduction |
| **TF-IDF + SVD** | Sparse TF-IDF (5000 features, 1–2 grams) → TruncatedSVD(100) |
| **Sentence-BERT** | all-MiniLM-L6-v2 dense sentence embeddings (384-dim) |

| Algorithm | Key Parameters |
|---|---|
| **Logistic Regression** | C=0.1, balanced class weights, max_iter=1000 |
| **Gradient Boosting** | 200 trees, depth=3, lr=0.05 |
| **SVM (Linear)** | Calibrated LinearSVC, balanced weights |
| **TabNet** | n_d=64, n_a=64, entmax masking (optional) |

All combinations evaluated on three binary outcomes (correct, omission, fabrication) via 5-fold stratified CV with AUC, accuracy, and F1 metrics.

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
- **Colab Pipeline Guide:** [`docs/colab_pipeline_guide.md`](docs/colab_pipeline_guide.md)
- **References Index:** [`references/REFERENCES_INDEX.md`](references/REFERENCES_INDEX.md)
- **Environment Setup:** [`references/environment_setup.md`](references/environment_setup.md)

## License

MIT — see [LICENCE](LICENCE) for details.

---

*Memorial Sloan Kettering Cancer Center | Goel Lab*