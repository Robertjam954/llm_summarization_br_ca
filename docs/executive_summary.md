---
title: "Prompt-Technique Evaluation for Feature-Level Human vs LLM Clinical Feature Extraction"
subtitle: "Executive Summary, Dataset Specification, Analytic Pipeline & Code Reference Map"
date: "2026-03-01"
author: "Robertjam954"
---

## Context

Large Language Models (LLMs) are increasingly used to extract structured clinical features from unstructured medical records — radiology reports, pathology reports, and operative notes. In breast cancer care, accurate feature extraction is critical for surgical planning, tumor board review, and multidisciplinary documentation. However, LLM-generated summaries may **fabricate** clinical features not present in the source documents or **omit** features that are present. Both failure modes pose direct patient safety risks.

As the volume of scanned clinical documents grows, manual validation by human annotators becomes a bottleneck. Understanding where, why, and how often the AI fails — and whether prompt engineering can mitigate these failures — is essential to safely deploying LLM-based extraction in clinical workflows.

## Objective

Evaluate the accuracy, omission rate, and fabrication rate of LLM-extracted clinical features compared to human annotator extraction across 14 clinical elements and 200 patient cases. Identify actionable patterns in AI failure modes and determine whether prompt engineering techniques can reduce extraction errors.

## Key Questions

1. Does the AI fabrication rate significantly exceed the human fabrication rate for any clinical element?
2. Which clinical elements are most fragile (highest fabrication / omission rates)?
3. Which document-level features (OCR quality, lexical diversity, negation frequency, embedding variance) best predict AI extraction errors?
4. Does RAG-based retrieval reduce hallucination compared to full-document prompting?
5. Are Radiology elements more or less fragile than Pathology elements?
6. How do prompt iterations change diagnostic metrics across extraction runs?
7. What is the comparative performance of different text extraction methods (pytesseract OCR vs. Claude Vision API) on scanned clinical documents?
8. Does ontology-guided extraction (mCodeGPT DAG methods RLS/BFOP/2POP) reduce fabrication rates compared to flat prompting?
9. Which prompt method (RLS, BFOP, 2POP) achieves the best balance of completeness and accuracy for breast cancer features?
10. Can embeddings + document quality predict extraction outcomes, and which embedding method (OpenAI, TF-IDF+SVD, Sentence-BERT) and algorithm (LR, GBM, SVM, TabNet) best predicts correctness, omission, and fabrication?

## Dataset Description

The primary dataset contains **200 patient cases × 45 columns**, covering 14 clinical elements each scored by a human annotator and an AI annotator against ground-truth source documents.

| Variable Group | Count | Description |
|---|---|---|
| Source (ground truth) | 14 | Binary (0/1): feature present in source documents |
| Human annotator | 14 | Coded 1/2/3/N/A: human extraction status |
| AI annotator | 14 | Coded 1/2/3/N/A: LLM extraction status |
| Covariates | 2 | `tumor_invasive_dcis` (1=Invasive, 2=DCIS), `complex_case_status` (0/1) |
| Identifier | 1 | `surgeon_id` (de-identified, 20 unique) |

**Annotator Coding:** 1 = Correct extraction, 2 = Omission, 3 = Fabrication, N/A = Not applicable

**Primary safety outcome:** `Fabrication rate = FP / (FP + TN)`

---

## 1. Executive Summary

This project evaluates whether different prompting techniques improve LLM-based validation of clinical summary features relative to human validation.

Independent Variable (IV)

Prompt technique:

zero_shot_structured_extraction_prod

chain_of_thought_3

rag_*

few_shot_*

program_aided_*

react_*

2pop_mcode_gpt

bfop_*

rls_mcode_gpt

Base model only (no fine-tuning in this phase).

Primary Dependent Variables (DVs)

Correct rate (label = 1)

Omission rate (label = 2)

Fabrication rate (label = 3; only when source == 1)

Secondary DVs

Cohen’s kappa (human vs AI)

Domain-stratified performance (radiology vs pathology)

Per-element performance

Time-series stability of metrics

Key Confounder (Required)

OCR quality / document image quality

2. Data Model and Label Encoding
Source Encoding

*_status_source

1 = present in source

0 = absent in source

Human & AI Encoding (only when source==1)

1 = correct

2 = omission

3 = fabrication

If source==0:

human and ai = NaN or "N/A"

3. LangChain + RAG Integration (New Core Layer)
Purpose

Replace full-document prompting with retrieval-augmented extraction to:

Reduce fabrication

Improve traceability

Enable evidence citation

Support interpretability analysis

3.1 OCR Loader

Scanned PDFs → OCR → LangChain Documents

def ocr_pdf_to_documents(pdf_path):
    # Rasterize via PyMuPDF
    # OCR via pytesseract
    # Return list[Document(page_content, metadata)]
3.2 Chunking Strategy
RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)
3.3 Embedding Options

OpenAI text-embedding-3-large (**prioritized** — demonstrated strong performance on clinical OCR text vectorization; see `references/ios_app/gpt-5_frontend.ipynb` for API pattern using `openai.OpenAI()` client)

HuggingFace sentence-transformers/all-mpnet-base-v2

Others allowed if logged

Embedding config must be stored per run.

3.4 Vector Store Options

FAISS (recommended default)

Chroma (persistent)

Qdrant / PGVector (if scaling later)

3.5 Per-Element RAG Extraction

Each feature uses:

Query:

Element name + definition + synonyms

Prompt must:

Restrict reasoning to provided context

Return structured JSON

Return null if not found

3.6 Required Extraction Output Schema
{
  "case_id": "CASE_0001",
  "model": "gpt-4o-mini",
  "prompt_id": "P2",
  "elements": [
    {
      "element_name": "ER_status",
      "value": "positive",
      "evidence": "Estrogen receptor: Positive (90%)",
      "page_refs": [2],
      "confidence": 0.86,
      "rationale": "Explicit statement in pathology section."
    }
  ]
}

3.7 RAG Verification Loop (v3+)

After initial extraction, a verification step re-queries the vector store to confirm
the extracted value is traceable to a source chunk:

1. For each extracted feature with confidence < 0.8, or for all safety-critical features
   (receptor_status, invasive_component_size):
   - Query: "Find the exact statement about {feature} being '{extracted_value}'"
   - Retrieve top-5 chunks
   - If the model can quote a verbatim supporting sentence → confidence = 1.0
   - If no supporting chunk found → override value to "Not reported", confidence = 0.0

Purpose: Directly measures whether an extraction is fabricated (unverifiable = fabrication).

3.8 Self-Consistency Check (v3+, Critical Features Only)

For safety-critical features (FEATURE 11: invasive_component_size, FEATURE 13: receptor_status):

1. Run extraction 3× independently (temperature > 0 to introduce variation)
2. If all 3 runs agree on the same value → confidence = 1.0
3. If any run disagrees → value = "Indeterminate", flag for human review

Trade-off: 3× token cost for selected features only (~15% total increase).
Catches fabrications because hallucinated values are unlikely to be consistent across runs.

3.9 RAG Configuration Recommendations

| Parameter | Setting | Rationale |
|---|---|---|
| chunk_size | 1000 tokens | Medical text is dense; needs context |
| chunk_overlap | 200 tokens | Features spanning chunk boundaries |
| top_k retrieval | 3–5 chunks | 1 insufficient; >5 adds hallucination risk |
| embedding_model | text-embedding-3-large | Clinical terminology requires dense embeddings |
| vector_store | FAISS | Fast, local; ideal for batch 200 cases |

4. Frozen Prompt Registry (Required)
Directory
prompts/
  frozen/
    P1_initial.txt
    P2_refined.txt
    P3_rag_structured.txt
  metadata.json
Rules

Frozen prompts are immutable.

Any modification → new prompt ID.

Prompt ID referenced in run logs.

Prompt escalation documented (problem → change → hypothesis).

5. Experiment Run Logging (Required)

Each experiment produces an immutable log entry.

5.1 Required Schema
{
  "run_id": "R23",
  "model": "gpt-4.x",
  "prompt_id": "P17",
  "approach": "RAG_structured",
  "timestamp": "2026-03-10",
  "overall_accuracy": 0.82,
  "fabrication_rate": 0.05,
  "domain_breakdown": {
    "radiology": 0.84,
    "pathology": 0.79
  }
}
5.2 Required Additional Fields

dataset hash

git commit hash

embedding model

chunk size

top_k retrieval

temperature

total elements evaluated (source==1 rows)

OCR thresholds used

6. Document Quality Assessment (Required)

OCR quality is a confounder and must be measured.

6.1 Per-Page Metrics

Variance of Laplacian (blur)

Tenengrad

RMS contrast

Intensity spread (p95 - p5)

Mean brightness

Skew angle

Resolution/DPI

6.2 Flag Definitions

Define percentile-based thresholds:

blurry = bottom 10% Laplacian variance
low_contrast = bottom 10% RMS contrast

Threshold values must be logged per dataset.

6.3 Required Case-Level JSON
{
  "case_id": "CASE_0123",
  "case_quality_summary": {
    "num_pages": 6,
    "pct_pages_blurry": 0.33,
    "pct_pages_low_contrast": 0.50,
    "worst_page_laplacian_var": 31.8
  },
  "annotation_performance": {
    "correct": 15,
    "fabrications": 1,
    "omissions": 2
  }
}
7. Interpretability Framework (Molnar-Aligned)

This project operationalizes:

Global interpretability: feature-level performance trends

Local interpretability: per-element evidence citations

Example-based explanations: retrieved chunks

Post-hoc analysis: document predictors of failure

Key analysis layers:

Which elements are most fragile?

Which document features predict fabrication?

Does RAG reduce hallucination?

Does OCR quality predict omission?

8. Predictive Modeling Layer
8.1 Binary Classification Models (NB10 §10.5)

Goal:
Predict probability of each extraction outcome per observation.

Three binary models:
- Model A: P(correct extraction)
- Model B: P(omission)
- Model C: P(fabrication)

Models: Logistic Regression + Gradient Boosting (5-fold stratified CV)

Evaluation: AUC, Brier score, accuracy

Output: binary_model_results.csv

8.2 Multi-Embedding × Multi-Algorithm Comparison (NB10 §10.6)

Goal:
Systematically compare embedding representations and classification algorithms across all three outcomes.

Embeddings tested:
- OpenAI text-embedding-3-large → PCA(50)
- TF-IDF (5000 features, 1–2 grams) → TruncatedSVD(100)
- Sentence-BERT (all-MiniLM-L6-v2, 384-dim)

Algorithms tested:
- Logistic Regression (C=0.1, balanced class weights)
- Gradient Boosting (200 trees, depth=3, lr=0.05)
- SVM (CalibratedLinearSVC, balanced weights)
- TabNet (n_d=64, n_a=64, entmax masking — optional)

Evaluation: 5-fold stratified CV; AUC, accuracy, F1 per combination

Outputs: comprehensive_model_comparison.csv, per-outcome AUC bar plots and heatmaps

8.3 Element-Level Regression (NB10 §10.7)

Goal:
Predict mean element-level accuracy, omission_rate, fabrication_rate.

Model: Multi-output Ridge regression (leave-one-element-out CV)

Output: model_b_regression_results.csv, model_b_multivariate_regression.png

8.4 Prompt Version Regression

Goal:
Predict mean overall AI metric by prompt version.

Outcome:
Mean accuracy or mean fabrication rate per run.

Used for:

Prompt optimization objective

Loss monitoring

9. Time-Series Forecasting

Track over runs:

Fabrication rate

Accuracy

Per-element fabrication

Domain-specific trends

Requirements:
Prompt versions must be sequential and hypothesis-driven.

Forecasting model:

ARIMA or Prophet

Detect drift

Trigger alerts

10. Folder Structure
data/
  raw/
  deidentified/
  processed/

prompts/
  frozen/
  library/

runs/
  logs/
  metrics/

models/
  configs/

eval/
  metrics_utils.py
  main_analysis.py

notebooks/
  rag_driver.ipynb
  forecasting.ipynb

docs/
  protocol.md
11. Full Pipeline Architecture

Raw scanned PDFs

OCR + image quality scoring

Deidentification

Chunking

Embeddings

Vector store

Per-element RAG extraction

Structured JSON output

Validation comparison

Metrics computation

Run logging

Forecasting + alerting

12. Scientific Contributions

This project produces:

Feature-level fabrication analysis in clinical summarization

Prompt-technique comparative effectiveness study

OCR quality as performance confounder analysis

Retrieval-based hallucination mitigation evaluation

Reproducible monitoring framework

13. Publication Alignment (npj Digital Medicine)

Study type:
Retrospective diagnostic accuracy evaluation

Index test:
LLM validation via structured RAG prompting

Reference standard:
Human validation of source-document feature presence

Primary endpoint:
Fabrication rate

Secondary:
Accuracy, omission, agreement, domain breakdown

Required reporting:

Flow diagram

Annotator training

Error taxonomy

OCR quality adjustment

Statistical analysis plan

Data availability statement

---

## Appendix A: Evaluation Framework — Metric Definitions

Confusion-matrix components are computed per element per annotator:

```
TP = ((data[source_col] == 1) & (data[annotator_col] == 1)).sum()
FN = ((data[source_col] == 1) & (data[annotator_col] == 2)).sum()
FP = ((data[source_col] == 0) & (data[annotator_col] == 3)).sum()
TN = ((data[source_col] == 0) & (data[annotator_col] == "N/A")).sum()
```

| Symbol | Interpretation |
|--------|----------------|
| TP | Correctly extracted feature present in source |
| FN | Omission — source present, not extracted |
| FP | Fabrication — source absent, extracted anyway |
| TN | Correctly not extracted |

**Primary safety outcome:**

```
Fabrication rate = FP / (FP + TN)
```

Statistical tests: One-sided exact McNemar (binomial on discordant pairs, `alternative="greater"`) testing H1: AI metric > Human metric for diagnostic metrics, and H1: AI fabrication rate > Human fabrication rate for fabrication.

## Appendix B: Source Document Feature Analysis

For each case, the following document-level features should be computed from OCR-extracted text:

- Number of reports per case
- Total token count / average tokens per report
- Unique token ratio
- Lexical diversity (type-token ratio)
- Embedding variance (sentence-transformer, BERT, or OpenAI `text-embedding-3-large` — preferred for OCR text vectorization based on observed performance)
- Negation frequency
- Uncertainty language frequency (e.g., "possible", "cannot exclude")
- Table density (structured vs free-text ratio)
- Cross-document semantic similarity (cosine similarity of document embeddings)

These features serve as predictors in downstream modeling of AI fabrication and omission.

## Appendix C: Deidentification Pipeline for Scanned PDFs

**Purpose:** OCR-based bounding-box redaction with black rectangles, applied to both source PDFs and the validation Excel spreadsheet.

**Dependencies:** `pymupdf`, `pytesseract`, `opencv-python`, `pillow`, `pandas`, `tqdm`

**Redaction rules (HIPAA-aligned):**
- Email, phone, SSN, dates, ZIP codes, URLs, address patterns
- MRN (context-based, with label lookahead)
- Contextual: if token matches a PHI label (Name, Patient, DOB, MRN), redact next N tokens on same line

**Key configuration:**
```python
CONTEXT_LABELS = {"name", "patient", "dob", "dateofbirth", "birth", "mrn", "acct", "account"}
redact_after_label_tokens = 6
```

**Patient mapping requirement:** Deidentified outputs must retain a `case_id ↔ original_filename` mapping table (stored separately in a secure location) to enable downstream observation-level linkage.

**Outputs:** Redacted PDFs + CSV redaction log (file, page, status, redaction count, rules applied).

## Appendix D: Interpretability Framework (Molnar-Aligned)

This project operationalizes four levels of interpretability:

1. **Global interpretability** — feature-level performance trends across all elements and domains
2. **Local interpretability** — per-element evidence citations from RAG retrieval chunks
3. **Example-based explanations** — retrieved source chunks that grounded each extraction
4. **Post-hoc analysis** — document-quality and text-complexity predictors of AI failure

Key analysis questions:
- Which elements are most fragile (highest fabrication rate)?
- Which document features predict fabrication?
- Does RAG reduce hallucination vs full-document prompting?
- Does OCR quality predict omission rate?

## Appendix E: Predictive Modeling Layer

### E.1 Binary Classification

**Goal:** Predict probability of correct extraction (label = 1) per observation.

**Models:** H2O AutoML, XGBoost, GBM, HistGradientBoostingClassifier

**Evaluation:** Brier score, calibration curves, ROC-AUC, feature importance (gain, SHAP)

### E.2 Multi-Embedding × Multi-Algorithm Comparison (NB10 §10.6)

**Goal:** Systematically compare embedding representations and classification algorithms for predicting all three extraction outcomes.

**Embeddings:** OpenAI text-embedding-3-large PCA(50), TF-IDF+SVD(100), Sentence-BERT (all-MiniLM-L6-v2)

**Algorithms:** Logistic Regression (balanced, C=0.1), Gradient Boosting (200 trees, depth=3), SVM (CalibratedLinearSVC), TabNet (n_d=64, n_a=64, entmax)

**Evaluation:** 5-fold stratified CV; AUC, accuracy, F1 per combination; best performer identification per outcome.

### E.3 Regression

**Goal:** Predict mean AI metric (accuracy or fabrication rate) per prompt version / run.

**Use cases:** Prompt optimization objective, loss monitoring, drift detection.

### E.4 Feature Interaction Analysis

**Goal:** Identify which combinations of AI element-level correctness features best predict overall accuracy and fabrication outcomes.

**Method:** H2O XGBoost/GBM tree-based feature interaction extraction (gain, FScore, wFScore).

**Outputs:** `feature_interactions_summary.csv`, interaction visualizations, markdown report

---

## Appendix F: Project Code Map

### F.1 Notebooks (`notebooks/`)

```
notebooks/
├── 01_deidentification.ipynb
│   ├── RedactionRule                        — dataclass: regex PHI pattern
│   ├── DeidConfig                           — dataclass: OCR/redaction settings
│   ├── pil_to_cv(), cv_to_pil()             — image format conversion
│   ├── compile_rules()                      — compile regex patterns
│   ├── ocr_tokens_with_boxes()              — pytesseract → DataFrame with bounding boxes
│   ├── get_redaction_boxes()                — match tokens to rules + contextual redaction
│   ├── apply_redactions_to_image()          — draw black rectangles over PHI
│   ├── render_pdf_page_to_pil()             — PDF page → PIL image at target DPI
│   ├── images_to_pdf()                      — stitch redacted pages back to PDF
│   ├── generate_case_id()                   — SHA-256 deterministic case_id from filename
│   ├── build_patient_mapping()              — case_id ↔ original_filename table
│   ├── deidentify_pdf_folder()              — main pipeline: OCR → redact → save
│   └── redact_cell()                        — regex redaction for Excel free-text cells
│   Outputs: deidentified PDFs, patient_case_id_mapping.csv, deid_pdf_log.csv, deidentified Excel
│
├── 02_missing_data_analysis.ipynb
│   ├── Per-feature missingness table        — n_missing, pct_missing per column
│   ├── Avg missing per obs grouped by feature
│   ├── Radiologic features missing table
│   ├── Pathologic features missing table
│   ├── Missingness heatmap (feature × annotator)
│   ├── Bar chart: Human vs AI missing rate
│   ├── Domain-level missingness summary
│   └── Per-observation missing count distribution (histogram + box plot)
│   Outputs: 5 CSVs + 3 PNGs in reports/
│
├── 03_eda_classification_diagnostic_metrics.ipynb
│   ├── Part 1: EDA — correct/omitted/fabricated per obs per feature per annotator
│   │   ├── Faceted bar charts by domain (Rad vs Path)
│   │   └── Stacked histograms per annotator
│   ├── Part 2: Classification & Diagnostic Metrics
│   │   ├── Element-level metrics (accuracy, sens, spec, PPV, NPV, fab rate + bootstrap CIs)
│   │   ├── Confusion matrix heatmaps (aggregate Human vs AI)
│   │   ├── Faceted diagnostic metrics with significance stars
│   │   └── Side-by-side count plots by domain
│   ├── Part 3: Domain-level aggregated metrics + bar chart
│   └── Part 4: Inference — one-sided McNemar p-values
│       ├── Element-level p-value table (H1: AI > Human)
│       ├── P-values stratified by domain
│       └── Fabrication rate focus table
│   Outputs: element_level_metrics.csv, element_pvalues_one_sided.csv, fabrication_rate_element_level.csv,
│            domain_level_aggregated_metrics.csv, ~8 PNGs
│
├── 04_source_doc_text_extraction.ipynb
│   ├── generate_case_id()                   — deterministic case_id
│   ├── infer_doc_type()                     — classify rad/path from filename
│   ├── build_document_mapping()             — case_id ↔ patient ↔ doc type table
│   ├── RedactionRule + redact_text()         — regex PHI removal from extracted text
│   ├── render_page_to_pil()                 — PDF page rasterization
│   ├── extract_and_deidentify_pdf()         — OCR → redact → per-page stats (confidence, word count)
│   └── Per-case text summary + quality plots
│   Outputs: extracted_text/*.txt, case_document_mapping.csv, text_extraction_log.csv, per_case_text_stats.csv
│
├── 05_feature_extraction_ocr_bert.ipynb
│   ├── Part 1: OCR Image Quality Scoring
│   │   ├── laplacian_variance()             — blur detection
│   │   ├── tenengrad()                      — gradient energy / sharpness
│   │   ├── rms_contrast()                   — RMS contrast
│   │   ├── intensity_spread()               — p95 - p5
│   │   ├── mean_brightness()
│   │   ├── estimate_skew_angle()            — Hough transform
│   │   └── compute_page_quality()           — composite per-page metrics
│   ├── Part 2: BERT Document Embeddings
│   │   └── SentenceTransformer('all-mpnet-base-v2') → 768-dim embeddings per case
│   ├── Part 3: Text-Based Document Features
│   │   └── compute_text_features()          — tokens, lexical diversity, negation/uncertainty rates
│   └── Part 4: H2O Feature Interaction Analysis
│       └── train_and_extract_interactions() — XGBoost/GBM → variable importance
│   Outputs: page_level_ocr_quality.csv, case_level_ocr_quality.csv, bert_document_embeddings.csv,
│            case_text_features.csv, case_all_features.csv, feature_interactions_summary.csv
│
├── 06_metadata_data_dictionary.ipynb
│   ├── infer_role()                         — Source / Human / AI / Covariate / ID
│   ├── infer_label()                        — human-readable label per column
│   ├── infer_description()                  — full variable description
│   ├── infer_valid_values()                 — coded values or range
│   ├── infer_data_type()                    — conceptual type
│   ├── infer_missing_code()                 — NaN / N/A conventions
│   ├── infer_notes()                        — additional context
│   └── style_excel()                        — openpyxl header/border formatting
│   Outputs: data_dictionary.xlsx (3 sheets), variable_names.xlsx (1 sheet)
│
├── 07_validation_methods_comparison.ipynb
│   ├── Part 1: Text Vectorization Benchmark (5 methods on clinical text)
│   │   ├── DictVectorizer (token-freq dicts → sparse matrix)
│   │   ├── FeatureHasher (hash trick → fixed-size vector)
│   │   ├── CountVectorizer (built-in tokenizer + word counts)
│   │   ├── HashingVectorizer (built-in tokenizer + hashing)
│   │   ├── TfidfVectorizer (TF-IDF weighted features)
│   │   └── OpenAI text-embedding-3-large (dense 3072-dim embeddings via API — planned 6th method;
│   │       demonstrated strong performance on clinical OCR text; requires OPENAI_API_KEY)
│   │   Vectorization method is a parameter — all six are benchmarked.
│   ├── Part 2: ML Validation — XGBoost classifier per vectorization method
│   │   ├── 5-fold stratified CV with early stopping
│   │   ├── Accuracy, F1, precision, recall per fold × vec method
│   │   └── Adapted from xgb_aft_preprocessing_feature_constuction_train_validate_evaluate.py
│   ├── Part 3: SHAP Feature Importance per vectorization method
│   │   ├── pred_contribs → mean |contribution| ranking
│   │   ├── Top-20 feature bar plots per vec method
│   │   └── Adapted from xgb_aft_shap_feature_importance.py + shap analysis and plot generation.R
│   ├── Part 4: Deep Learning Validation — BERT via TensorFlow Hub
│   │   ├── build_bert_classifier() — BERT encoder + classification head
│   │   ├── 5-fold stratified CV, 3-class (correct/omitted/fabricated)
│   │   └── Adapted from run_classifier_with_tfhub.py
│   ├── Part 5: Human Validation Baseline (from NB03 metrics)
│   └── Part 6: Stratified Comparison — Eval Method × Vectorization Method
│       ├── Unified comparison table (Human, LLM, ML×5 vec methods, DL/BERT)
│       ├── Accuracy bar chart + grouped metrics plot
│       ├── Performance heatmap (eval method × vec method)
│       └── Domain-stratified comparison (Radiology vs Pathology)
│   Outputs: vectorization_benchmark.csv, ml_validation_cv_results.csv, dl_validation_cv_results.csv,
│            shap_feature_rankings_by_vec_method.csv, validation_methods_comparison.csv,
│            validation_methods_by_domain.csv, ~6 PNGs
│
├── 08_ocr_image_quality_deblur.ipynb
│   ├── Part 1: Blur detection scan across all source PDFs
│   │   ├── Laplacian variance blur detector
│   │   ├── Blur score distribution and file-type comparison
│   │   └── Outputs: ocr_blur_scan_results.csv, ocr_blur_score_distribution.png
│   ├── Part 2: Deblur pipeline — before vs after OCR comparison
│   │   ├── Full preprocessing: contrast stretch → resize → denoise → threshold → sharpen → morph
│   │   ├── Word count delta and word-level precision/recall against ground truth
│   │   └── Outputs: ocr_deblur_comparison.csv
│   ├── Part 3: Aggregate quality report
│   │   ├── Histogram of word count changes; boxplots of recall/F1 delta
│   │   └── Outputs: ocr_deblur_improvement.png, ocr_quality_summary.csv
│   Core functions: detect_blur(), deblur_image(), extract_text(), ocr_metrics()
│   Dependencies: opencv-python, pytesseract, PyMuPDF, Pillow
│
├── 09_mcodegpt_dag_extraction.ipynb
│   ├── §9.1: Ontology definition — 4-level NetworkX DiGraph for 14 breast cancer elements
│   │   ├── Root → Domain (imaging/pathologic) → Sub-domain → Leaf elements
│   │   └── Node attributes: description (value extraction), description_yesno (gate queries)
│   ├── §9.2: DAG visualization — hierarchical layout, color-coded by layer
│   │   └── Output: dag_ontology_structure.png
│   ├── §9.3: Extraction engines (updated to openai v1.x)
│   │   ├── RLS (Root-to-Leaf Streamliner): all 14 leaves in one prompt
│   │   ├── BFOP (Breadth-First Ontology Pruner): layer-by-layer yes/no gating → prune absent branches
│   │   └── 2POP (Two-Phase Ontology Parser): yes/no gate on all leaves, then extract confirmed-present
│   ├── §9.4–§9.6: Run on extracted_text/CASE_*.txt, compare extraction rates by method
│   │   └── Outputs: dag_extraction_results.csv, dag_method_comparison.csv, dag_method_comparison.png
│   Hypothesis: BFOP/2POP reduce fabrication by pruning absent branches
│
└── 10_openai_predictive_model.ipynb
    ├── §10.1: Generate/cache OpenAI text-embedding-3-large embeddings (3072-dim) per case
    │   └── Output: openai_embeddings.csv
    ├── §10.2: PCA(50) reduction + scree plot → openai_pca_features.csv
    ├── §10.3: Assemble feature matrix
    │   ├── PCA embeddings + OCR quality (blur/contrast from NB08) + text complexity (NB05) + prompt one-hot
    │   └── Joins on case_id
    ├── §10.4: Build outcome variables — is_correct, is_omission, is_fabrication (observation-level)
    │   ├── Stack AI status columns to one row per (case × element)
    │   └── Element-level outcome rates (accuracy, omission_rate, fabrication_rate)
    ├── §10.5: Three binary models (Logistic Regression + Gradient Boosting, 5-fold stratified CV)
    │   ├── Model A: P(correct), Model B: P(omission), Model C: P(fabrication)
    │   ├── Metrics: AUC, Brier score, accuracy per model
    │   └── Output: binary_model_results.csv
    ├── §10.6: Multi-embedding × multi-algorithm model comparison
    │   ├── Embeddings: OpenAI PCA(50), TF-IDF+SVD(100), Sentence-BERT (all-MiniLM-L6-v2)
    │   ├── Algorithms: Logistic Regression, Gradient Boosting, SVM (CalibratedLinearSVC), TabNet
    │   ├── All 3 outcomes × 3 embeddings × 4 algorithms = up to 36 combinations
    │   ├── 5-fold stratified CV with AUC, accuracy, F1 per combination
    │   ├── Best performer identification per outcome
    │   └── Outputs: comprehensive_model_comparison.csv,
    │       model_comparison_{correct,omission,fabrication}_auc.png,
    │       model_comparison_{correct,omission,fabrication}_heatmap.png
    ├── §10.7: Model D — Element-level outcome rate prediction (multi-output Ridge regression)
    │   ├── Leave-one-element-out CV for accuracy, omission_rate, fabrication_rate
    │   ├── Predicted vs actual scatter per outcome
    │   └── Outputs: model_b_regression_results.csv, model_b_multivariate_regression.png
    └── §10.8: Summary dashboard — combined performance table
        └── Output: openai_model_summary.csv
```

### F.2 Data Sources (`data/`)

```
data/
├── raw/
│   ├── merged_llm_summary_validation_datasheet_deidentified.xlsx
│   │   200 obs × 45 cols — primary validation dataset
│   │   Columns: 14 elements × 3 roles (source/human/ai) + 2 covariates + surgeon_id
│   └── *.pdf — scanned source documents (radiology + pathology reports)
│
├── processed/
│   ├── comprehensive_enhanced_dataset_with_all_metrics.csv
│   │   Observation-level dataset with element/domain metrics, confusion status, error types
│   ├── observation_level_metrics_summary.csv
│   ├── prompt_library_updated_v4.xlsx
│   └── prompt_library_updated_v5.xlsx
│
├── deidentified/                            (generated by Notebook 01)
│   ├── pdfs/                                — redacted PDFs named by case_id
│   ├── patient_case_id_mapping.csv          — case_id ↔ original_filename
│   ├── deid_pdf_log.csv                     — per-page redaction log
│   └── validation_datasheet_deidentified.xlsx
│
├── extracted_text/                          (generated by Notebook 04)
│   └── CASE_*.txt                           — deidentified OCR text per case
│
└── features/                                (generated by Notebook 05)
    ├── page_level_ocr_quality.csv
    ├── case_level_ocr_quality.csv
    ├── bert_document_embeddings.csv
    ├── case_text_features.csv
    ├── case_all_features.csv
    └── feature_interactions_summary.csv
```

### F.3 Source Scripts (`src/`)

```
src/
├── llm_eval_by_human/
│   ├── main_analysis.py                     — primary analysis: element/domain metrics, bootstrap CIs,
│   │                                          McNemar p-values, confusion matrices, all plots/tables
│   ├── metric_utils.py                      — compute_confusion_counts, compute_metrics_from_counts,
│   │                                          bootstrap_ci, element_metric_pvalue, mcnemar_exact_from_masks,
│   │                                          metric_correct_masks, plot_confusion_heatmap
│   ├── metrics_utils.py                     — duplicate of metric_utils.py (imported by some scripts)
│   ├── human_judge_analysis_classification_metrics.py
│   │                                        — extended classification analysis with inline p-value functions
│   ├── add_observation_metrics.py           — add_observation_level_metrics, generate_observation_summary
│   ├── create_comprehensive_enhanced_dataset.py — integrates obs/element/domain metrics into one CSV
│   ├── observation_level_metrics.py         — per-row confusion status, row-level summary metrics
│   ├── main analysis.py                     — older version of main_analysis.py (deprecated)
│   └── modeling_feature_importance/
│       ├── ai_feature_interaction_analysis.py      — H2O XGBoost/GBM feature interactions
│       ├── ai_feature_interaction_clean.py         — cleaned version of interaction analysis
│       ├── ai_element_accuracy_predictors_analysis.py
│       ├── ai_fabrication_binary_analysis.py
│       ├── ai_fabrication_comprehensive_analysis.py
│       ├── ai_fabrication_predictors_analysis.py
│       ├── h2o_ai_only_feature_importance.py
│       ├── h2o_feature_importance_analysis.py
│       └── h2o_model_selection_feature_importance.py
│
├── classifier_models_prompt_optimization/
│   ├── classifiers.py                       — sklearn classifiers (RF, SVM, Logistic, etc.)
│   ├── PCA.py                               — PCA dimensionality reduction
│   ├── decision tree classifier_importance.py
│   ├── gaussian naive bayes.py
│   ├── model selection.py
│   ├── sgd_classifier.py
│   └── tsne knn classifier.py
│
├── data collection and processing - fix/
│   ├── analyze missing_descriptive analysis_descriptive plots_tables.py
│   ├── h2o_automl_advanced.py
│   ├── h2o_automl_example.py
│   ├── h2o_automl_starter.py
│   ├── h2o_local_automl.py
│   └── h2o_simple_test.py
│
└── llm_eval_by_llm/
    ├── api.py                               — LLM API calls for extraction
    ├── deep_eval_llm_judge_api.py           — DeepEval LLM-as-judge evaluation
    ├── demo_extraction.py                   — demo extraction pipeline
    └── xgb_aft_*.py                         — XGBoost AFT feature processing + training
```

### F.4 Generated Reports (`reports/`)

```
reports/
├── Tables (CSV)
│   ├── diagnostic_tests.csv / diagnostic_tests_with_p.csv
│   ├── element_level_summary_wide.csv
│   ├── domain_level_element_balanced_metrics.csv / domain_agg_metrics_with_p.csv
│   ├── confusion_human.csv / confusion_ai.csv
│   ├── comprehensive_enhanced_dataset_with_all_metrics.csv
│   └── comprehensive_dataset_column_mapping.csv / comprehensive_dataset_sample.csv
│
├── Plots (PNG)
│   ├── confusion_heatmaps.png / confusion_tables.png
│   ├── human_ai_metrics_facet_ci.png
│   ├── element_level_diagnostic_metrics_human_vs_ai.png
│   ├── domain_aggregated_diagnostic_metrics_human_vs_ai.png / domain_level_diagnostic_metrics_table.png
│   ├── avg_metrics_rad_vs_path_grouped_ci_stars.png
│   ├── fabrication_rate_element_plot.png / fabrication_rate_element_table.png
│   ├── fabrication_rate_aggregate_plots.png / fabrication_rate_aggregate_table.png / fabrication_rate_domain_table.png
│   ├── specificity_ai.png / roc_example.png / pr_example.png
│   ├── bias_variance_demo.png / cv_boxplot.png
│   └── (notebook-generated plots added on execution)
│
├── Modeling Reports (subdirectories)
│   ├── ai_element_accuracy_predictors/      — importance CSVs, PNGs, report.md
│   ├── ai_fabrication_binary/               — binary fabrication analysis outputs
│   ├── ai_fabrication_predictors/           — fabrication predictor importance
│   ├── ai_feature_interactions/             — feature interaction summary + report
│   └── ai_only_feature_importance/          — AI-only feature importance outputs
│
├── Validation Methods Comparison (Notebook 07)
│   ├── vectorization_benchmark.csv          — speed + feature count per vectorizer
│   ├── vectorization_benchmark_plot.png     — throughput + dimensionality bar charts
│   ├── ml_validation_cv_results.csv         — XGBoost 5-fold CV per vec method
│   ├── dl_validation_cv_results.csv         — BERT TF-Hub 5-fold CV results
│   ├── shap_feature_rankings_by_vec_method.csv — SHAP importance per vectorizer
│   ├── shap_feature_importance_by_vec_method.png — top-20 feature bar plots
│   ├── validation_methods_comparison.csv    — unified Human vs LLM vs ML vs DL table
│   ├── validation_methods_comparison_plot.png — accuracy bar + grouped metrics plot
│   ├── validation_methods_heatmap.png       — eval method × vec method heatmap
│   ├── validation_methods_by_domain.csv     — domain-stratified comparison
│   └── validation_by_domain_plot.png        — Radiology vs Pathology comparison
│
└── Predictive Modeling (Notebook 10)
    ├── embedding_pca_scree.png              — PCA variance explained for OpenAI embeddings
    ├── binary_model_results.csv             — LR + GB AUC/accuracy/Brier per outcome
    ├── comprehensive_model_comparison.csv   — 3 embeddings × 4 algorithms × 3 outcomes
    ├── model_comparison_correct_auc.png     — AUC bar plot (correct prediction)
    ├── model_comparison_omission_auc.png    — AUC bar plot (omission prediction)
    ├── model_comparison_fabrication_auc.png — AUC bar plot (fabrication prediction)
    ├── model_comparison_correct_heatmap.png — AUC heatmap (embedding × algorithm, correct)
    ├── model_comparison_omission_heatmap.png — AUC heatmap (embedding × algorithm, omission)
    ├── model_comparison_fabrication_heatmap.png — AUC heatmap (embedding × algorithm, fabrication)
    ├── model_b_regression_results.csv       — Ridge LOO R²/RMSE per element-level outcome
    ├── model_b_multivariate_regression.png  — predicted vs actual scatter (3 panels)
    └── openai_model_summary.csv             — combined binary + regression performance
```

---

## Appendix G: LangChain / LangGraph Architecture

### G.1 Rationale

The pipeline uses **LangChain** as the unified model abstraction layer so that each task calls the best-performing model via a consistent interface (`ChatAnthropic`, `ChatOpenAI`, `OpenAIEmbeddings`). This decouples task logic from model vendor and enables model swapping without rewriting task code.

In **production** (iOS app monitoring), the pipeline will be lifted into a **LangGraph** state machine where each task is a node and the best model (determined from NB03/NB07 benchmarks) is wired per node.

---

### G.2 Task → Model Mapping

| Task | Notebook | LangChain Class | Planned Best Model |
|------|----------|-----------------|-------------------|
| PDF text extraction (native) | NB04 | `ChatAnthropic` | Claude Sonnet 4.5 (transcription) |
| PDF text extraction (scanned OCR) | NB04 | `pytesseract` + `ChatAnthropic` vision | Claude Sonnet 4.5 vision |
| Document embeddings | NB05 | `OpenAIEmbeddings` | `text-embedding-3-large` |
| Clinical element extraction | NB04 / src/llm_eval_by_llm | `ChatAnthropic` / `ChatOpenAI` | TBD from NB07 benchmark |
| Vectorization (ML validation) | NB07 | `OpenAIEmbeddings` + sklearn | `text-embedding-3-large` (6th method) |
| Fabrication/omission prediction | NB07 | H2O AutoML / XGBoost | TBD from NB07 CV results |

---

### G.3 LangChain Integration Points (Current Notebooks)

**NB04 — Text Extraction:**
```python
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

llm = ChatAnthropic(model="claude-sonnet-4-5-20250929")
```

**NB05 — Document Embeddings (Part 2b):**
```python
from langchain_openai import OpenAIEmbeddings

embed_model = OpenAIEmbeddings(model="text-embedding-3-large")
vectors = embed_model.embed_documents(texts)   # → list of 3072-dim floats
```

**NB07 — Vectorization Benchmark (6th method):**
```python
from langchain_openai import OpenAIEmbeddings
import numpy as np

embed_model = OpenAIEmbeddings(model="text-embedding-3-large")
X_openai = np.array(embed_model.embed_documents(corpus))
```

---

### G.4 LangGraph Production Pipeline Design

```
                  ┌─────────────────────────────────┐
                  │   iOS App / REST API trigger     │
                  └────────────────┬────────────────┘
                                   │ PDF upload
                                   ▼
                  ┌────────────────────────────────────┐
  Node 1          │  ocr_quality_check                 │
                  │  pytesseract + OpenCV (NB08 logic) │
                  │  → blur_score, contrast_score      │
                  └────────────────┬───────────────────┘
                                   │ source_type routing
                        ┌──────────┴──────────┐
                        ▼                     ▼
           ┌────────────────────┐  ┌─────────────────────┐
  Node 2a  │  native_extract    │  │  ocr_extract        │  Node 2b
           │  PyMuPDF direct    │  │  pytesseract + deblur│
           │  text extraction   │  │  (NB08 pipeline)    │
           └────────┬───────────┘  └──────────┬──────────┘
                    └──────────────┬───────────┘
                                   ▼
                  ┌────────────────────────────────────┐
  Node 3          │  embed_document                    │
                  │  OpenAIEmbeddings text-embedding-  │
                  │  3-large → 3072-dim vector         │
                  └────────────────┬───────────────────┘
                                   ▼
                  ┌────────────────────────────────────┐
  Node 4          │  extract_elements                  │
                  │  ChatAnthropic / ChatOpenAI        │
                  │  Prompt version: v2/v3/v3+         │
                  │  → structured JSON per element     │
                  └────────────────┬───────────────────┘
                                   ▼
                  ┌────────────────────────────────────┐
  Node 5          │  rag_verify (v3+ only)             │
                  │  Re-query vector store per feature  │
                  │  If unverifiable → "Not reported"   │
                  │  Targets: conf < 0.8 + critical     │
                  └────────────────┬───────────────────┘
                                   ▼
                  ┌────────────────────────────────────┐
  Node 6          │  self_consistency (v3+, critical)   │
                  │  3× extraction on receptor_status,  │
                  │  invasive_component_size            │
                  │  Disagree → "Indeterminate"         │
                  └────────────────┬───────────────────┘
                                   ▼
                  ┌────────────────────────────────────┐
  Node 7          │  validate_output                   │
                  │  Rule-based schema check +         │
                  │  fabrication risk score (NB07 model)│
                  └────────────────┬───────────────────┘
                                   ▼
                  ┌────────────────────────────────────┐
  Node 8          │  return_result                     │
                  │  Structured JSON → iOS app display  │
                  │  + log to monitoring dashboard      │
                  └────────────────────────────────────┘
```

**State schema (TypedDict):**
```python
class PipelineState(TypedDict):
    pdf_path: str
    source_type: str          # native_pdf | scanned_pdf | native_docx_converted
    raw_text: str
    embedding: list[float]
    extracted_elements: dict
    verification_results: dict    # per-feature RAG verification (v3+)
    consistency_results: dict     # self-consistency check results (v3+)
    validation_flags: dict
    fabrication_risk_score: float
    prompt_version: str           # v2 | v3 | v3+
    run_metadata: dict            # model used, prompt_id, timestamp, git_hash
```

**Key LangGraph packages:**
```
langgraph>=0.2
langchain-anthropic>=0.3
langchain-openai>=0.2
langchain-core>=0.3
```

---

### G.5 Model Selection Workflow

1. Run NB07 with all 6 vectorization methods (including `text-embedding-3-large`)
2. Run NB03 across all extraction methods (pytesseract, Claude Vision, Claude Transcription, GPT-5)
3. Select best model per task based on:
   - **Embedding task:** Highest XGBoost 5-fold CV accuracy in NB07
   - **Extraction task:** Highest sensitivity for high-risk elements (receptor status, laterality) in NB03
4. Hard-code winning model into each LangGraph node for production deploy
5. Re-evaluate quarterly or on new prompt library version

---

## Appendix H: Prompt Technique Taxonomy

### H.1 Prompt Version Progression

| Version | Technique Label | Key Additions | File |
|---|---|---|---|
| v1 | Initial extraction | Basic structured prompt, JSON output | `prompts/generated/` (early drafts) |
| v2 | Zero-shot + CoT + anti-fabrication | Role assignment, 5-step reasoning framework, 6 anti-fabrication rules, confidence scoring, strict JSON schema | `prompts/updated_developer_prompt_v2.txt` |
| v3 | Few-shot enhanced | v2 + 2–3 clinical examples per modality (standard, conflict, missing data) | TBD — requires example curation |
| v3+ | Few-shot + RAG verification + self-consistency | v3 + RAG verification loop (§3.7) + self-consistency on critical features (§3.8) | TBD — requires RAG pipeline |

**Expected fabrication rate progression** (hypothetical; actual depends on report quality):
- v2 (zero-shot): 12–18%
- v3 (few-shot): 6–10%
- v3+ (few-shot + RAG + self-consistency): 3–6%

### H.2 Seven Prompting Techniques Employed

Based on the Anthropic Prompt Engineering Interactive Tutorial:

| # | Technique | Tutorial Chapter | Implementation in v2 | Purpose |
|---|---|---|---|---|
| 1 | **Role Assignment** | Ch. 3 | "You are a breast imaging + pathology information extraction system" | Narrows context; improves domain-specific accuracy |
| 2 | **Few-Shot Prompting** | Ch. 7 | v3 only (examples per modality) | Shows desired output format and edge cases |
| 3 | **Chain-of-Thought** | Ch. 6 | 5-step reasoning: Locate → Verify explicitness → Check conflicts → Preserve wording → Cite evidence | Reduces hallucination ~20–30% |
| 4 | **Separating Data from Instructions** | Ch. 4 | XML-style sections (`INSTRUCTIONS`, `EVIDENCE`, `CONFLICTING_INFORMATION`) | Prevents confusion between rules and data |
| 5 | **Formatted Output** | Ch. 5 | Strict JSON schema + confidence scores | Machine-parseable; no output ambiguity |
| 6 | **Avoiding Hallucinations** | Ch. 8 | 6 anti-fabrication rules; "Not reported" / "Indeterminate" defaults | Core safety mechanism for fabrication rate measurement |
| 7 | **Clear & Direct** | Ch. 2 | Priority order: Accuracy > Prevent Fabrication > Verbatim Fidelity > Chronology | No competing objectives |

### H.3 Advanced Reasoning Methods

| Method | Deployment | Cost | Value for Fabrication Measurement |
|---|---|---|---|
| **RAG Verification** | All features (v3+); re-query vector store to confirm traceability | ~1.5× baseline tokens | Very high — directly measures whether extraction is fabricated |
| **Self-Consistency** | Critical features only: receptor_status, invasive_component_size (v3+) | 3× for 2/13 features (~15% increase) | High — hallucinated values unlikely to be consistent across runs |
| **Tree-of-Thought** | Design phase only; used to resolve ambiguous interpretations, codified as few-shot examples | N/A at runtime | High for prompt design; zero runtime cost |
| **Meta-Reasoning / Confidence Calibration** | All features (v2+); numeric confidence 0.0–1.0 with reasoning | <1% extra tokens | High — enables quality filtering and human review flagging |

### H.4 Anti-Fabrication Framework (v2 Hard Constraints)

The following rules are enforced in all prompt versions ≥ v2:

1. Use **only** information explicitly stated in the input text. No inference.
2. If missing → output `"Not reported"`.
3. If conflicting or unclear → output `"Indeterminate"` + verbatim conflicting snippets in evidence field.
4. Preserve units and wording exactly as written. No conversion or normalization.
5. Confidence = 1.0 **only** for explicit statements. Inferred content ≤ 0.4.
6. Pre-submission execution checklist: verify no fabrications, inferences, or unwarranted interpretations.

### H.5 Recommended Method Stack by Phase

| Phase | Methods | Cases | Purpose |
|---|---|---|---|
| **Phase 1** (baseline) | v2 zero-shot + CoT + anti-fabrication | 10–20 cases | Measure zero-shot fabrication rate; document failure modes |
| **Phase 2** (enhanced) | v3 few-shot + CoT | Same cases + additional | Measure few-shot improvement; consistency gains |
| **Phase 3** (production) | v3+ few-shot + RAG verification + self-consistency | All 200 cases | Measure RAG impact on fabrication; full traceability |

### H.6 v2 Prompt Architecture Reference

Current v2 prompt (`prompts/updated_developer_prompt_v2.txt`) contains:

- **Role + priorities** — 4-level priority stack (accuracy > low fabrication > verbatim fidelity > chronology)
- **Anti-fabrication hard constraints** — 6 rules (see §H.4)
- **Modality-specific extraction rules** — mammogram, ultrasound, MRI, post-procedure mammogram, pathology, receptors
- **14 numbered features** — aligned with the 14 clinical elements in the primary dataset
- **Strict JSON output schema** — per-lesion structure with value + evidence fields
- **Value conventions** — "Not reported" for missing, "Indeterminate" + evidence for conflicts
