# LLM Summarization for Breast Cancer

Prompt-technique evaluation for feature-level **comparison of human and LLM-based clinical feature extraction** in breast cancer radiology and pathology documentation.

## Project Overview

This repository evaluates how reliably LLMs extract structured clinical features from unstructured/scanned source documents.

- **Primary goal:** reduce clinically unsafe extraction errors, especially hallucinated features.
- **Core comparison:** Human annotator vs LLM annotator against source-document ground truth.
- **Dataset scale:** 200 de-identified patient cases, 14 clinical elements, 45 core columns.
- **Primary safety metric:** fabrication rate.

## Research Questions

1. Which clinical elements are most fragile for LLM extraction?
2. Where does AI fabrication/omission differ significantly from human performance?
3. How do different prompt strategies (zero-shot, chain-of-thought, retrieval-augmented generation, few-shot, etc.) change outcomes?
4. Which document and OCR features predict AI failure?

## What Is in This Repository

- End-to-end analysis notebooks for data quality, diagnostics, modeling, and validation.
- Prompt artifacts and prompt-library resources for extraction experiments.
- Evaluation scripts for human-vs-LLM comparisons and LLM-as-judge workflows.
- Reports and figures generated from notebook and pipeline outputs.
- Supporting manuscript and conference materials.

## Repository Layout

```text
<project_root>
├── docs/                 # project docs, architecture notes, summaries, guides
├── notebooks/            # main analysis notebooks
├── src/                  # pipeline/evaluation/modeling scripts
├── prompts/              # prompt library, prompt assets, generated prompts
├── eval/                 # evaluation schemas and metric resources
├── experiments/          # run tracking structure
├── reports/              # generated tables/plots/analysis outputs
├── models/               # model configs
├── references/           # papers and technical references
└── conferences/          # conference submission material
```

## Documentation Guide

Start here based on your need:

- **High-level summary:** `docs/executive_summary.md`
- **Dataset details:** `docs/dataset_metadata.md`
- **LangGraph framework and RAG design:** `docs/langgraph_agentic_rag_design.md`
- **LangChain ecosystem architecture:** `docs/ARCHITECTURE_LANGCHAIN_ECOSYSTEM.md`
- **Metric selection decisions:** `docs/FINAL_METRICS_SELECTION.md`, `docs/hcat_metrics_selection.md`
- **Colab execution workflow:** `docs/colab_pipeline_guide.md`
- **Project structure + privacy rules:** `docs/project_directory_structure_privacy_rules.md`

## Data and Labeling Conventions

- **Source label:** feature present/absent in source documents.
- **Annotator coding:**
  - `1` = Correct extraction
  - `2` = Omission
  - `3` = Fabrication
  - `N/A` = Not applicable
- **Domains covered:** radiology and pathology elements.

## Environment and Setup

### Requirements

- Python 3.11+
- Local dependencies from `pyproject.toml` or `requirements.txt`
- API keys via `.env` (do not commit secrets)

### Installation

Using `uv`:

```bash
uv sync
```

Using `pip`:

```bash
pip install -r requirements.txt
```

## Typical Workflow

1. Prepare/confirm de-identified inputs.
2. Run notebooks in sequence for preprocessing, diagnostics, extraction analysis, and validation.
3. Generate updated outputs in `reports/`.
4. Compare metrics across prompt variants and evaluation methods.

## Privacy and Safety

- Never commit PHI or identifiable patient data.
- Keep sensitive data in controlled private storage.
- Commit only de-identified/approved derived outputs.
- Store credentials only in `.env`.

## Project Status

<<<<<<< HEAD
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

### E.2 Regression

**Goal:** Predict mean AI metric (accuracy or fabrication rate) per prompt version / run.

**Use cases:** Prompt optimization objective, loss monitoring, drift detection.

### E.3 Feature Interaction Analysis

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
│   Outputs: 5 CSVs + 3 PNGs in data reports/
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
├── 04b_text_consolidation_per_case.ipynb
│   ├── parse_patient_case_id()              — extract case folder name from original path
│   ├── parse_surgeon_name()                 — extract surgeon folder from path
│   ├── classify_doc_section()               — assign hpi/radiology/pathology/genetics per filename
│   ├── consolidate_case()                   — merge all docs for one case with section headers
│   └── Validation: spot-check + missing case audit
│   Outputs: extracted_text_consolidated/{patient_case_id}.txt, patient_case_manifest.csv,
│            data/processed/patient_case_manifest.csv, reports/consolidation_summary.png
│
├── 04c_rag_question_generation.ipynb
│   ├── generate_template_question()         — deterministic question per feature from extraction hint
│   ├── generate_case_feature_question()     — GPT-4o case-specific question per case × feature
│   ├── create_vector_store()                — OpenAI Vector Store (file_search tool)
│   ├── upload_single_txt()                  — upload consolidated .txt per case
│   ├── query_vector_store()                 — Responses API file_search, top-k retrieval
│   └── process_eval_row()                   — correct@k, reciprocal rank, average precision per question
│   Outputs: data/processed/rag_question_dataset.csv, rag_evaluation_results.csv,
│            vector_store_config.json, reports/rag_evaluation_summary.png
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
└── 07_validation_methods_comparison.ipynb
    ├── Part 1: Text Vectorization Benchmark (5 methods on clinical text)
    │   ├── DictVectorizer (token-freq dicts → sparse matrix)
    │   ├── FeatureHasher (hash trick → fixed-size vector)
    │   ├── CountVectorizer (built-in tokenizer + word counts)
    │   ├── HashingVectorizer (built-in tokenizer + hashing)
    │   └── TfidfVectorizer (TF-IDF weighted features)
    │   Vectorization method is a parameter — all five are benchmarked.
    ├── Part 2: ML Validation — XGBoost classifier per vectorization method
    │   ├── 5-fold stratified CV with early stopping
    │   ├── Accuracy, F1, precision, recall per fold × vec method
    │   └── Adapted from xgb_aft_preprocessing_feature_constuction_train_validate_evaluate.py
    ├── Part 3: SHAP Feature Importance per vectorization method
    │   ├── pred_contribs → mean |contribution| ranking
    │   ├── Top-20 feature bar plots per vec method
    │   └── Adapted from xgb_aft_shap_feature_importance.py + shap analysis and plot generation.R
    ├── Part 4: Deep Learning Validation — BERT via TensorFlow Hub
    │   ├── build_bert_classifier() — BERT encoder + classification head
    │   ├── 5-fold stratified CV, 3-class (correct/omitted/fabricated)
    │   └── Adapted from run_classifier_with_tfhub.py
    ├── Part 5: Human Validation Baseline (from NB03 metrics)
    └── Part 6: Stratified Comparison — Eval Method × Vectorization Method
        ├── Unified comparison table (Human, LLM, ML×5 vec methods, DL/BERT)
        ├── Accuracy bar chart + grouped metrics plot
        ├── Performance heatmap (eval method × vec method)
        └── Domain-stratified comparison (Radiology vs Pathology)
    Outputs: vectorization_benchmark.csv, ml_validation_cv_results.csv, dl_validation_cv_results.csv,
             shap_feature_rankings_by_vec_method.csv, validation_methods_comparison.csv,
             validation_methods_by_domain.csv, ~6 PNGs
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

### F.4 Generated Reports (`data reports/`)

```
data reports/
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
└── Validation Methods Comparison (Notebook 07)
    ├── vectorization_benchmark.csv          — speed + feature count per vectorizer
    ├── vectorization_benchmark_plot.png     — throughput + dimensionality bar charts
    ├── ml_validation_cv_results.csv         — XGBoost 5-fold CV per vec method
    ├── dl_validation_cv_results.csv         — BERT TF-Hub 5-fold CV results
    ├── shap_feature_rankings_by_vec_method.csv — SHAP importance per vectorizer
    ├── shap_feature_importance_by_vec_method.png — top-20 feature bar plots
    ├── validation_methods_comparison.csv    — unified Human vs LLM vs ML vs DL table
    ├── validation_methods_comparison_plot.png — accuracy bar + grouped metrics plot
    ├── validation_methods_heatmap.png       — eval method × vec method heatmap
    ├── validation_methods_by_domain.csv     — domain-stratified comparison
    └── validation_by_domain_plot.png        — Radiology vs Pathology comparison
```
=======
Current phase: prompt and evaluation-method optimization. Notebook outputs and reports are research-grade and may change as experiments are re-run with updated prompt variants.
>>>>>>> 740d1020037861cf7104139ecb25299bde8f60a1
