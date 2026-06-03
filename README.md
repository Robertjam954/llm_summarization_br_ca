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

Current phase: prompt and evaluation-method optimization, plus build-out of HCAT safety (NB11) and LangGraph agentic RAG validation (NB12). Notebook outputs and reports are research-grade and may change as experiments are re-run with updated prompt variants.

Key analysis questions:
- Which elements are most fragile (highest fabrication rate)?
- Which document features predict fabrication?
- Does RAG reduce hallucination vs full-document prompting?
- Does OCR quality predict omission rate?
- What is the residual PHI risk after deidentification (HCAT)?
- Does LangGraph agentic RAG catch high-risk fabrications missed by single-shot extraction?

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
│
├── 08_ocr_image_quality_deblur.ipynb
│   ├── Per-page OCR image quality scoring (sharpness, contrast, skew, brightness)
│   ├── Deblurring experiments and quality-vs-extraction-error analysis
│   └── Per-document quality summary plots
│   Outputs: ocr_image_quality_log.csv, doc_text_eval_quality_plots.png
│
├── 09_mcodegpt_dag_extraction.ipynb
│   ├── mCODE-aligned structured extraction via DAG decomposition
│   ├── Element-level extraction with conditional branching per feature
│   └── Comparison of DAG vs flat extraction strategies
│   Outputs: mcodegpt_dag_extractions.csv, mcodegpt_extraction_log.csv
│
├── 10_openai_predictive_model.ipynb
│   ├── OpenAI-API-driven classifier for predicting AI extraction errors
│   ├── Calibration and uncertainty visualization
│   └── Feature-importance attribution
│   Outputs: accuracy_forecast_uncertainty.png, accuracy_vs_balanced_accuracy.csv
│
├── 11_hcat_embedding_evaluation.ipynb
│   ├── HCAT (HIPAA Compliance Assessment Tool) safety metrics
│   ├── PHI recall, false-negative rate, over-redaction rate, residual PHI risk score
│   ├── Re-identification risk via k-anonymity / l-diversity proxies
│   ├── Manual PHI annotation comparison on stratified 50-PDF sample
│   └── Per-document and aggregated safety reporting
│   Outputs: hcat_safety_metrics.csv, hcat_summary_statistics.csv,
│            hcat_residual_phi_risk_heatmap.png, hcat_phi_type_breakdown.csv
│
└── 12_langgraph_deepeval_integration.ipynb
    ├── Knowledge-graph construction (Patient / Observation / ClinicalFeature / Evidence nodes)
    ├── Evidence chunking + embedding (text-embedding-3-large, 3072-dim)
    ├── LangGraph agentic RAG nodes:
    │   ├── generate_query_or_validate
    │   ├── kg_retriever_tool
    │   ├── grade_retrieved_evidence
    │   ├── generate_validation_result (CORRECT / FABRICATION / OMISSION / UNCERTAIN)
    │   └── rewrite_query
    ├── DeepEval integration (Faithfulness, Hallucination, ContextualRecall/Relevancy/Precision,
    │   AnswerRelevancy, Toxicity, PIILeakage, TaskCompletion, DAGMetric)
    └── High-risk fabrication validation workflow (fabrication_rate > 0.15 from NB03)
    Outputs: data/knowledge_graph/clinical_kg.graphml, data/knowledge_graph/evidence_embeddings.npy,
             langgraph_fabrication_validation.csv, langgraph_fabrication_correction_rate.csv,
             langgraph_evidence_quality_scores.csv, langgraph_validation_confusion_matrix.png
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
│   │                                          McNemar p-values, confusion matrices, plots/tables
│   ├── metric_utils.py                      — compute_confusion_counts, compute_metrics_from_counts,
│   │                                          bootstrap_ci, element_metric_pvalue, mcnemar_exact_from_masks,
│   │                                          metric_correct_masks, plot_confusion_heatmap
│   ├── metrics_utils.py                     — alias module (imported by some scripts)
│   ├── human_judge_analysis_classification_metrics.py / .ipynb
│   │                                        — extended classification analysis with inline p-value functions
│   └── main analysis.py                     — older version of main_analysis.py (deprecated)
│
├── llm_eval_by_llm/
│   ├── api.py                               — LLM API calls for extraction
│   ├── deepeval_multi_model_pipeline.py     — multi-model DeepEval evaluation harness
│   ├── deep_eval_llm_judge_api.py           — DeepEval LLM-as-judge evaluation
│   ├── document_similarity_analysis.py      — pairwise document similarity for fab analysis
│   ├── feature_document_context.py          — feature-level context extraction
│   ├── source_document_feature_extraction*.py — v1/v2/v3 (OCR + simple) extraction pipelines
│   ├── apply_text_deidentification.py / simple_text_deidentification.py — text-only deid utilities
│   ├── parse_v2_summaries.py / reprocess_failed_v2_patients.py — v2 summary parsing + reruns
│   ├── prompt_iteration_tracker.py          — versioned prompt run logging
│   ├── timeseries_prompt_forecasting.py     — prompt-metric forecast modeling
│   ├── v2_prompt_fabrication_test.py        — fabrication probe on v2 prompts
│   ├── xgb_aft_*.py                         — XGBoost AFT preprocessing / SHAP / feature processing
│   ├── shap analysis and plot generation.R  — SHAP visualization (R)
│   ├── needle_haystack_*                    — context-length / NIAH analysis assets
│   ├── phoenix_prompt_tutorial.ipynb        — Arize Phoenix tracing tutorial
│   └── test_llm_app.py / test_similarity_analysis.py — pipeline tests
│
├── llm_eval_by_ml/
│   ├── plot_hashing_vs_dict_vectorizer.ipynb
│   ├── text vec and judgement.ipynb         — text vectorization + ML judge
│   ├── xgb_aft_preprocessing_feature_constuction_train_validate_evaluate.py
│   ├── xgb_aft_shap_feature_importance.py
│   ├── xgb_aft_feature_.processing_feature_processing_feature_importance.py
│   └── shap analysis and plot generation.R
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
├── prompt_eng/                              — prompt engineering scaffolds and helpers
├── modeling/                                — modeling pipeline scaffolds (train/predict)
├── data collection and processing/          — data collection + missingness analysis utilities
└── misc_scripts/
    ├── fab_page_quality_analysis.py         — per-page fab quality features (May 30 work)
    ├── document_processor.py / pdf_txt_conversion.py
    ├── rouge_bleu_semantic_similarity.py / rouge_blue_semantic_visualization.py
    ├── vector_store_manager.py / vector_store_manager_embedding.py
    ├── shap_feature_importance_classification_prediction.py
    ├── xgb_binary_outcome_example.py
    └── run_mistral7b.py
```

### F.4 Generated Reports (`reports/`)

Current contents of `reports/` (regenerated as notebooks/pipelines run):

```
reports/
├── Element / domain metrics (NB03)
│   ├── element_level_metrics.csv
│   ├── element_pvalues_one_sided.csv
│   ├── domain_level_aggregated_metrics.csv
│   ├── overall_mean_metric_paired_tests.csv
│   ├── fabrication_rate_element_level.csv
│   ├── confusion_heatmaps.png
│   ├── faceted_diagnostic_pathology.png / faceted_diagnostic_radiology.png
│   ├── domain_aggregated_diagnostic_metrics_human_vs_ai.png
│   ├── eda_counts_pathology_human_vs_ai.png / eda_counts_radiology_human_vs_ai.png
│   ├── eda_pathology_correct_omitted_fabricated.png / eda_radiology_correct_omitted_fabricated.png
│   ├── eda_obs_histogram_human.png / eda_obs_histogram_ai.png
│   └── case_accuracy_by_complexity.csv / .png
│
├── Missingness (NB02)
│   ├── missingness_per_feature_column.csv
│   ├── missingness_avg_by_feature.csv
│   ├── missingness_pathologic_features.csv / missingness_radiologic_features.csv
│   ├── missingness_domain_summary.csv
│   ├── missingness_bar_human_vs_ai.png
│   ├── missingness_heatmap_by_feature_annotator.png
│   └── missingness_per_observation_distribution.png
│
├── Source text extraction (NB04 / NB08)
│   ├── text_extraction_log.csv / text_extraction_quality.png
│   ├── per_case_text_stats.csv
│   ├── docx_to_pdf_conversion_log.csv
│   ├── doc_text_eval_quality_plots.png / doc_text_eval_summary.json
│   └── nb01_timing_log.csv
│
├── Fabrication / document quality (May 30 work)
│   ├── fab_quality_results.csv / fab_quality_stats.csv
│   ├── fab_document_quality_results.csv / fab_document_quality_stats.csv
│   └── fab_page_quality_analysis.png
│
├── Prompt iteration / extraction comparison
│   ├── prompt_history_metric_trends.png / prompt_metric_trajectories.csv
│   ├── extraction_method_comparison.csv / extraction_method_boxplots.png
│   ├── accuracy_vs_balanced_accuracy.csv / accuracy_vs_balanced_accuracy_scatter.png
│   └── accuracy_forecast_uncertainty.png
│
├── Data dictionary (NB06)
│   ├── data_dictionary.xlsx
│   └── variable_names.xlsx
│
├── HCAT safety (NB11, planned)
│   ├── hcat_safety_metrics.csv
│   ├── hcat_summary_statistics.csv
│   ├── hcat_residual_phi_risk_heatmap.png
│   └── hcat_phi_type_breakdown.csv
│
└── LangGraph agentic RAG validation (NB12, planned)
    ├── langgraph_fabrication_validation.csv
    ├── langgraph_fabrication_correction_rate.csv
    ├── langgraph_evidence_quality_scores.csv
    └── langgraph_validation_confusion_matrix.png
```
