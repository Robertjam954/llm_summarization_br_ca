# Repository File Map
**llm_summarization_br_ca** — AST-style annotated structure of all files and folders.

Last updated: March 2026 | Memorial Sloan Kettering | Goel Lab

---

```
llm_summarization_br_ca/                              ← PROJECT_ROOT
│
├── .env.example                                       CONFIG  Path variables + API key template
├── .gitignore                                         CONFIG  Excludes: .env, references/, data/raw/, data_private/
├── .windsurf/
│   └── rules/
│       └── project_structure_privacy_rules.md         CONFIG  Windsurf AI rules: paths, privacy, stack
├── pyproject.toml                                     CONFIG  Project metadata, dependencies (uv)
├── requirements.txt                                   CONFIG  Pip-compatible dependency list
├── uv.lock                                            CONFIG  Locked dependency tree (uv)
├── LICENCE                                            META    MIT license
├── README.md                                          META    Project overview, structure, getting started
│
│── notebooks/                                         ← 7 main analysis notebooks (run in order)
│   ├── 01_deidentification.ipynb                      NB      PHI redaction: OCR bounding-box redaction on
│   │                                                          source PDFs; regex redaction on validation Excel
│   │                                                          RAW_DIR → DATA_PRIVATE_DIR/raw/
│   │                                                          DEID_DIR → DATA_PRIVATE_DIR/deidentified/
│   ├── 02_missing_data_analysis.ipynb                 NB      Feature- and observation-level missingness;
│   │                                                          heatmaps, bar charts, domain breakdown tables
│   ├── 03_eda_classification_diagnostic_metrics.ipynb NB      Primary analysis: EDA, confusion matrices,
│   │                                                          element/domain diagnostic metrics (Se, Sp, PPV,
│   │                                                          NPV, F1, fabrication/omission rates), bootstrap
│   │                                                          CIs, McNemar one-sided p-values
│   │                                                          Imports: src/llm_eval_by_human/metrics_utils.py
│   ├── 04_source_doc_text_extraction.ipynb            NB      Multi-method PDF text extraction: pytesseract
│   │                                                          OCR, Claude Vision API, Claude Transcription API
│   │                                                          Outputs .txt files → DATA_PRIVATE_DIR/extracted_text/
│   ├── 05_feature_extraction_ocr_bert.ipynb           NB      OCR image quality scoring (BRISQUE, Laplacian),
│   │                                                          BERT document embeddings (TF-Hub), text features
│   │                                                          (TTR, negation freq, sentence length)
│   │                                                          Outputs → data/features/
│   ├── 06_metadata_data_dictionary.ipynb              NB      Auto-generates data_dictionary.xlsx (3 sheets,
│   │                                                          styled) and variable_names.xlsx
│   └── 07_validation_methods_comparison.ipynb         NB      Vectorization benchmark (5 methods), XGBoost
│                                                              5-fold CV, BERT fine-tuning 5-fold CV, SHAP
│                                                              feature importance, stratified domain comparison
│
├── study_records/                                     ← IRB/Protocol documents (PI-reviewed)
│   ├── Hypothesis and Aims.docx                       DOC     Original hypothesis and specific aims
│   ├── Hypothesis and Aims_edited Moo.docx            DOC     PI-edited version with revisions
│   └── protocol plan.docx                             DOC     Study protocol and data collection plan
│
├── docs/                                              ← Project documentation
│   ├── executive_summary.md                           DOC     Full technical executive summary: context,
│   │                                                          objectives, key questions, dataset description
│   ├── dataset_metadata.md                            DOC     YAML-front-matter dataset specification:
│   │                                                          200 cases × 45 cols, variable definitions
│   ├── colab_pipeline_guide.md                        DOC     How to run NB02-NB07 from Google Colab with
│   │                                                          deidentified dataset; Drive setup, session cell,
│   │                                                          output persistence (push/Drive/download)
│   ├── project_directory_structure_privacy_rules.md   DOC     Mirror of .windsurf/rules/ for reference
│   ├── project_outline_variables_defined_output_format.docx  DOC  Full project outline with variable
│   │                                                              definitions and expected output format
│   ├── manuscript/
│   │   └── Project Outline_File                       DOC     Project outline and code reference map
│   └── manuscript_components/
│       ├── Abstract.docx                              DOC     Manuscript abstract draft
│       ├── Appendix_...docx                           DOC     Structured feature extraction framework appendix
│       ├── Full Developer Prompt...docx               DOC     Developer prompt documentation
│       ├── Supplementary Methods...docx               DOC     LLM system prompt supplementary methods
│       └── robert_james_cover_letter.docx             DOC     Journal submission cover letter
│
├── conferences/
│   └── acs_clinical_congress/
│       └── acs_clincal_congress_abstract_draft.docx   DOC     ACS Clinical Congress abstract submission
│
├── reports/                                           ← Committed analysis outputs (non-PHI)
│   ├── ── Diagnostic Metrics ──
│   ├── diagnostic_tests.csv                           OUT     Element-level sensitivity/specificity/PPV/NPV
│   ├── diagnostic_tests_with_p.csv                    OUT     + McNemar p-values (one-sided)
│   ├── element_level_metrics.csv                      OUT     Full element-level diagnostic + classification
│   ├── element_level_summary_wide.csv                 OUT     Wide-format summary for tables
│   ├── domain_level_aggregated_metrics.csv            OUT     Radiology vs. Pathology domain aggregates
│   ├── domain_level_element_balanced_metrics.csv      OUT     Domain metrics balanced by element count
│   ├── domain_agg_metrics_with_p.csv                  OUT     Domain aggregates + McNemar p-values
│   ├── fabrication_rate_element_level.csv             OUT     FP/(FP+TN) per element (human vs AI)
│   ├── fabrication_rate_aggregate.csv                 OUT     Overall fabrication rate summary
│   ├── fabrication_rate_domain_aggregate.csv          OUT     Domain-level fabrication rates
│   ├── fabrication_rate_debug_counts.csv              OUT     Raw TP/FP/TN/FN counts for QC
│   ├── observation_level_metrics_summary.csv          OUT     Per-case error profiles
│   ├── confusion_ai.csv                               OUT     AI confusion matrix (2×2)
│   ├── confusion_human.csv                            OUT     Human confusion matrix (2×2)
│   ├── ── Figures ──
│   ├── element_level_diagnostic_metrics_human_vs_ai.png  FIG  Faceted bar chart: all 14 elements × 6 metrics
│   ├── human_ai_metrics_facet_ci.png                  FIG     Human vs AI metrics with 95% bootstrap CIs
│   ├── human_ai_metrics_facet_ci (2).png              FIG     Alternate version / updated iteration
│   ├── avg_metrics_rad_vs_path_grouped_ci_stars.png   FIG     Radiology vs Pathology grouped bar + CI + stars
│   ├── domain_aggregated_diagnostic_metrics_human_vs_ai.png  FIG  Domain-level aggregated metrics
│   ├── confusion_heatmaps.png                         FIG     Human + AI confusion matrix heatmaps
│   ├── confusion_tables.png                           FIG     Confusion table visualization
│   ├── element_level_diagnostic_metrics_table.png     FIG     Table figure: element-level diagnostics
│   ├── element_level_classification_metrics_table.png FIG     Table figure: classification metrics
│   ├── domain_level_diagnostic_metrics_table.png      FIG     Table figure: domain diagnostics
│   ├── domain_level_classification_metrics_table.png  FIG     Table figure: domain classification
│   ├── cv_boxplot.png                                 FIG     Cross-validation results boxplot
│   ├── error_scatter.png                              FIG     Per-case error scatter plot
│   ├── bias_variance_demo.png                         FIG     Bias-variance tradeoff illustration
│   ├── pr_example.png                                 FIG     Precision-recall curve example
│   ├── roc_example.png                                FIG     ROC curve example
│   ├── bar_metrics_stratified_rads_path.html          FIG     Interactive Plotly: metrics by domain
│   ├── ── Enhanced Datasets ──
│   ├── comprehensive_enhanced_dataset_with_all_metrics.csv  OUT  Full dataset + all computed metrics
│   ├── enhanced_dataset_with_observation_metrics.csv  OUT     Observation-level features + metrics
│   ├── comprehensive_dataset_column_mapping.csv       OUT     Column name → description mapping
│   ├── comprehensive_dataset_sample.csv               OUT     10-row sample for documentation
│   └── ── Sub-reports (feature importance) ──
│       ├── ai_element_accuracy_predictors/            DIR     Feature importance for AI accuracy prediction
│       ├── ai_fabrication_binary/                     DIR     Binary fabrication outcome analysis
│       ├── ai_fabrication_predictors/                 DIR     Feature importance for fabrication prediction
│       ├── ai_feature_interactions/                   DIR     H2O feature interaction outputs
│       ├── ai_only_feature_importance/                DIR     AI-only feature importance analysis
│       └── feature_importance_analysis/               DIR     Top-level feature importance summary
│
├── data/                                              ← Non-PHI committed data
│   ├── processed/                                     DIR     Metrics CSVs, prompt library, analysis outputs
│   ├── features/                                      DIR     BERT embeddings, OCR quality, text features (NB05)
│   └── splits/                                        DIR     Train/test split definitions
│
├── src/                                               ← Source code (pre-refactor scripts + modules)
│   ├── __init__.py
│   ├── config.py                                      (empty scaffold)
│   ├── llm_eval_by_human/                             ← Primary analysis scripts
│   │   ├── metrics_utils.py                           PY      Core diagnostic metric functions: Se/Sp/PPV/NPV,
│   │   │                                                      bootstrap CIs, McNemar test — IMPORTED BY NB03
│   │   ├── metric_utils.py                            PY      Earlier version of metrics_utils.py
│   │   ├── main_analysis.py                           PY      Main analysis pipeline (pre-notebook version)
│   │   ├── main analysis.py                           PY      Alternate iteration of main analysis
│   │   ├── human_judge_analysis_classification_metrics.py  PY  Full classification + diagnostic analysis
│   │   └── human_judge_analysis_classification_metrics.ipynb  NB  Notebook version of above
│   ├── llm_eval_by_ml/                                ← ML validation scripts (NB07 precursors)
│   │   ├── xgb_aft_preprocessing_feature_constuction_train_validate_evaluate.py  PY  XGBoost AFT pipeline
│   │   ├── xgb_aft_feature_.processing_feature_processing_feature_importance.py  PY  Feature processing
│   │   ├── xgb_aft_shap_feature_importance.py         PY      SHAP feature importance for XGBoost
│   │   ├── shap analysis and plot generation.R        R       SHAP visualization in R
│   │   ├── plot_hashing_vs_dict_vectorizer.ipynb      NB      Vectorizer comparison notebook
│   │   └── text vec and judgement.ipynb               NB      Text vectorization + LLM judgment notebook
│   ├── llm_eval_by_llm/                               ← LLM extraction + evaluation pipeline
│   │   ├── source_document_feature_extraction.py      PY      v1: base Claude extraction pipeline
│   │   ├── source_document_feature_extraction_v2.py   PY      v2: improved with retry logic
│   │   ├── source_document_feature_extraction_v3.py   PY      v3: multi-doc context support
│   │   ├── source_document_feature_extraction_v3_ocr.py  PY   v3 + OCR preprocessing
│   │   ├── source_document_feature_extraction_v3_simple.py  PY  v3 simplified for speed
│   │   ├── deepeval_multi_model_pipeline.py           PY      DeepEval multi-model evaluation pipeline
│   │   ├── prompt_iteration_tracker.py                PY      Track prompt versions + metric outcomes
│   │   ├── document_similarity_analysis.py            PY      Cosine/semantic similarity between docs
│   │   ├── timeseries_prompt_forecasting.py           PY      Time-series modeling of prompt performance
│   │   ├── api.py                                     PY      Claude API wrapper (NB04 precursor)
│   │   ├── deep_eval_llm_judge_api.py                 PY      DeepEval LLM judge via API
│   │   ├── deep_eval_llm_judge.md                     DOC     DeepEval setup + usage documentation
│   │   ├── pipeline_files_summary.md                  DOC     Summary of all pipeline files
│   │   ├── check_missing_annotations.py               PY      QC: find cases missing human annotations
│   │   ├── check_source_values_zero_word.py           PY      QC: identify zero-word OCR cases
│   │   ├── check_zero_word_cases.py                   PY      QC: detailed zero-word case analysis
│   │   ├── _check_output.py                           PY      Output validation utility
│   │   ├── demo_extraction.py                         PY      Demo extraction script for testing
│   │   ├── test_similarity_analysis.py                PY      Unit tests for similarity analysis
│   │   ├── needle_haystack_context length llm accuracy visualization.ipynb  NB  Context length vs accuracy
│   │   ├── needle_haystack_llm_viz.md                 DOC     Needle-in-haystack visualization notes
│   │   ├── phoenix_prompt_tutorial.ipynb              NB      Arize Phoenix prompt monitoring tutorial
│   │   ├── requirements_pipeline.txt                  CONFIG  Pipeline-specific pip requirements
│   │   ├── shap analysis and plot generation.R        R       SHAP plots in R
│   │   ├── xgb_aft_*.py                               PY      XGBoost AFT scripts (duplicated from llm_eval_by_ml)
│   │   └── deep check.py                              PY      DeepEval quick check script
│   ├── classifier_models_prompt_optimization/         ← Classifier benchmarks
│   │   ├── classifiers.py                             PY      Multi-classifier comparison (RF, SVM, LR, XGB)
│   │   ├── PCA.py                                     PY      PCA dimensionality reduction
│   │   ├── tsne knn classifier.py                     PY      t-SNE visualization + KNN classifier
│   │   ├── decision tree classifier_importance.py     PY      Decision tree + feature importance
│   │   ├── gaussian naive bayes.py                    PY      Gaussian Naive Bayes classifier
│   │   ├── sgd_classifier.py                          PY      SGD classifier (linear models)
│   │   └── model selection.py                         PY      Model selection with cross-validation
│   ├── data collection and processing/                ← Data prep scripts
│   │   ├── deidentify_validation_datasheet.py         PY      Regex PHI redaction on validation Excel
│   │   ├── deidentify_source_doc.R                    R       PHI redaction on source docs (R version)
│   │   ├── llm_validation_collection_sheet_merge.R    R       Merge validation collection sheets
│   │   └── analyze missing_descriptive analysis...py  PY      Descriptive analysis and missingness
│   ├── prompt_eng/                                    ← Prompt engineering artifacts
│   │   ├── updated_developer_prompt_feature_extraction.txt  TXT  Current developer system prompt (plain text)
│   │   ├── Full Developer Prompt...docx               DOC     Full developer prompt with formatting
│   │   ├── Initial prompt for extraction.docx         DOC     Original extraction prompt (v1)
│   │   ├── promp draft.rtf                            DOC     Early prompt draft notes
│   │   ├── mcode_structure.xlsx                       DATA    mCODE oncology data model structure
│   │   └── mcode gpt zeroshot extraction.pdf          REF     mCODE + GPT zero-shot extraction paper
│   ├── misc_scripts/                                  ← Utility scripts from old notebooks/
│   │   ├── pdf_txt_conversion.py                      PY      PDF → plain text conversion
│   │   ├── rouge_bleu_semantic_similarity.py          PY      ROUGE/BLEU/semantic similarity metrics
│   │   ├── rouge_blue_semantic_visualization.py       PY      Visualization of ROUGE/BLEU results
│   │   ├── document_processor.py                      PY      Document preprocessing utilities
│   │   ├── shap_feature_importance_classification_prediction.py  PY  SHAP for classification
│   │   ├── vector_store_manager.py                    PY      Vector store interface (stub)
│   │   ├── vector_store_manager_embedding.py          PY      Vector store with embeddings
│   │   ├── run_mistral7b.py                           PY      Local Mistral 7B inference script
│   │   ├── xgb_binary_outcome_example.py              PY      XGBoost binary outcome example
│   │   └── test_file.py                               PY      General test/scratch script
│   ├── notebooks_legacy/                              ← Pre-refactor notebooks
│   │   ├── data_analysis_notebook.ipynb               NB      Original combined analysis notebook
│   │   ├── llm_analysis_classification_metrics.ipynb  NB      LLM classification metrics notebook
│   │   ├── starter_notebook.ipynb                     NB      Blank starter template
│   │   ├── deep_eval_llm_judge.md                     DOC     DeepEval judge documentation
│   │   └── conda_environment_instructions.md          DOC     Conda env setup instructions
│   ├── modeling/
│   │   ├── __init__.py
│   │   ├── train.py                                   (scaffold)
│   │   └── predict.py                                 (scaffold)
│   ├── services/
│   │   └── __init__.py                                (scaffold)
│   ├── data reports/                                  (scaffold — old reports scripts may go here)
│   └── exploratory data analysis/                     (scaffold)
│
├── repo_and_ide_config/                               ← IDE, git, API, MCP configuration guides
│   ├── FILE_MAP.md                                    THIS FILE
│   ├── 01_vscode_startup_guide.md                     GUIDE   Every-session VSCode startup checklist
│   ├── 02_jupyter_notebook_vscode.md                  GUIDE   Jupyter in VSCode: kernel, nbautoexport, settings
│   ├── 03_api_setup_and_usage.md                      GUIDE   Anthropic, OpenAI, DeepEval API setup + usage
│   ├── 04_mcp_integrations.md                         GUIDE   MCP servers: GitKraken, GitHub, Playwright, etc.
│   ├── 05_git_commands_reference.md                   GUIDE   Git command reference (Windows PowerShell)
│   ├── 06_github_integration.md                       GUIDE   PAT auth, VSCode GitHub, gh CLI, Colab push
│   └── vscode_config/                                 ← From github.com/Robertjam954/vs_code_ai_ds_config
│       ├── VS Code for Data and AI Projects Setup.docx  DOC   Full VSCode DS/AI setup guide
│       ├── jupyter lab kernel set up.txt              DOC     Workspace kernel registration steps
│       ├── jupyter templates.txt                      CONFIG  VSCode settings.json snippets for notebooks
│       ├── conda env local creation and activation.txt  DOC   Prefix-based conda env guide
│       ├── nbautoexport_export jupyter notebook to script.rtf  DOC  nbautoexport setup + usage
│       ├── vs code and jupyter notebook config on open.rtf  DOC  Original VSCode workflow notes
│       ├── smart conda env creation and activation.docx  DOC  Advanced conda env strategy
│       ├── conda commands.pdf                         REF     Conda command cheat sheet
│       └── activate.R                                 R       R environment activation script
│
├── foundational_knowledge/                            ← Educational reference materials
│   └── educational_materials_for_analysis_and_coding/   ← From github.com/Robertjam954/educational_materials_for_analysis_and_coding
│       ├── Hands_on_Notebook_ExploratoryDataAnalysis.ipynb  NB  Hands-on EDA notebook with real dataset
│       ├── Uber_Case_Study_1.ipynb                    NB      Uber data case study (EDA + modeling)
│       ├── lect01_handouts.pdf                        REF     Lecture 01: Introduction to Data Science
│       ├── lect02_handouts.pdf                        REF     Lecture 02: Data types and structures
│       ├── lect03_handouts.pdf                        REF     Lecture 03: Exploratory data analysis
│       ├── lect04_handouts.pdf                        REF     Lecture 04: Statistical inference
│       ├── lect05_handouts.pdf                        REF     Lecture 05: Probability and distributions
│       ├── lect06_handouts.pdf                        REF     Lecture 06: Regression methods
│       ├── lect07_handouts.pdf                        REF     Lecture 07: Classification
│       ├── lect08_handouts.pdf                        REF     Lecture 08: Model evaluation and selection
│       ├── lect09_handouts.pdf                        REF     Lecture 09: Unsupervised learning
│       ├── lect10_handouts.pdf                        REF     Lecture 10: Neural networks intro
│       ├── lect11_handouts.pdf                        REF     Lecture 11: Deep learning
│       ├── lect12_handouts.pdf                        REF     Lecture 12: NLP fundamentals
│       ├── lect13_handouts.pdf                        REF     Lecture 13: Advanced topics
│       ├── Hypothesis Testing in Data Science for Beginners.pdf  REF  Analytics Vidhya hypothesis testing
│       ├── Phases of the Data Science Life Cycle.docx REF     DS lifecycle overview document
│       ├── nbautoexport_export jupyter notebook to script.rtf  DOC  nbautoexport guide (same as vscode_config)
│       └── vs code and jupyter notebook config on open.rtf  DOC  VSCode workflow original notes
│
├── prompts/                                           ← Prompt library (version-controlled)
│   ├── prompt_library.csv                             DATA    9 prompt variants with metadata
│   ├── library/                                       DIR     Frozen prompt templates (.txt/.md)
│   └── generated/                                     DIR     Agent-derived prompts
│
├── eval/                                              ← Evaluation schemas and definitions
├── models/                                            ← Model configurations and weights
├── experiments/                                       ← Run tracking (run_id, commit, prompt_id, metrics)
│
└── tools/
    ├── colab/
    │   └── bert_finetuning_with_cloud_tpus.ipynb      NB      BERT fine-tuning on Google TPUs
    └── watch_repo.sh                                  SCRIPT  macOS: poll remote + send notification on changes
```

---

## Key Entry Points

| Goal | Start here |
|---|---|
| Run main analysis | `notebooks/03_eda_classification_diagnostic_metrics.ipynb` |
| Deidentify raw PDFs | `notebooks/01_deidentification.ipynb` |
| Extract text from source docs | `notebooks/04_source_doc_text_extraction.ipynb` |
| Generate feature matrix | `notebooks/05_feature_extraction_ocr_bert.ipynb` |
| Compare validation methods | `notebooks/07_validation_methods_comparison.ipynb` |
| Core metric functions | `src/llm_eval_by_human/metrics_utils.py` |
| Developer system prompt | `src/prompt_eng/updated_developer_prompt_feature_extraction.txt` |
| VSCode startup | `repo_and_ide_config/01_vscode_startup_guide.md` |
| Jupyter kernel setup | `repo_and_ide_config/02_jupyter_notebook_vscode.md` |
| API keys setup | `repo_and_ide_config/03_api_setup_and_usage.md` |
| MCP tools setup | `repo_and_ide_config/04_mcp_integrations.md` |
| Git commands (Windows) | `repo_and_ide_config/05_git_commands_reference.md` |
| GitHub auth + push | `repo_and_ide_config/06_github_integration.md` |
| Run from Colab | `docs/colab_pipeline_guide.md` |

---

## Private Data Paths (Never Committed)

```
C:\Users\jamesr4\loc\data_private\             ← DATA_PRIVATE_DIR
├── raw\                                        Source PDFs with PHI + validation Excel
│   └── <Surgeon>\<CaseID>\*.pdf               1,770 files across 20 surgeon folders
├── deidentified\                               Redacted PDFs + patient_case_id_mapping.csv
├── extracted_text\                             Per-case .txt files (NB04 output)
└── extracted_text_comparison\                 Method comparison text outputs
```
